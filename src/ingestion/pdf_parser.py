"""
src/ingestion/pdf_parser.py — Stage 2
Extract text with PyMuPDF while preserving page numbers and section headings.
Headings are detected by font size larger than the body text (a simple heuristic —
refine later). Output saved to data/parsed/<paper_id>.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.config import settings
from src.models import PageText, ParsedPaper, Section

HEADING_PATTERN = re.compile(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 \-:&]{2,80}$")


def _body_font_size(doc: pymupdf.Document) -> float:
    sizes = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def parse_pdf(pdf_path: Path, paper_id: str, title: str = "") -> ParsedPaper:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    body = _body_font_size(doc)
    pages: list[PageText] = []
    sections: list[Section] = []
    cur = {"heading": "Front Matter", "text": [], "start": 1}

    for pno, page in enumerate(doc, start=1):
        pages.append(PageText(page_number=pno, text=page.get_text()))
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text:
                    continue
                size = max(s["size"] for s in line["spans"])
                if size > body + 0.5 and HEADING_PATTERN.match(text):
                    if cur["text"]:
                        sections.append(Section(heading=cur["heading"], text=" ".join(cur["text"]),
                                                start_page=cur["start"], end_page=pno))
                    cur = {"heading": text, "text": [], "start": pno}
                else:
                    cur["text"].append(text)
    if cur["text"]:
        sections.append(Section(heading=cur["heading"], text=" ".join(cur["text"]),
                                start_page=cur["start"], end_page=len(doc)))
    return ParsedPaper(paper_id=paper_id, title=title, pages=pages,
                       sections=sections, num_pages=len(doc))


def save_parsed(parsed: ParsedPaper) -> Path:
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.parsed_dir / f"{parsed.paper_id.replace('/', '_')}.json"
    out.write_text(json.dumps(parsed.model_dump(mode="json"), indent=2))
    return out


def load_parsed(paper_id: str) -> ParsedPaper | None:
    path = settings.parsed_dir / f"{paper_id.replace('/', '_')}.json"
    return ParsedPaper.model_validate_json(path.read_text()) if path.exists() else None
