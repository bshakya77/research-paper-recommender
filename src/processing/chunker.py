"""
src/processing/chunker.py — Stage 3
Split papers into overlapping chunks with paper_id/date/section/page/url metadata.
Uses word counts as a lightweight proxy for tokens.
"""

from src.config import settings
from src.models import Chunk, Paper, ParsedPaper


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def abstract_chunk(paper: Paper) -> Chunk:
    """One chunk per paper (title + abstract) for paper-level topic search."""
    return Chunk(chunk_id=f"{paper.paper_id}::abstract", paper_id=paper.paper_id,
                 text=f"{paper.title}\n\n{paper.abstract}", section="Abstract", page=1,
                 published_date=paper.published_date,
                 url=str(paper.source_url) if paper.source_url else None, chunk_type="abstract")


def chunk_paper(paper: Paper, parsed: ParsedPaper | None,
                size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Return the abstract chunk plus full-text chunks (section-aware) when available."""
    size, overlap = size or settings.chunk_size, overlap or settings.chunk_overlap
    chunks = [abstract_chunk(paper)]
    if not parsed:
        return chunks
    idx = 0
    for sec in parsed.sections:
        for piece in _split_words(sec.text, size, overlap):
            chunks.append(Chunk(chunk_id=f"{paper.paper_id}::{idx}", paper_id=paper.paper_id,
                                text=piece, section=sec.heading, page=sec.start_page,
                                published_date=paper.published_date,
                                url=str(paper.pdf_url) if paper.pdf_url else None))
            idx += 1
    return chunks
