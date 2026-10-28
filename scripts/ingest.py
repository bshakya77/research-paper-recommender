"""Stages 1–2 CLI: python scripts/ingest.py --days 7 --top 10 [--keyword TOPIC] [--no-pdf]"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.ingestion.deduplicator import deduplicate, load_catalog, save_catalog
from src.ingestion.hf_feed import fetch_daily_papers
from src.ingestion.pdf_downloader import download_pdf
from src.ingestion.pdf_parser import parse_pdf, save_parsed
from src.ingestion.workflow import matches_keyword, rank_papers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--top", type=int, default=10,
                    help="Download PDFs for the top N papers by Hugging Face upvotes")
    ap.add_argument("--keyword",
                    help="Only ingest papers whose title or abstract contains this keyword")
    ap.add_argument("--no-pdf", action="store_true", help="Metadata only")
    args = ap.parse_args()
    if args.top < 1:
        ap.error("--top must be at least 1")
    keyword = args.keyword.strip() if args.keyword else None
    if args.keyword is not None and not keyword:
        ap.error("--keyword must not be empty")

    logging.basicConfig(level=settings.log_level)
    settings.ensure_dirs()

    catalog = load_catalog()
    fetched = fetch_daily_papers(args.days)
    fetched_count = len(fetched)
    if keyword:
        fetched = [
            p for p in fetched
            if matches_keyword(p.title, p.abstract, keyword)
        ]
    papers = deduplicate(fetched)
    for p in papers:
        if p.paper_id not in catalog:
            catalog[p.paper_id] = p
        else:
            catalog[p.paper_id].hf_votes = max(catalog[p.paper_id].hf_votes, p.hf_votes)
            p.hf_votes = catalog[p.paper_id].hf_votes

    pending = [p for p in papers if not catalog[p.paper_id].full_text_available]
    top_papers = rank_papers(pending, args.top, ranking="votes")
    selected_ids = {p.paper_id for p in top_papers}

    filter_description = f" matching {keyword!r}" if keyword else ""
    if keyword:
        print(
            f"{len(papers)} of {fetched_count} fetched papers matched {keyword!r} "
            f"in the last {args.days} days"
        )
    else:
        print(f"{len(papers)} distinct papers from the last {args.days} days")
    if keyword and not papers:
        print("No papers matched all keyword terms; no PDFs will be downloaded.")
    if not args.no_pdf:
        print(f"Downloading PDFs for up to {len(top_papers)} papers{filter_description}")

    for p in papers:
        stored_paper = catalog[p.paper_id]
        if not args.no_pdf and p.paper_id in selected_ids:
            pdf = download_pdf(stored_paper)
            if pdf:
                try:
                    save_parsed(parse_pdf(pdf, stored_paper.paper_id, stored_paper.title))
                    stored_paper.full_text_available = True
                    stored_paper.full_text_error = None
                except Exception as e:  # record, don't crash the batch
                    stored_paper.full_text_error = f"parse_error: {e}"

    save_catalog(catalog)
    missing = [paper_id for paper_id in selected_ids if not catalog[paper_id].full_text_available]
    print(f"Catalog: {len(catalog)} papers | full text missing for {len(missing)} selected papers")


if __name__ == "__main__":
    main()
