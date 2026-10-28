"""Reusable Stage 1–2 workflows for the CLI and Streamlit UI."""

from dataclasses import dataclass
from datetime import date, timedelta
import re

import requests

from src.ingestion.deduplicator import deduplicate, load_catalog, save_catalog
from src.ingestion.hf_feed import fetch_daily_papers_for_date
from src.ingestion.pdf_downloader import download_pdf
from src.ingestion.pdf_parser import load_parsed, parse_pdf, save_parsed
from src.models import Paper


@dataclass
class PaperSearchReport:
    papers: list[Paper]
    fetched_count: int
    matched_count: int
    failed_dates: list[str]


@dataclass
class PaperIngestionReport:
    selected_count: int
    full_text_count: int
    failures: dict[str, str]


def matches_keyword(paper_title: str, abstract: str, keyword: str) -> bool:
    terms = set(re.findall(r"\w+", keyword.casefold()))
    paper_terms = set(re.findall(r"\w+", f"{paper_title}\n{abstract}".casefold()))
    return bool(terms) and terms.issubset(paper_terms)


def rank_papers(papers: list[Paper], limit: int, ranking: str = "votes") -> list[Paper]:
    """Rank papers by votes (newest fill) or newest date, matching Stage 1 rules."""
    if limit < 1:
        raise ValueError("Result limit must be at least 1")
    if ranking == "latest":
        return sorted(
            papers,
            key=lambda paper: (-paper.published_date.toordinal(), -paper.hf_votes, paper.paper_id),
        )[:limit]
    if ranking != "votes":
        raise ValueError("ranking must be 'votes' or 'latest'")

    voted = sorted(
        (paper for paper in papers if paper.hf_votes > 0),
        key=lambda paper: (-paper.hf_votes, paper.paper_id),
    )[:limit]
    selected_ids = {paper.paper_id for paper in voted}
    if len(voted) < limit:
        voted.extend(sorted(
            (paper for paper in papers if paper.paper_id not in selected_ids),
            key=lambda paper: (-paper.published_date.toordinal(), paper.paper_id),
        )[:limit - len(voted)])
    return voted


def search_daily_papers(start_date: date, end_date: date, keyword: str,
                        limit: int = 5, ranking: str = "votes") -> PaperSearchReport:
    """Search a date range and rank matching HF Daily Papers like Stage 1."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date")
    if limit < 1:
        raise ValueError("Result limit must be at least 1")
    if ranking not in {"votes", "latest"}:
        raise ValueError("ranking must be 'votes' or 'latest'")
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Enter a topic keyword")

    fetched: list[Paper] = []
    failed_dates = []
    current = start_date
    while current <= end_date:
        try:
            fetched.extend(fetch_daily_papers_for_date(current))
        except requests.RequestException:
            failed_dates.append(current.isoformat())
        current += timedelta(days=1)

    matched = [
        paper for paper in deduplicate(fetched)
        if matches_keyword(paper.title, paper.abstract, keyword)
    ]

    catalog = load_catalog()
    for paper in matched:
        existing = catalog.get(paper.paper_id)
        if existing:
            paper.hf_votes = max(paper.hf_votes, existing.hf_votes)
            # Preserve prior ingestion state while refreshing feed metadata.
            paper.full_text_available = existing.full_text_available
            paper.full_text_error = existing.full_text_error
            if load_parsed(paper.paper_id):
                paper.full_text_available = True
                paper.full_text_error = None
        catalog[paper.paper_id] = paper
    if matched:
        save_catalog(catalog)

    ranked = rank_papers(matched, limit, ranking)

    return PaperSearchReport(
        papers=ranked,
        fetched_count=len(fetched),
        matched_count=len(matched),
        failed_dates=failed_dates,
    )


def ingest_selected_papers(paper_ids: list[str]) -> PaperIngestionReport:
    """Download and parse selected papers, recording per-paper failures."""
    catalog = load_catalog()
    failures: dict[str, str] = {}
    for paper_id in dict.fromkeys(paper_ids):
        paper = catalog.get(paper_id)
        if paper is None:
            failures[paper_id] = "Paper metadata is missing from the catalog"
            continue

        parsed = load_parsed(paper_id)
        if parsed:
            paper.full_text_available = True
            paper.full_text_error = None
            continue

        pdf = download_pdf(paper)
        if not pdf:
            failures[paper_id] = paper.full_text_error or "PDF download failed"
            continue
        try:
            save_parsed(parse_pdf(pdf, paper.paper_id, paper.title))
            paper.full_text_available = True
            paper.full_text_error = None
        except Exception as exc:
            paper.full_text_available = False
            paper.full_text_error = f"parse_error: {exc}"
            failures[paper_id] = paper.full_text_error

    save_catalog(catalog)
    return PaperIngestionReport(
        selected_count=len(set(paper_ids)),
        full_text_count=sum(
            1 for paper_id in set(paper_ids)
            if paper_id in catalog and catalog[paper_id].full_text_available
        ),
        failures=failures,
    )
