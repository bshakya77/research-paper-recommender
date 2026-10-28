"""
src/ingestion/hf_feed.py — Stage 1
Fetch Hugging Face Daily Papers metadata and normalize to `Paper` objects.
Endpoint: https://huggingface.co/api/daily_papers?date=YYYY-MM-DD
"""

import logging
from datetime import date, timedelta

import requests

from src.config import settings
from src.models import Paper

logger = logging.getLogger(__name__)
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.hf_api_token}"} if settings.hf_api_token else {}


def _normalize(item: dict, feed_date: date) -> Paper | None:
    """Map one raw HF record to a Paper. Returns None if essential fields are missing."""
    p = item.get("paper", {})
    pid = p.get("id")
    if not pid:
        return None
    return Paper(
        paper_id=pid,
        title=(p.get("title") or item.get("title") or "").strip(),
        authors=[a.get("name", "") for a in p.get("authors", []) if a.get("name")],
        abstract=(p.get("summary") or "").strip(),
        published_date=feed_date,
        hf_votes=p.get("upvotes", 0) or 0,
        source_url=f"https://huggingface.co/papers/{pid}",
        pdf_url=f"https://arxiv.org/pdf/{pid}",
    )


def fetch_daily_papers_for_date(feed_date: date, timeout: int = 30) -> list[Paper]:
    """Fetch papers featured on a single day."""
    resp = requests.get(HF_DAILY_URL, params={"date": feed_date.isoformat()},
                        headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    papers = [p for p in (_normalize(i, feed_date) for i in resp.json()) if p]
    logger.info("Fetched %d papers for %s", len(papers), feed_date)
    return papers


def fetch_daily_papers(days: int = 7, end_date: date | None = None) -> list[Paper]:
    """Fetch papers for the last `days` days (inclusive of end_date). Failed days are skipped."""
    end_date = end_date or date.today()
    papers: list[Paper] = []
    for offset in range(days):
        d = end_date - timedelta(days=offset)
        try:
            papers.extend(fetch_daily_papers_for_date(d))
        except requests.RequestException as e:
            logger.warning("Skipping %s: %s", d, e)
    return papers
