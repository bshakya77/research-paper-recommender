"""
src/ingestion/pdf_downloader.py — Stage 2
Download accessible paper PDFs to data/raw/<paper_id>.pdf.
Failures are recorded on the Paper (full_text_error) instead of raised.
"""

import logging
import time
from pathlib import Path

import requests

from src.config import settings
from src.models import Paper

logger = logging.getLogger(__name__)
USER_AGENT = "research-paper-recommender/0.1 (educational project)"


def download_pdf(paper: Paper, delay_s: float = 3.0, timeout: int = 60) -> Path | None:
    """Return the local PDF path, or None if unavailable. `delay_s` respects arXiv rate limits."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    out = settings.raw_dir / f"{paper.paper_id.replace('/', '_')}.pdf"
    if out.exists() and out.stat().st_size > 0:
        return out  # cached
    if not paper.pdf_url:
        paper.full_text_error = "no_pdf_url"
        return None
    try:
        resp = requests.get(str(paper.pdf_url), headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", ""):
            paper.full_text_error = "not_a_pdf"
            return None
        out.write_bytes(resp.content)
        return out
    except requests.RequestException as e:
        logger.warning("Download failed for %s: %s", paper.paper_id, e)
        paper.full_text_error = f"download_error: {e}"
        return None
    finally:
        time.sleep(delay_s)
