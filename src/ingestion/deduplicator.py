"""
src/ingestion/deduplicator.py — Stage 1
Deduplicate papers by paper_id and persist the metadata catalog (data/papers.json).
"""

import json
from pathlib import Path

from src.config import settings
from src.models import Paper


def catalog_path() -> Path:
    return settings.data_dir / "papers.json"


def load_catalog() -> dict[str, Paper]:
    path = catalog_path()
    if not path.exists():
        return {}
    return {d["paper_id"]: Paper.model_validate(d) for d in json.loads(path.read_text())}


def save_catalog(catalog: dict[str, Paper]) -> None:
    catalog_path().parent.mkdir(parents=True, exist_ok=True)
    data = [p.model_dump(mode="json") for p in catalog.values()]
    catalog_path().write_text(json.dumps(data, indent=2))


def deduplicate(papers: list[Paper], existing_ids: set[str] | None = None) -> list[Paper]:
    """Drop duplicates within the batch and papers already in the catalog.
    When a paper appears twice (multiple feed days), keep the earliest date and max votes."""
    existing_ids = existing_ids or set()
    unique: dict[str, Paper] = {}
    for p in papers:
        if p.paper_id in existing_ids:
            continue
        if p.paper_id in unique:
            prev = unique[p.paper_id]
            prev.published_date = min(prev.published_date, p.published_date)
            prev.hf_votes = max(prev.hf_votes, p.hf_votes)
        else:
            unique[p.paper_id] = p
    return list(unique.values())
