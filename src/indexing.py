"""Reusable incremental FAISS indexing workflow for CLI and UI callers."""

from dataclasses import dataclass

from src.config import settings
from src.ingestion.deduplicator import load_catalog
from src.ingestion.pdf_parser import load_parsed
from src.models import Chunk
from src.processing.chunker import chunk_paper
from src.vectorstore import FaissVectorStore, clear_vectorstore_cache


@dataclass
class IndexBuildReport:
    requested_papers: int
    added_papers: int
    added_chunks: int
    total_papers: int
    total_chunks: int


def build_index(paper_ids: list[str] | None = None, rebuild: bool = False) -> IndexBuildReport:
    """Index all catalog papers or only the requested IDs, skipping existing IDs by default."""
    catalog = load_catalog()
    if not catalog:
        raise ValueError("No papers found — run ingestion first")

    selected_ids = list(dict.fromkeys(paper_ids if paper_ids is not None else catalog.keys()))
    missing_ids = [paper_id for paper_id in selected_ids if paper_id not in catalog]
    if missing_ids:
        raise ValueError(f"Paper IDs are missing from the catalog: {', '.join(missing_ids)}")
    if not selected_ids:
        raise ValueError("Select at least one paper to index")

    index_exists = (
        (settings.index_dir / "papers.faiss").exists()
        and (settings.index_dir / "chunks.json").exists()
    )
    if index_exists and not rebuild:
        store = FaissVectorStore().load()
        indexed_ids = {chunk.paper_id for chunk in store.chunks}
    else:
        store = FaissVectorStore()
        indexed_ids = set()

    new_ids = [paper_id for paper_id in selected_ids if paper_id not in indexed_ids]
    chunks: list[Chunk] = []
    for paper_id in new_ids:
        paper = catalog[paper_id]
        chunks.extend(chunk_paper(paper, load_parsed(paper_id)))

    if chunks:
        if rebuild or not index_exists:
            store.build(chunks)
        else:
            store.add(chunks)
        store.save()
        clear_vectorstore_cache()

    total_papers = len({chunk.paper_id for chunk in store.chunks})
    return IndexBuildReport(
        requested_papers=len(selected_ids),
        added_papers=len(new_ids),
        added_chunks=len(chunks),
        total_papers=total_papers,
        total_chunks=len(store.chunks),
    )
