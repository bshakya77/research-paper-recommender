"""Paper-level cleanup for the catalog, vector index, and local source files."""

from dataclasses import dataclass, field
import re

from src.config import settings
from src.ingestion.deduplicator import load_catalog, save_catalog
from src.vectorstore import FaissVectorStore, clear_vectorstore_cache


@dataclass
class PaperDeletionReport:
    removed_chunks: int = 0
    removed_pdf: bool = False
    removed_parsed_text: bool = False
    cleanup_errors: list[str] = field(default_factory=list)


def _storage_stem(paper_id: str) -> str:
    """Return a safe filename stem matching the ingestion filename convention."""
    stem = paper_id.replace("/", "_").replace("\\", "_")
    if stem in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", stem):
        raise ValueError(f"Paper ID {paper_id!r} is not a valid stored paper ID")
    return stem


def _remove_paper_chunks(store: FaissVectorStore, paper_id: str) -> int:
    remove_indices = [
        index for index, chunk in enumerate(store.chunks)
        if chunk.paper_id == paper_id
    ]
    if not remove_indices:
        return 0
    if store.index is None:
        raise RuntimeError("Cannot remove paper chunks because the vector index is missing")

    import numpy as np

    removed_count = store.index.remove_ids(np.asarray(remove_indices, dtype="int64"))
    if removed_count != len(remove_indices):
        raise RuntimeError(
            f"FAISS removed {removed_count} vectors; expected {len(remove_indices)}"
        )
    removed = set(remove_indices)
    store.chunks = [
        chunk for index, chunk in enumerate(store.chunks)
        if index not in removed
    ]
    store._refresh_lexical_index()
    return removed_count


def delete_paper_data(paper_id: str) -> PaperDeletionReport:
    """Remove a paper from the vector index/catalog and delete local files."""
    catalog = load_catalog()
    if paper_id not in catalog:
        raise ValueError(f"Paper {paper_id!r} is not present in the catalog")

    stem = _storage_stem(paper_id)
    index_file = settings.index_dir / "papers.faiss"
    chunks_file = settings.index_dir / "chunks.json"
    if index_file.exists() != chunks_file.exists():
        raise RuntimeError("The vector index files are incomplete; paper deletion was stopped")

    report = PaperDeletionReport()
    if index_file.exists():
        store = FaissVectorStore().load()
        report.removed_chunks = _remove_paper_chunks(store, paper_id)
        if report.removed_chunks:
            store.save()

    catalog.pop(paper_id)
    save_catalog(catalog)
    clear_vectorstore_cache()

    for label, path in (
        ("PDF", settings.raw_dir / f"{stem}.pdf"),
        ("parsed text", settings.parsed_dir / f"{stem}.json"),
    ):
        try:
            if path.exists():
                path.unlink()
                if label == "PDF":
                    report.removed_pdf = True
                else:
                    report.removed_parsed_text = True
        except OSError as exc:
            report.cleanup_errors.append(f"Could not delete {label} file: {exc}")

    return report
