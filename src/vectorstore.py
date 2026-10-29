"""
src/vectorstore.py — Stage 3
FAISS index (inner product on normalized vectors = cosine) + JSON chunk metadata.
Filters (paper_id, date range, chunk_type) are applied after search with over-fetch.
Keep this small interface so Weaviate/Pinecone can be swapped in later.
"""

import json
import math
import re
from collections import Counter
from datetime import date
from functools import lru_cache

import faiss

from src.config import settings
from src.models import Chunk, RetrievedChunk


class FaissVectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self._document_tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._average_document_length = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple Unicode-aware-enough lexical tokens for BM25 matching."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _refresh_lexical_index(self) -> None:
        self._document_tokens = [
            self._tokenize(f"{chunk.paper_id} {chunk.section} {chunk.text}")
            for chunk in self.chunks
        ]
        self._term_frequencies = [Counter(document) for document in self._document_tokens]
        self._document_frequencies = Counter(
            token
            for document in self._document_tokens
            for token in set(document)
        )
        total_tokens = sum(map(len, self._document_tokens))
        self._average_document_length = total_tokens / len(self.chunks) if self.chunks else 0.0

    def build(self, chunks: list[Chunk]) -> None:
        """Replace the index with embeddings for the supplied chunks."""
        if not chunks:
            raise ValueError("Cannot build a vector index with no chunks")
        from src.embedding import embed

        vecs = embed([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks = chunks
        self._refresh_lexical_index()

    def add(self, chunks: list[Chunk]) -> list[Chunk]:
        """Append chunks only for papers that are not already indexed.

        The paper-level check avoids chunking work in the caller and this
        second check protects the store from accidentally adding a paper twice.
        Returns the chunks actually added.
        """
        existing_paper_ids = {chunk.paper_id for chunk in self.chunks}
        new_paper_ids = {chunk.paper_id for chunk in chunks} - existing_paper_ids
        new_chunks = [chunk for chunk in chunks if chunk.paper_id in new_paper_ids]
        if not new_chunks:
            return []

        from src.embedding import embed

        vecs = embed([c.text for c in new_chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(new_chunks)
        self._refresh_lexical_index()
        return new_chunks

    def save(self) -> None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(settings.index_dir / "papers.faiss"))
        (settings.index_dir / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks]))

    def load(self) -> "FaissVectorStore":
        self.index = faiss.read_index(str(settings.index_dir / "papers.faiss"))
        raw = json.loads((settings.index_dir / "chunks.json").read_text())
        self.chunks = [Chunk.model_validate(c) for c in raw]
        self._refresh_lexical_index()
        return self

    def search(self, query: str, top_k: int | None = None, paper_id: str | None = None,
               start_date: date | None = None, end_date: date | None = None,
               chunk_type: str | None = None,
               search_mode: str = "hybrid") -> list[RetrievedChunk]:
        """Search by meaning, BM25 keywords, or both (RRF fused by default)."""
        if search_mode not in {"hybrid", "semantic", "keyword"}:
            raise ValueError("search_mode must be 'hybrid', 'semantic', or 'keyword'")
        top_k = top_k or settings.top_k_retrieval
        fetch = len(self.chunks) if paper_id else min(len(self.chunks), top_k * 20)
        semantic = []
        if search_mode in {"hybrid", "semantic"} and self.index is not None and fetch:
            from src.embedding import embed

            q = embed([query])
            scores, ids = self.index.search(q, fetch)
            semantic = [
                (int(i), float(score))
                for score, i in zip(scores[0], ids[0])
                if i >= 0 and self._matches_filters(
                    self.chunks[int(i)], paper_id, start_date, end_date, chunk_type
                )
            ]

        keyword = []
        if search_mode in {"hybrid", "keyword"}:
            keyword = self._keyword_search(
                query, fetch, paper_id, start_date, end_date, chunk_type
            )

        if search_mode == "semantic":
            ranked = semantic[:top_k]
        elif search_mode == "keyword":
            ranked = keyword[:top_k]
        else:
            # RRF combines ranks instead of raw BM25 and cosine scores, whose
            # scales are not comparable. Ranks are one-based.
            fused_scores: dict[int, float] = {}
            for candidates in (semantic, keyword):
                for rank, (chunk_index, _) in enumerate(candidates, start=1):
                    fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1.0 / (60 + rank)
            ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(chunk=self.chunks[i], score=score)
            for i, score in ranked
        ]

    @staticmethod
    def _matches_filters(chunk: Chunk, paper_id: str | None,
                         start_date: date | None, end_date: date | None,
                         chunk_type: str | None) -> bool:
        return not (
            (paper_id and chunk.paper_id != paper_id)
            or (chunk_type and chunk.chunk_type != chunk_type)
            or (start_date and chunk.published_date < start_date)
            or (end_date and chunk.published_date > end_date)
        )

    def _keyword_search(self, query: str, fetch: int, paper_id: str | None,
                        start_date: date | None, end_date: date | None,
                        chunk_type: str | None) -> list[tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._document_tokens:
            return []

        document_count = len(self.chunks)
        k1, b = 1.5, 0.75
        scored = []
        for i, (chunk, term_counts) in enumerate(zip(self.chunks, self._term_frequencies)):
            if not self._matches_filters(chunk, paper_id, start_date, end_date, chunk_type):
                continue
            doc_length = sum(term_counts.values())
            score = 0.0
            for term in query_tokens:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                doc_frequency = self._document_frequencies[term]
                inverse_frequency = math.log(
                    1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * doc_length / max(self._average_document_length, 1.0)
                )
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:fetch]


@lru_cache(maxsize=1)
def get_vectorstore() -> FaissVectorStore:
    """Load the persisted index once per process for UI and chat requests."""
    return FaissVectorStore().load()


def clear_vectorstore_cache() -> None:
    """Drop the loaded index after an external rebuild or explicit UI refresh."""
    get_vectorstore.cache_clear()
