"""
src/models.py
─────────────
Pydantic data models shared across all pipeline stages.

Keeping the corpus (Paper / ParsedPaper) separate from the vector index
(Chunk) mirrors the roadmap guidance: the corpus holds source text and
bibliographic metadata; the index holds searchable embeddings + chunk metadata.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Paper metadata (from HF Daily Papers feed)
# ─────────────────────────────────────────────────────────────────────────────
class Paper(BaseModel):
    """Normalized bibliographic record for one paper."""

    paper_id: str = Field(..., description="Canonical ID, e.g. arXiv ID '2410.12345'")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published_date: date = Field(..., description="Date featured on HF Daily Papers")
    hf_votes: int = Field(default=0, description="HF community upvotes (trending signal)")
    source_url: Optional[HttpUrl] = Field(default=None, description="HF paper page URL")
    pdf_url: Optional[HttpUrl] = Field(default=None, description="Direct PDF link (arXiv)")

    # Populated during Stage 2
    full_text_available: bool = False
    full_text_error: Optional[str] = Field(
        default=None, description="Reason the PDF could not be obtained/parsed"
    )

    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Parsed full text (structure-preserving)
# ─────────────────────────────────────────────────────────────────────────────
class PageText(BaseModel):
    """Raw text of one PDF page."""

    page_number: int = Field(..., ge=1)
    text: str


class Section(BaseModel):
    """A logical section detected in the paper (e.g. 'Introduction')."""

    heading: str
    text: str
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)


class ParsedPaper(BaseModel):
    """Full-text representation saved to data/parsed/<paper_id>.json."""

    paper_id: str
    title: str
    pages: list[PageText] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    num_pages: int = 0
    parser: str = Field(default="pymupdf", description="Which parser produced this")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Chunk (unit stored in the vector index)
# ─────────────────────────────────────────────────────────────────────────────
class Chunk(BaseModel):
    """
    A retrievable text segment. Metadata fields enable filtered retrieval:
      - paper_id   → paper-scoped chat (Stage 6)
      - published_date → date-range search (Stage 4)
      - section / page → evidence references (Stage 5)
    """

    chunk_id: str = Field(..., description="'<paper_id>::<index>'")
    paper_id: str
    text: str
    section: str = "Unknown"
    page: int = Field(default=1, ge=1)
    published_date: date
    url: Optional[str] = None
    chunk_type: str = Field(
        default="fulltext", description="'abstract' (paper-level search) or 'fulltext'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4+ — Retrieval output
# ─────────────────────────────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store along with its similarity score."""

    chunk: Chunk
    score: float


class SearchResult(BaseModel):
    """Paper-level search result aggregating its best-matching chunks."""

    paper: Paper
    score: float
    matched_chunks: list[RetrievedChunk] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6+ — Chat
# ─────────────────────────────────────────────────────────────────────────────
class EvidenceRef(BaseModel):
    """Pointer from an answer claim back to its source."""

    paper_id: str
    section: str
    page: int
    snippet: str


class PaperAnalysis(BaseModel):
    """LLM-generated analysis grounded in retrieved paper chunks."""

    paper_id: str
    title: str
    question: str
    analysis: str
    provider: str
    model: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class PaperChatResponse(BaseModel):
    """One grounded follow-up response in a paper-scoped chat session."""

    session_id: str
    paper_id: str
    title: str
    question: str
    answer: str
    provider: str
    model: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
