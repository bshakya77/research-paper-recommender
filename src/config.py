"""
src/config.py
─────────────
Central application settings loaded from environment variables.

All modules should import `settings` from here — no scattered os.getenv calls.

Usage:
    from src.config import settings
    print(settings.groq_model_id)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field


class Settings(BaseSettings):
    """
    Reads every value from .env (or actual environment variables).
    Pydantic-settings auto-validates types and raises on missing required fields.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────
    # Optional for Stages 1–4 (ingestion/indexing don't call the LLM).
    # Required from Stage 5 onward — validated lazily by the generation layer.
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model_id: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq chat model to use for generation",
    )
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY"
        ),
        description="Google Gemini API key used as the generation fallback",
    )
    gemini_model_id: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices(
            "GEMINI_MODEL_ID", "GOOGLE_MODEL_ID", "GOOGLE_GEMINI_MODEL_ID"
        ),
        description="Gemini model to use when Groq has a retryable failure",
    )

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers model for text embeddings",
    )

    # ── Hugging Face ─────────────────────────────────────────────────
    hf_api_token: str = Field(default="", description="HF token (optional for public API)")

    # ── Vector Store ─────────────────────────────────────────────────
    vectorstore_backend: str = Field(
        default="faiss",
        description="Vector store backend: faiss | weaviate | pinecone",
    )
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="papers")
    weaviate_url: str = Field(default="")
    weaviate_api_key: str = Field(default="")

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"))
    index_dir: Path = Field(default=Path("./data/index"))
    parsed_dir: Path = Field(default=Path("./data/parsed"))
    raw_dir: Path = Field(default=Path("./data/raw"))

    # ── RAG Hyperparameters ───────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=64, description="Overlap tokens between chunks")
    top_k_retrieval: int = Field(default=5, description="Number of chunks returned per query")

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Derived helpers (not env vars) ────────────────────────────────
    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        for directory in [self.data_dir, self.index_dir, self.parsed_dir, self.raw_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this object everywhere: `from src.config import settings`
settings = Settings()
