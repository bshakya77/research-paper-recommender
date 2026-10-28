"""src/embedding.py — Stage 3: Sentence Transformers wrapper (normalized vectors for cosine/IP search)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    # HF_API_TOKEN is the app's .env setting; forward it to the Hub client so
    # model downloads use the authenticated rate limits for HF_TOKEN.
    token = settings.hf_api_token or None
    return SentenceTransformer(settings.embedding_model_id, token=token)


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = load_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=len(texts) > 100)
    return np.asarray(vecs, dtype="float32")
