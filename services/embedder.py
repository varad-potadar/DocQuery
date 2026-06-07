"""
services/embedder.py

Thin wrapper around SentenceTransformers. Kept intentionally minimal.
Model: all-MiniLM-L6-v2 (384-dim, fast, good quality for semantic search)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_chunks(chunks: list) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings).astype("float32")
