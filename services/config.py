"""
services/config.py

Small, dependency-free constants shared across services.

Kept in their own file (instead of living inside embedder.py) so that
lightweight modules — like the on-disk cache — don't have to import
sentence-transformers (a heavy ML library) just to read a number.

If you ever swap the embedding model, change EMBEDDING_MODEL_NAME and
EMBEDDING_DIM here. Bump CACHE_SCHEMA_VERSION whenever chunking or
preprocessing changes in a way that makes previously cached chunks/
embeddings untrustworthy — that guarantees an old cache is never
silently reused after such a change.
"""

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CACHE_SCHEMA_VERSION = 1

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
