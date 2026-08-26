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

CACHE_SCHEMA_VERSION = 3  # v3: CSV/XLSX now chunked row-aware via chunk_sections_tabular() instead of the generic character-budget packer

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

MAX_ROWS_PER_CHUNK = 8  # cap for chunk_sections_tabular() (CSV/XLSX) -- see services/chunker.py

# How many past conversations the sidebar shows (most recently active
# first). A simple cap, not pagination — keeps the sidebar bounded
# without added UI complexity.
MAX_CONVERSATIONS_LISTED = 30
CONVERSATION_TITLE_MAX_LEN = 60
