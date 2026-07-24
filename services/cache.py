"""
services/cache.py

Server-side, on-disk cache of processed documents, keyed by a SHA-256
hash of the file's raw bytes — not by filename, since two files can
share a name and a modified file can keep its old name.

Why this instead of browser localStorage/IndexedDB:
  By the time a document is processed, this Python process already
  holds exactly what retrieval needs: chunk text and embedding vectors.
  Saving that same data to a small local folder means it can be
  reloaded exactly as-is next time — no encoding numpy arrays into
  browser storage and back, no browser storage-quota limits, and it
  works identically whether the app is opened through Streamlit or
  through the FastAPI endpoints, since it's just files on disk rather
  than something tied to one browser tab or profile.

Each cached document is one small folder:

    data/doc_cache/<sha256>/meta.json        filename, title, model info, version
    data/doc_cache/<sha256>/chunks.json      chunk text + page/heading metadata
    data/doc_cache/<sha256>/embeddings.npy   the chunk embedding matrix

A cache entry is only reused if its recorded embedding model, embedding
dimension, and CACHE_SCHEMA_VERSION all match what the running app
currently uses. If any of those differ (e.g. the embedding model or the
chunking logic changed since it was cached), it's treated as a miss and
the document is reprocessed — so a future pipeline change can never
silently load an incompatible cache.

Deliberately NOT persisted: the original uploaded file bytes. Only the
already-cleaned, already-chunked, already-embedded result is kept. This
is what's actually needed to answer questions without reprocessing, and
it keeps disk usage and the amount of raw personal-document data at
rest both smaller.
"""

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

from services.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, CACHE_SCHEMA_VERSION

CACHE_DIR = os.path.join("data", "doc_cache")

# Keys written into every meta.json that describe the cache entry itself
# (as opposed to the document's own metadata) — used to filter what
# get merged back in when only the filename needs updating.
_INTERNAL_META_KEYS = {
    "content_hash", "filename", "extension", "num_chunks",
    "cache_schema_version", "embedding_model", "embedding_dim", "cached_at",
}


def compute_content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _entry_dir(content_hash: str) -> str:
    return os.path.join(CACHE_DIR, content_hash)


def _is_compatible(meta: Dict) -> bool:
    return (
        meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION
        and meta.get("embedding_model") == EMBEDDING_MODEL_NAME
        and meta.get("embedding_dim") == EMBEDDING_DIM
    )


def load_cached_document(content_hash: str) -> Optional[Dict]:
    """
    Returns {"meta": dict, "chunks": list, "embeddings": np.ndarray} if a
    compatible cache entry exists for this content hash, else None.
    """
    entry = _entry_dir(content_hash)
    meta_path = os.path.join(entry, "meta.json")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if not _is_compatible(meta):
            return None

        with open(os.path.join(entry, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        embeddings = np.load(os.path.join(entry, "embeddings.npy"))

        return {"meta": meta, "chunks": chunks, "embeddings": embeddings}

    except Exception as e:
        print(f"[cache] Failed to load cache entry {content_hash[:8]}: {e}")
        return None


def save_document_cache(
    content_hash: str,
    filename: str,
    extension: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    extra_meta: Dict,
) -> None:
    """Writes (or overwrites) the cache entry for this content hash."""
    entry = _entry_dir(content_hash)
    os.makedirs(entry, exist_ok=True)

    meta = {
        "content_hash": content_hash,
        "filename": filename,
        "extension": extension,
        "num_chunks": len(chunks),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "cached_at": time.time(),
        **extra_meta,
    }

    with open(os.path.join(entry, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(entry, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    np.save(os.path.join(entry, "embeddings.npy"), embeddings)


def update_cached_filename(content_hash: str, filename: str) -> None:
    """
    Updates just the remembered display filename for a cache entry.
    Used when identical content is re-uploaded under a new name — cheap,
    since it rewrites only meta.json rather than the (possibly large)
    chunks/embeddings files.
    """
    entry = _entry_dir(content_hash)
    meta_path = os.path.join(entry, "meta.json")
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("filename") != filename:
        meta["filename"] = filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def list_cached_documents() -> List[Dict]:
    """Lists all cache entries, most recent first — for the 'previously
    processed documents' UI. Entries from an incompatible cache schema
    are marked but not hidden by this function (callers decide)."""
    if not os.path.isdir(CACHE_DIR):
        return []

    entries = []
    for content_hash in sorted(os.listdir(CACHE_DIR)):
        meta_path = os.path.join(_entry_dir(content_hash), "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["compatible"] = _is_compatible(meta)
            entries.append(meta)
        except Exception:
            continue

    entries.sort(key=lambda m: m.get("cached_at", 0), reverse=True)
    return entries


def delete_cached_document(content_hash: str) -> bool:
    entry = _entry_dir(content_hash)
    if os.path.isdir(entry):
        shutil.rmtree(entry)
        return True
    return False


def clear_all_cache() -> None:
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
