"""
services/ingest.py

The one place where "a file comes in, and DocQuery becomes able to
answer questions about it" happens. Previously app.py (Streamlit) and
main.py (FastAPI) each had their own copy of this logic; now both call
the two functions below instead.

process_upload()     — handle a freshly uploaded file (bytes + filename).
                        Reuses a cached, already-processed version of the
                        exact same content when one exists; otherwise
                        detects the file type, extracts, chunks, embeds,
                        and saves the result to the on-disk cache.

restore_from_cache() — load a previously processed document back into
                        the live session directly from the cache, with
                        no re-extraction, re-chunking, or re-embedding.
"""

from typing import Dict

from services.loaders import load_document, is_supported, get_extension, SUPPORTED_EXTENSIONS
from services.chunker import chunk_sections
from services.embedder import embed_chunks
from services.config import CHUNK_SIZE, CHUNK_OVERLAP
from services import cache


def is_useful_chunk(text: str) -> bool:
    """Filter out boilerplate chunks that add noise — the same rule for
    every file format, so it lives here once instead of once per format."""
    t = text.lower().strip()
    if len(t) < 20:
        return False
    noise_patterns = [
        "creative commons",
        "doi.org",
        "all rights reserved",
        "terms and conditions",
        "this page intentionally left blank",
        "table of contents",
    ]
    return not any(p in t for p in noise_patterns)


def _build_embed_text(chunk: Dict, doc_title: str) -> str:
    """
    What actually gets embedded and BM25-indexed for this chunk --
    prefixed with the document title and (when known) the section
    heading it came from, e.g. "Vendor Agreement — Termination — {text}".

    chunk["text"] itself is untouched by this and stays exactly what's
    shown to the LLM and cited in answers -- this prefix only helps
    retrieval find a chunk whose relevant words live in its title or
    heading rather than its body (a chunk about renewal sitting under a
    "Termination" heading is otherwise invisible to a query that says
    "termination").
    """
    parts = [p for p in (doc_title, chunk.get("heading")) if p]
    parts.append(chunk["text"])
    return " — ".join(parts)


def process_upload(
    file_bytes: bytes,
    filename: str,
    vector_store,
    doc_registry: Dict,
    use_cache: bool = True,
) -> Dict:
    """
    Processes one uploaded file end-to-end and adds it to the live
    vector_store + doc_registry. Returns the doc_info dict that was
    added (includes a "source" field: "cache" or "processed").

    Raises ValueError with a clear, user-facing message on any failure
    (unsupported type, corrupt/encrypted/empty file, no usable text).
    """
    if not file_bytes:
        raise ValueError(f"'{filename}' is empty.")

    if not is_supported(filename):
        supported = ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS)
        raise ValueError(f"'{filename}': unsupported file type. Supported: {supported}")

    extension = get_extension(filename)
    content_hash = cache.compute_content_hash(file_bytes)

    cached = cache.load_cached_document(content_hash) if use_cache else None

    if cached:
        chunks = cached["chunks"]
        embeddings = cached["embeddings"]
        meta = dict(cached["meta"])
        if meta.get("filename") != filename:
            cache.update_cached_filename(content_hash, filename)
            meta["filename"] = filename
        source = "cache"

    else:
        document = load_document(file_bytes, filename)

        chunks = chunk_sections(document["sections"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunks = [c for c in chunks if is_useful_chunk(c["text"])]

        if not chunks:
            raise ValueError(f"'{filename}': no usable text could be extracted.")

        doc_title = document.get("title") or filename
        for c in chunks:
            c["embed_text"] = _build_embed_text(c, doc_title)

        embeddings = embed_chunks([c["embed_text"] for c in chunks])

        meta = {"title": document["title"], **document["metadata"]}

        if use_cache:
            cache.save_document_cache(content_hash, filename, extension, chunks, embeddings, meta)

        source = "processed"

    vector_store.add(embeddings, chunks, filename)

    doc_info = {
        **meta,
        "num_chunks": len(chunks),
        "doc_id": filename,
        "filename": filename,
        "extension": extension,
        "content_hash": content_hash,
        "source": source,
    }
    doc_registry[filename] = doc_info
    return doc_info


def restore_from_cache(content_hash: str, vector_store, doc_registry: Dict) -> Dict:
    """
    Loads a previously cached document straight into the live session.
    No file bytes are needed — everything required is already on disk,
    which is the whole point of caching.
    """
    cached = cache.load_cached_document(content_hash)
    if not cached:
        raise ValueError("This cached document is no longer available or is out of date.")

    chunks, embeddings, meta = cached["chunks"], cached["embeddings"], cached["meta"]
    filename = meta["filename"]

    vector_store.add(embeddings, chunks, filename)

    doc_info = {
        **meta,
        "num_chunks": len(chunks),
        "doc_id": filename,
        "content_hash": content_hash,
        "source": "cache",
    }
    doc_registry[filename] = doc_info
    return doc_info
