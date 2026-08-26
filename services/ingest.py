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
from services.chunker import chunk_sections, chunk_sections_tabular, contextual_text
from services.embedder import embed_chunks
from services.config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_ROWS_PER_CHUNK
from services import cache

# source_type values (set by services/loaders/*.py in document["metadata"])
# that are row-structured rather than prose -- these get chunked by whole
# row instead of by character count. See chunk_sections_tabular().
TABULAR_SOURCE_TYPES = {"csv", "xlsx"}


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

        if document["metadata"].get("source_type") in TABULAR_SOURCE_TYPES:
            # Row-structured data (CSV/XLSX): chunk by whole row so a
            # chunk never mixes a partial row into the next one -- see
            # chunk_sections_tabular() for why the generic character-
            # budget packer below is the wrong tool for this.
            chunks = chunk_sections_tabular(document["sections"], max_rows=MAX_ROWS_PER_CHUNK, max_chars=CHUNK_SIZE)
        else:
            chunks = chunk_sections(document["sections"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        chunks = [c for c in chunks if is_useful_chunk(c["text"])]

        if not chunks:
            raise ValueError(f"'{filename}': no usable text could be extracted.")

        # Prefix each chunk with its doc title + section heading before
        # embedding/BM25-indexing -- NOT before showing it to the LLM, so
        # chunk["text"] itself stays untouched. This is what lets a query
        # like "termination" find a chunk about renewal terms that sits
        # under a "Termination" heading but never uses that word itself.
        # Stored on the chunk dict (like "page"/"heading" already are) so
        # it's cached alongside the chunk and reused on cache hits too.
        doc_title = document["title"]
        for c in chunks:
            c["index_text"] = contextual_text(c, doc_title)

        texts = [c["index_text"] for c in chunks]
        embeddings = embed_chunks(texts)

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
