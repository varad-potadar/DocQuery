"""
services/chunker.py

Context-aware chunker. Works for any document type — not only research papers.
Uses paragraph boundaries first, falls back to sentence splitting.
Each chunk carries positional metadata so the vector store can use it.
"""

import re
from typing import List, Dict


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines (common in PDFs after extraction cleanup)."""
    paras = re.split(r"\n{2,}", text)
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 120,
) -> List[Dict]:
    """
    Returns a list of dicts:
        {
            "text": str,
            "chunk_index": int,
            "char_start": int,
        }

    Strategy:
    1. Split into paragraphs.
    2. Pack paragraphs into chunks up to chunk_size chars.
    3. When a paragraph alone exceeds chunk_size, split it by sentences.
    4. Overlap is carried forward as a trailing slice of the previous chunk.
    """
    paragraphs = _split_paragraphs(text)

    # Flatten any paragraph that is still too large into sentence pieces
    units: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sents = _split_sentences(para)
            units.extend(sents)

    chunks: List[Dict] = []
    current_parts: List[str] = []
    current_len: int = 0
    char_cursor: int = 0

    def flush(parts: List[str], start: int) -> Dict:
        body = " ".join(parts).strip()
        return {"text": body, "chunk_index": len(chunks), "char_start": start}

    overlap_tail: str = ""
    chunk_start: int = 0

    for unit in units:
        unit_len = len(unit)

        if current_len + unit_len > chunk_size and current_parts:
            # Save chunk
            chunks.append(flush(current_parts, chunk_start))

            # Build overlap tail from end of current chunk
            combined = " ".join(current_parts)
            overlap_tail = combined[-overlap:] if len(combined) > overlap else combined

            current_parts = [overlap_tail, unit] if overlap_tail else [unit]
            current_len = len(overlap_tail) + unit_len
            chunk_start = char_cursor
        else:
            current_parts.append(unit)
            current_len += unit_len

        char_cursor += unit_len

    if current_parts:
        chunks.append(flush(current_parts, chunk_start))

    return chunks


def chunk_sections(
    sections: List[Dict],
    chunk_size: int = 600,
    overlap: int = 120,
) -> List[Dict]:
    """
    Chunk a document that's been split into sections (e.g. one per PDF
    page, or one per DOCX heading) instead of one flat string.

    Each input section is a dict like {"text": str, "page": int|None,
    "heading": str|None} — this is exactly what services/loaders
    produces for any file type. Each output chunk carries the page/
    heading of the section it came from, on top of the normal
    chunk_text() fields, so retrieval can cite "page 3" or a heading
    when it's available.

    This is a thin wrapper: it calls chunk_text() once per section and
    just re-numbers chunk_index to be continuous across the whole
    document. chunk_text() itself is unchanged.
    """
    all_chunks: List[Dict] = []

    for section in sections:
        text = (section.get("text") or "").strip()
        if not text:
            continue

        sub_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for c in sub_chunks:
            c["chunk_index"] = len(all_chunks)
            c["page"] = section.get("page")
            c["heading"] = section.get("heading")
            all_chunks.append(c)

    return all_chunks


def chunk_rows(
    rows: List[str],
    max_rows: int = 8,
    max_chars: int = 600,
) -> List[Dict]:
    """
    Packs pre-split, atomic row strings (see csv_loader.py / xlsx_loader.py
    -- each row already renders as "Column: value, Column: value") into
    chunks of up to max_rows rows or max_chars characters, whichever comes
    first.

    Deliberately no overlap: chunk_text()'s overlap exists to avoid losing
    context across an arbitrary character-level cut in flowing prose, but
    each row here is already a complete, independent record, so there's
    nothing an overlap would usefully preserve -- and naively slicing the
    last N characters of a packed chunk (as chunk_text() does) risks
    cutting a row's own text in half, mixing a fragment of one record into
    the next chunk. Every chunk this returns contains only whole rows, so
    a chunk that matches a query on one row's ID is guaranteed to also
    contain that same row's other column values.
    """
    chunks: List[Dict] = []
    current: List[str] = []
    current_len = 0

    def flush():
        if current:
            chunks.append({
                "text": "\n\n".join(current),
                "chunk_index": len(chunks),
                "char_start": 0,  # not meaningful for row-packed chunks
            })

    for row in rows:
        row_len = len(row)
        would_exceed = current and (
            current_len + row_len > max_chars or len(current) >= max_rows
        )
        if would_exceed:
            flush()
            current = []
            current_len = 0
        current.append(row)
        current_len += row_len

    flush()
    return chunks


def chunk_sections_tabular(
    sections: List[Dict],
    max_rows: int = 8,
    max_chars: int = 600,
) -> List[Dict]:
    """
    Row-aware counterpart to chunk_sections(), for sources where each
    section's text is a blank-line-joined list of independent rows --
    CSV, XLSX (see services/loaders/csv_loader.py and xlsx_loader.py,
    which already join one row per paragraph exactly like chunk_text()
    expects). Splits each section back into its rows with the same
    paragraph splitter chunk_text() uses, then packs them with
    chunk_rows() instead of chunk_text()'s character-budget prose packer,
    so a chunk never mixes a partial row with the next chunk.

    Same output contract as chunk_sections(): each chunk carries the
    page/heading of the section it came from (so e.g. an XLSX sheet name
    still comes through as the heading), with chunk_index numbered
    continuously across the whole document.
    """
    all_chunks: List[Dict] = []

    for section in sections:
        text = (section.get("text") or "").strip()
        if not text:
            continue

        rows = _split_paragraphs(text)
        sub_chunks = chunk_rows(rows, max_rows=max_rows, max_chars=max_chars)
        for c in sub_chunks:
            c["chunk_index"] = len(all_chunks)
            c["page"] = section.get("page")
            c["heading"] = section.get("heading")
            all_chunks.append(c)

    return all_chunks


def contextual_text(chunk: Dict, doc_title: str) -> str:
    """
    Text for embedding/BM25 indexing ONLY -- never for what's shown to
    the user or sent to the LLM as context (chunk["text"] stays clean
    for that; see qa_engine._build_context_block).

    Prefixes the chunk with its doc title + section heading (when the
    loader found one) so a chunk about e.g. renewal terms sitting under
    a "Termination" heading is retrievable by a query like
    "termination", even though that word may never appear in the chunk
    body itself.
    """
    parts = [doc_title] if doc_title else []
    heading = chunk.get("heading")
    if heading:
        parts.append(heading)

    if not parts:
        return chunk["text"]

    prefix = " > ".join(parts)
    return f"{prefix}\n{chunk['text']}"
