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
