"""
services/loaders/text_utils.py

Small helpers shared by the text-based loaders (txt, md, csv, html), so
each loader doesn't reimplement the same encoding fallback / title
guess logic.
"""

from typing import List

# Tried in order. Covers the large majority of real-world .txt/.md/.csv
# files without pulling in an extra dependency just for encoding
# detection.
_ENCODINGS_TO_TRY: List[str] = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def decode_bytes(file_bytes: bytes) -> str:
    """Decode raw bytes to text, trying a few common encodings first."""
    if not file_bytes:
        return ""
    for enc in _ENCODINGS_TO_TRY:
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Never crash on encoding — replace whatever doesn't decode.
    return file_bytes.decode("utf-8", errors="replace")


def guess_title(text: str, fallback: str) -> str:
    """Best-effort title: first non-empty line of the text, else fallback."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line[:200]
    return fallback
