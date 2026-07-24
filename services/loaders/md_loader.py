"""
services/loaders/md_loader.py

Loads Markdown files. Markdown is already close to plain text, so this
only strips the most common syntax noise (heading '#'s, bullet
markers, code-fence lines) instead of fully rendering it to HTML —
that keeps this loader tiny and dependency-free.
"""

import re
from typing import Dict

from services.loaders.text_utils import decode_bytes, guess_title


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    return text


def load(file_bytes: bytes, filename: str) -> Dict:
    raw = decode_bytes(file_bytes).strip()

    if not raw:
        raise ValueError("This Markdown file appears to be empty.")

    heading_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
    title = heading_match.group(1).strip()[:200] if heading_match else guess_title(raw, filename)

    return {
        "sections": [{"text": _strip_markdown_noise(raw), "page": None, "heading": None}],
        "title": title,
        "metadata": {
            "num_pages": None,
            "ocr_used": False,
            "source_type": "md",
        },
    }
