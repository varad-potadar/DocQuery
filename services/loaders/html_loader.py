"""
services/loaders/html_loader.py

Loads HTML files by stripping tags/scripts/styles down to visible text.
Uses BeautifulSoup with Python's built-in "html.parser" backend rather
than lxml, to avoid an extra dependency.
"""

from typing import Dict

from bs4 import BeautifulSoup

from services.loaders.text_utils import decode_bytes, guess_title


def load(file_bytes: bytes, filename: str) -> Dict:
    html = decode_bytes(file_bytes)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").split("\n") if line.strip()]
    text = "\n\n".join(lines)

    if not text.strip():
        raise ValueError("No readable text could be found in this HTML file.")

    page_title = soup.title.string.strip() if (soup.title and soup.title.string) else None
    title = (page_title or guess_title(text, filename))[:200]

    return {
        "sections": [{"text": text, "page": None, "heading": None}],
        "title": title,
        "metadata": {
            "num_pages": None,
            "ocr_used": False,
            "source_type": "html",
        },
    }
