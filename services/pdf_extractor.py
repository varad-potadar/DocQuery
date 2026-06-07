"""
services/pdf_extractor.py

Extracts text from PDFs using PyMuPDF.
Returns both the raw text and a small metadata dict derived from the first page.
"""

import re
import fitz  # PyMuPDF
from typing import Tuple, Dict


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("+e", "the").replace("+is", "this")
    text = re.sub(r"-\n(\w)", r"\1", text)      # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse horizontal spaces
    return text.strip()


def extract_text_from_pdf(path: str) -> Tuple[str, Dict]:
    """
    Returns:
        text     – full cleaned text of the document
        metadata – {"title": str, "num_pages": int, "filename": str}
    """
    doc = fitz.open(path)
    pages_text = []

    for page in doc:
        pages_text.append(page.get_text("text"))

    raw = "\n".join(pages_text)
    text = _clean(raw)

    # Best-effort title: first non-empty line of the document
    first_lines = [l.strip() for l in text.split("\n") if l.strip()]
    guessed_title = first_lines[0] if first_lines else path

    metadata = {
        "title": guessed_title[:200],
        "num_pages": len(doc),
        "filename": path.split("/")[-1],
    }

    return text, metadata
