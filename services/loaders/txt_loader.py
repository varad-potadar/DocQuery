"""
services/loaders/txt_loader.py

Loads plain-text files. Not every .txt file is UTF-8, so this tries a
short list of common encodings before giving up.
"""

from typing import Dict

from services.loaders.text_utils import decode_bytes, guess_title


def load(file_bytes: bytes, filename: str) -> Dict:
    text = decode_bytes(file_bytes).strip()

    if not text:
        raise ValueError("This text file appears to be empty.")

    return {
        "sections": [{"text": text, "page": None, "heading": None}],
        "title": guess_title(text, filename),
        "metadata": {
            "num_pages": None,
            "ocr_used": False,
            "source_type": "txt",
        },
    }
