"""
services/loaders/image_loader.py

Loads a single image (.png/.jpg/.jpeg) by running OCR on it. This is
what makes a photo of a document, a screenshot, or a whiteboard shot
queryable the same way a PDF is.
"""

from typing import Dict

from services.ocr import ocr_image


def load(file_bytes: bytes, filename: str) -> Dict:
    text = ocr_image(file_bytes)

    if not text.strip():
        raise ValueError(
            "No readable text could be found in this image. "
            "Try a clearer photo or a higher-resolution scan."
        )

    return {
        "sections": [{"text": text, "page": 1, "heading": None}],
        "title": filename,
        "metadata": {
            "num_pages": 1,
            "ocr_used": True,
            "source_type": "image",
        },
    }
