"""
services/loaders/pdf_loader.py

Extracts text from PDFs page by page using PyMuPDF (fitz).

Strategy, per page:
  1. Try native text extraction first (fast, no OCR).
  2. If a page comes back with almost no text, treat it as a scanned/
     image page: render just that page to an image and OCR it instead.

Because the decision is made per page, a normal machine-readable PDF
never triggers OCR at all (same speed as before), while a scanned PDF —
or a PDF that mixes native-text and scanned pages — still produces
usable text. Rendering the page for OCR reuses PyMuPDF (already a
dependency for text extraction), so no extra system dependency
(e.g. poppler/pdf2image) is needed just to turn a page into an image.
"""

import re
from typing import Dict

import fitz  # PyMuPDF

from services.ocr import ocr_image, MIN_TEXT_CHARS

# OCR render resolution. 200 DPI is a middle ground: noticeably faster
# and lighter than 300 DPI on a low-power machine, while still sharp
# enough for Tesseract to read normal printed/scanned text well.
OCR_DPI = 200


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("+e", "the").replace("+is", "this")
    text = re.sub(r"-\n(\w)", r"\1", text)      # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse horizontal spaces
    return text.strip()


def _page_needs_ocr(text: str) -> bool:
    """A page is treated as scanned/image-based if it yields almost no
    real text via native extraction."""
    stripped = re.sub(r"\s+", "", text or "")
    return len(stripped) < MIN_TEXT_CHARS


def load(file_bytes: bytes, filename: str) -> Dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    if doc.needs_pass:
        doc.close()
        raise ValueError("This PDF is password-protected and can't be read.")

    sections = []
    ocr_used = False

    for page_num, page in enumerate(doc, start=1):
        native_text = page.get_text("text")

        if _page_needs_ocr(native_text):
            pix = page.get_pixmap(dpi=OCR_DPI)
            ocr_text = ocr_image(pix.tobytes("png"))
            if ocr_text.strip():
                native_text = ocr_text
                ocr_used = True

        cleaned = _clean(native_text)
        if cleaned:
            sections.append({"text": cleaned, "page": page_num, "heading": None})

    num_pages = len(doc)
    doc.close()

    if not sections:
        raise ValueError("No readable text could be found in this PDF, even after OCR.")

    first_lines = [l.strip() for l in sections[0]["text"].split("\n") if l.strip()]
    title = first_lines[0][:200] if first_lines else filename

    return {
        "sections": sections,
        "title": title,
        "metadata": {
            "num_pages": num_pages,
            "ocr_used": ocr_used,
            "source_type": "pdf",
        },
    }
