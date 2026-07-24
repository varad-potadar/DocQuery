"""
services/ocr.py

Thin wrapper around Tesseract (via the pytesseract package), used by
both the PDF loader (for scanned pages) and the image loader.

Why Tesseract, and not EasyOCR / PaddleOCR / a cloud API:
  - Pure CPU. No GPU involved at any point — it never even checks for one.
  - No PyTorch/TensorFlow dependency. EasyOCR and PaddleOCR both sit on
    top of a full deep-learning framework, which means a much bigger
    install and much heavier RAM/CPU use for every page, even on a
    machine with no GPU to speed that up.
  - It's one mature, well-known binary. It can be tested completely on
    its own from a terminal (`tesseract image.png out`) independent of
    the rest of the app, which makes OCR problems easy to isolate.
  - Fully local — no API key, no per-page cost, and document images
    never have to leave the machine.

The trade-off: Tesseract needs its own system binary (not a pip
package), so it has to be installed separately — see requirements.txt
and README.md for the one-line install command per OS.
"""

import io
from typing import Union

from PIL import Image
import pytesseract

# A page/image with fewer non-space characters than this is treated as
# "no real text" — used here and by loaders/pdf_loader.py to decide
# when a PDF page needs OCR instead of native extraction.
MIN_TEXT_CHARS = 20


def ocr_image(image_source: Union[bytes, "Image.Image"]) -> str:
    """
    Runs OCR on an image and returns the extracted text.

    image_source can be raw image bytes (png/jpg/etc.) or an already-open
    PIL.Image. Never raises — on failure it returns "", which callers
    treat the same as "no text found" rather than crashing the upload.
    """
    try:
        if isinstance(image_source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_source))
        else:
            image = image_source

        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        return pytesseract.image_to_string(image).strip()

    except Exception as e:
        print(f"[ocr] OCR failed: {e}")
        return ""


def is_tesseract_available() -> bool:
    """
    Checks whether the Tesseract binary is actually installed and
    reachable. Used at startup so a missing system dependency shows up
    as a clear message instead of a confusing failure deep in a loader.
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
