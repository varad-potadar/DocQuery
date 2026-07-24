"""
services/loaders/__init__.py

Maps a file extension to the loader that knows how to read it. Every
loader module exposes one function, load(file_bytes, filename), which
returns a normalized "Document" dict:

    {
        "sections": [{"text": str, "page": int|None, "heading": str|None}, ...],
        "title": str,
        "metadata": {"num_pages": int|None, "ocr_used": bool, "source_type": str, ...},
    }

That's the one shape every other part of the app (chunker, cache,
vector store) works with, no matter what the original file was.

To support a new file type: write one loader module with a load()
function, then add one line to EXTENSION_LOADERS below. Nothing else
in the app needs to change.
"""

from typing import Dict

from services.loaders import (
    pdf_loader,
    docx_loader,
    txt_loader,
    md_loader,
    csv_loader,
    html_loader,
    image_loader,
)

EXTENSION_LOADERS = {
    "pdf": pdf_loader,
    "docx": docx_loader,
    "txt": txt_loader,
    "md": md_loader,
    "markdown": md_loader,
    "csv": csv_loader,
    "html": html_loader,
    "htm": html_loader,
    "png": image_loader,
    "jpg": image_loader,
    "jpeg": image_loader,
}

SUPPORTED_EXTENSIONS = sorted(EXTENSION_LOADERS.keys())


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def is_supported(filename: str) -> bool:
    return get_extension(filename) in EXTENSION_LOADERS


def load_document(file_bytes: bytes, filename: str) -> Dict:
    """
    Detects the file type from its extension, runs the matching loader,
    and returns the normalized Document dict. Raises ValueError with a
    clear, user-facing message for unsupported or unreadable files.
    """
    ext = get_extension(filename)
    loader = EXTENSION_LOADERS.get(ext)

    if loader is None:
        supported = ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS)
        raise ValueError(f"'.{ext or '?'}' files aren't supported. Supported types: {supported}")

    return loader.load(file_bytes, filename)
