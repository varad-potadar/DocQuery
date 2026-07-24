"""
services/loaders/docx_loader.py

Extracts text from .docx files using python-docx, keeping structure
instead of flattening everything into one blob:

  - Each Heading/Title paragraph starts a new "section", so retrieval
    can report which section/heading a chunk came from.
  - Regular paragraphs are appended to the current section's text.
  - Tables are rendered as simple "cell | cell | cell" rows and appended
    to the current section too, so their content stays searchable
    without needing a separate table data model.

python-docx exposes paragraphs and tables as two separate flat lists by
default, which loses their real order in the document. _iter_block_items
below is the standard python-docx recipe for walking both in the order
they actually appear.
"""

import io
from typing import Dict, List

import docx
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph


def _iter_block_items(parent):
    """Yield each paragraph and table in the order it appears in the document."""
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent for _iter_block_items")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _table_to_text(table: Table) -> str:
    lines = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def load(file_bytes: bytes, filename: str) -> Dict:
    document = docx.Document(io.BytesIO(file_bytes))

    sections: List[Dict] = []
    current_heading = None
    current_parts: List[str] = []
    headings_found: List[str] = []

    def flush():
        text = "\n".join(p for p in current_parts if p.strip())
        if text.strip():
            sections.append({"text": text, "page": None, "heading": current_heading})

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            style_name = (block.style.name or "") if block.style else ""
            text = block.text.strip()
            is_heading = style_name.lower().startswith("heading") or style_name.lower() == "title"

            if is_heading:
                if text:
                    flush()
                    current_parts = []
                    current_heading = text
                    headings_found.append(text)
                continue

            if text:
                current_parts.append(text)

        elif isinstance(block, Table):
            table_text = _table_to_text(block)
            if table_text:
                current_parts.append(table_text)

    flush()

    if not sections:
        raise ValueError("No readable text could be found in this document.")

    core_title = (document.core_properties.title or "").strip()
    title = core_title or (headings_found[0] if headings_found else filename)

    return {
        "sections": sections,
        "title": title[:200],
        "metadata": {
            "num_pages": None,   # Word paginates dynamically; not derivable without rendering.
            "ocr_used": False,
            "source_type": "docx",
        },
    }
