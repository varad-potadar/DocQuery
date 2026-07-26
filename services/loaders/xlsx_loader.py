"""
services/loaders/xlsx_loader.py

Loads Excel (.xlsx) files with openpyxl — not pandas, since pandas
would pull in a much heavier dependency just to read cells row by row,
which cuts against keeping this app light on a modest machine.

Each worksheet becomes its own "section" (the same concept pdf_loader
uses for pages and docx_loader uses for headings), with the sheet name
as the heading — so retrieval can report which sheet an answer came
from. Within a sheet, the first row is treated as the column headers,
and every later row is rendered as "column: value" pairs (same
approach as csv_loader.py) so a keyword search for a column name still
matches — a bare cell dump loses that context.

Known, deliberate limitations (kept simple on purpose):
  - The first row of each sheet is always treated as the header row.
    A sheet with a title row above its real header will misparse —
    same trade-off csv_loader.py already makes for CSV files.
  - Formula cells are read as their last-saved calculated value
    (openpyxl's data_only=True). If a workbook was generated
    programmatically and never opened in Excel, that cached value can
    be missing; such cells are skipped rather than showing raw
    formula text, since a formula string isn't useful to search over.
  - MAX_ROWS caps how many data rows are read per sheet, same
    reasoning as csv_loader.py's cap.
"""

import io
from datetime import date, datetime
from typing import Dict, List

import openpyxl

MAX_ROWS = 5000


def _cell_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.time() == datetime.min.time():
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sheet_to_text(sheet) -> tuple[str, int, bool]:
    """Returns (rendered_text, data_rows_used, was_truncated) for one sheet."""
    rows_iter = sheet.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return "", 0, False

    header = [str(h).strip() if h is not None else "" for h in header]

    paragraphs = []
    row_count = 0
    truncated = False

    for row in rows_iter:
        if row_count >= MAX_ROWS:
            truncated = True
            break
        pairs = [
            f"{h}: {_cell_to_text(v)}"
            for h, v in zip(header, row)
            if h and _cell_to_text(v)
        ]
        if pairs:
            paragraphs.append(", ".join(pairs))
            row_count += 1

    return "\n\n".join(paragraphs), row_count, truncated


def load(file_bytes: bytes, filename: str) -> Dict:
    try:
        workbook = openpyxl.load_workbook(
            io.BytesIO(file_bytes), data_only=True, read_only=True
        )
    except Exception as e:
        raise ValueError(f"This Excel file couldn't be read: {e}")

    sections: List[Dict] = []
    total_rows = 0
    any_truncated = False

    for sheet in workbook.worksheets:
        text, rows_used, truncated = _sheet_to_text(sheet)
        if text.strip():
            sections.append({"text": text, "page": None, "heading": sheet.title})
            total_rows += rows_used
            any_truncated = any_truncated or truncated

    workbook.close()

    if not sections:
        raise ValueError("No usable data was found in this spreadsheet.")

    title = sections[0]["heading"] or filename

    return {
        "sections": sections,
        "title": title,
        "metadata": {
            "num_pages": len(sections),  # one "page" per sheet with data
            "ocr_used": False,
            "source_type": "xlsx",
            "rows_indexed": total_rows,
            "rows_truncated": any_truncated,
        },
    }
