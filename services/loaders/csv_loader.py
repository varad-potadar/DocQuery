"""
services/loaders/csv_loader.py

Loads CSV files. Each row is rendered as "column: value" pairs rather
than a bare comma-dump, so a keyword search for a column name (e.g.
"status") still matches — plain CSV text loses that context once it's
just one long line of values.

MAX_ROWS caps very large files so processing stays fast on a modest
machine; the loader still reports how many rows it actually indexed.
"""

import csv
import io
from typing import Dict

from services.loaders.text_utils import decode_bytes

MAX_ROWS = 5000


def load(file_bytes: bytes, filename: str) -> Dict:
    text = decode_bytes(file_bytes)
    if not text.strip():
        raise ValueError("This CSV file appears to be empty.")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("This CSV file appears to be empty.")

    header, data_rows = rows[0], rows[1:]
    truncated = len(data_rows) > MAX_ROWS
    data_rows = data_rows[:MAX_ROWS]

    paragraphs = []
    for row in data_rows:
        pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row) if v.strip()]
        if pairs:
            paragraphs.append(", ".join(pairs))

    text_out = "\n\n".join(paragraphs)
    if not text_out.strip():
        raise ValueError("No usable rows were found in this CSV file.")

    return {
        "sections": [{"text": text_out, "page": None, "heading": None}],
        "title": filename,
        "metadata": {
            "num_pages": None,
            "ocr_used": False,
            "source_type": "csv",
            "rows_indexed": len(data_rows),
            "rows_truncated": truncated,
        },
    }
