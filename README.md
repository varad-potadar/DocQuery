# DocQuery

Conversational document intelligence: upload documents, ask questions
about them in a chat interface, get answers grounded in the actual
document text with sources cited.

## What it does

- Upload PDF, DOCX, TXT, Markdown, CSV, and HTML files, plus PNG/JPG
  images — scanned pages and photographed/screenshotted documents are
  read automatically with OCR.
- Ask questions, get follow-up-aware conversational answers with the
  source document (and page, where available) shown.
- Summarize and compare across multiple documents at once.
- Previously processed documents are remembered — reopening the app
  doesn't mean re-uploading and reprocessing everything again.

## Requirements

- Python 3.10+
- The **Tesseract OCR** engine, installed as a system package (this is
  what actually reads scanned pages and photos — separate from the
  `pytesseract` Python package in requirements.txt):

  ```bash
  # Debian/Ubuntu
  sudo apt-get install tesseract-ocr

  # macOS
  brew install tesseract

  # Windows: installer at https://github.com/UB-Mannheim/tesseract/wiki
  ```

  Deploying to Streamlit Community Cloud needs no extra step — it
  reads `packages.txt` in this repo automatically.

- A [Groq API key](https://console.groq.com) (free tier available),
  used for the chat answers and follow-up-question rewriting. Copy
  `.streamlit/secrets.toml` to `.env` and fill in `GROQ_API_KEY`, or
  set it as an environment variable.

## Install & run

```bash
pip install -r requirements.txt

# Streamlit app (the main interface)
streamlit run app.py

# FastAPI backend (optional — a separate API onto the same pipeline)
uvicorn main:app --reload
```

Everything runs on CPU. No GPU is used or required anywhere in the
pipeline (embeddings, OCR, and retrieval are all CPU-only; the chat
model itself runs on Groq's cloud, not on your machine).

## How it's organized

```
app.py                    Streamlit UI (chat, uploads, document management)
main.py                   FastAPI equivalent of the same pipeline
services/
  loaders/                One small file per supported format
    __init__.py             extension -> loader registry
    pdf_loader.py            native text, falls back to OCR per page
    docx_loader.py           keeps headings/paragraphs/tables in order
    txt_loader.py, md_loader.py, csv_loader.py, html_loader.py
    image_loader.py          OCR on a standalone image
  ocr.py                  Shared Tesseract wrapper (used by pdf + image loaders)
  chunker.py              Splits text into overlapping chunks for retrieval
  embedder.py             Sentence-transformer embeddings
  vector_store.py         FAISS (semantic) + BM25 (keyword) + RRF fusion
  cache.py                On-disk cache, keyed by content hash
  ingest.py               The one pipeline app.py and main.py both call
  qa_engine.py             Retrieval + grounded answer generation
  query_rewriter.py       Resolves "it"/"that"/"the other one" in follow-ups
  memory.py               Per-session conversation history
  config.py               Shared constants (embedding model, chunk size, cache version)
```

To add a new file format: write one loader module with a `load(file_bytes,
filename)` function, add one line to `EXTENSION_LOADERS` in
`services/loaders/__init__.py`. Nothing else needs to change.

## A few design choices worth knowing

**OCR uses Tesseract**, not EasyOCR/PaddleOCR/a cloud API. Tesseract is
plain CPU, has no PyTorch/TensorFlow underneath it, and is one binary
you can test on its own (`tesseract image.png out`) independent of the
rest of the app. That keeps memory/CPU use low and install size small
— the deciding factor given this is meant to run on an ordinary laptop
with no GPU.

**PDF OCR is per-page, not per-document.** Each page's native text is
checked first; only a page with almost no extractable text gets
rendered to an image and OCR'd. A normal, already-text-based PDF never
triggers OCR at all.

**Caching is on disk, not in the browser.** By the time a document is
processed, the app already has the chunk text and embedding vectors it
needs — saving those to a small local folder means they can be
reloaded exactly as-is, with no size limits and no difference between
opening the app via Streamlit or via the API. See `services/cache.py`
for the full reasoning. Only the processed result is kept, not the
original uploaded file.

**Document identity is a content hash (SHA-256), not a filename.** Two
files can share a name; a modified file can keep its old name. The
cache is keyed by hash so neither situation causes stale or incorrect
reuse — see `services/cache.py` docstring.

**Light/dark theming uses `prefers-color-scheme`, not a Streamlit
theme config.** All custom colors in `app.py` are CSS variables
defined once in light form and overridden in a
`@media (prefers-color-scheme: dark)` block — plain, standard CSS with
no dependency on Streamlit's internal implementation, so it can't
break on a Streamlit upgrade. Deliberately, `.streamlit/config.toml`
has no `[theme]` section: setting even one theme option there makes
Streamlit stop auto-following the system light/dark preference for
its own native widgets (a real, documented gotcha), which would fight
against the CSS approach above. Leaving it unset keeps Streamlit's own
chrome and this app's custom styling reading the same signal.

## Managing documents & cache

- **Remove** (🗑️ next to an active document) — takes it out of the
  current session only. The cached copy on disk is untouched.
- **Delete cached copy** (🗑️ in "Previously processed documents") —
  deletes that document's cache from disk permanently.
- **Clear Chat** — clears the conversation only.
- **Reset Workspace** — clears chat and active documents for this
  session. Cached copies on disk are kept, so they can be restored
  again later.
- **Delete ALL cached data** (Danger zone) — wipes the entire on-disk
  cache. Cannot be undone.

Cache files live under `data/doc_cache/`. Deleting that folder by hand
has the same effect as "Delete ALL cached data".

## Limitations

- The on-disk cache is per-machine (it's a local folder, not a shared
  database) — it won't follow you to a different computer.
- Cache entries are tied to the embedding model and chunking version
  in `services/config.py`. If either changes, older cache entries are
  automatically ignored rather than reused incorrectly (see
  `CACHE_SCHEMA_VERSION` in `services/config.py`).
- CSV files are capped at 5,000 rows for processing speed (configurable
  in `services/loaders/csv_loader.py`).
- Very low-quality scans/photos may still OCR poorly — this is a
  limitation of OCR in general, not specific to this app.
