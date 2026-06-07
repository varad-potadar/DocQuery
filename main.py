"""
main.py  — FastAPI backend for DocQuery

Endpoints:
  POST /upload          Upload and index a PDF
  POST /ask             Ask a question (with session memory)
  GET  /documents       List indexed documents
  DELETE /session/{id}  Clear session memory
  GET  /health          Health check
"""

from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from services.pdf_extractor import extract_text_from_pdf
from services.chunker import chunk_text
from services.embedder import embed_chunks
from services.vector_store import VectorStore
from services.qa_engine import answer_question
from services.memory import get_history, append_turn, clear_session, get_last_n_turns

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="DocQuery API",
    description="Conversational Document Intelligence Assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = VectorStore(dim=384)

# Metadata registry: doc_id -> {title, num_chunks, num_pages, filename}
doc_registry: dict = {}


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class QuestionPayload(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    rewritten_query: str


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def is_useful_chunk(text: str) -> bool:
    """Filter out boilerplate chunks that add noise."""
    t = text.lower().strip()
    if len(t) < 20:
        return False
    noise_patterns = [
        "creative commons",
        "doi.org",
        "all rights reserved",
        "terms and conditions",
        "this page intentionally left blank",
        "table of contents",
    ]
    return not any(p in t for p in noise_patterns)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_docs": len(doc_registry),
        "total_vectors": vector_store.debug_count(),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract
    try:
        text, metadata = extract_text_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

    # Chunk
    chunks = chunk_text(text, chunk_size=600, overlap=120)
    chunks = [c for c in chunks if is_useful_chunk(c["text"])]

    if not chunks:
        raise HTTPException(status_code=422, detail="No usable text extracted from PDF.")

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(texts)

    # Index
    doc_id = file.filename
    vector_store.add(embeddings, chunks, doc_id)

    # Registry
    doc_registry[doc_id] = {
        **metadata,
        "num_chunks": len(chunks),
        "doc_id": doc_id,
    }

    print(f"[upload] Indexed '{doc_id}': {len(chunks)} chunks, "
          f"{vector_store.debug_count()} total vectors.")

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "num_pages": metadata["num_pages"],
        "title": metadata["title"],
        "status": "Document indexed successfully",
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: QuestionPayload):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = get_last_n_turns(payload.session_id, n=4)

    result = answer_question(
        question=payload.question,
        vector_store=vector_store,
        history=history,
    )

    append_turn(
        payload.session_id,
        payload.question,
        result["answer"],
        result["sources"],
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewritten_query=result["rewritten_query"],
    )


@app.get("/documents")
def list_documents():
    return {
        "documents": list(doc_registry.values()),
        "total": len(doc_registry),
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "Session cleared", "session_id": session_id}
