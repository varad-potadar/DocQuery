"""
main.py  — FastAPI backend for DocQuery

Endpoints:
  POST   /upload                Upload and process a document (any supported format)
  POST   /ask                   Ask a question (with session memory; also saves the turn
                                 to that session's persisted conversation)
  GET    /documents             List documents active in this process's live session
  GET    /documents/cached      List previously processed documents available on disk
  POST   /documents/restore     Restore a cached document into this session
  DELETE /documents/cache/{h}   Delete one cached document from disk
  GET    /conversations         List saved conversations (most recently updated first)
  GET    /conversations/{id}    Get one conversation's full message history
  DELETE /conversations/{id}    Delete one saved conversation from disk
  DELETE /session/{id}          Clear a session's short-term rewriting memory
                                 (does not delete its saved conversation — see above)
  GET    /health                Health check

This mirrors app.py's Streamlit behavior — both call the same
services.ingest pipeline and the same services.conversations store, so
a document processed — or a conversation held — through one is visible
through the other.
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from services.vector_store import VectorStore
from services.qa_engine import answer_question
from services.memory import get_history, append_turn, clear_session, get_last_n_turns
from services.embedder import EMBEDDING_DIM
from services import ingest, cache, conversations

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="DocQuery API",
    description="Conversational Document Intelligence Assistant",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Global state (single-process, in-memory — same model as before;
# the on-disk cache in services/cache.py is what survives a restart)
# ------------------------------------------------------------------

vector_store = VectorStore(dim=EMBEDDING_DIM)
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


class RestorePayload(BaseModel):
    content_hash: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_docs": len(doc_registry),
        "total_vectors": vector_store.debug_count(),
        "cached_docs": len(cache.list_cached_documents()),
        "saved_conversations": len(conversations.list_conversations()),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        doc_info = ingest.process_upload(file_bytes, file.filename, vector_store, doc_registry)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    print(f"[upload] '{doc_info['filename']}' — {doc_info['source']}, "
          f"{doc_info['num_chunks']} chunks, {vector_store.debug_count()} total vectors.")

    return {
        "filename": doc_info["filename"],
        "num_chunks": doc_info["num_chunks"],
        "num_pages": doc_info.get("num_pages"),
        "title": doc_info.get("title"),
        "ocr_used": doc_info.get("ocr_used", False),
        "source": doc_info["source"],
        "status": "Document processed successfully",
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

    append_turn(payload.session_id, payload.question, result["answer"], result["sources"])

    conversations.append_messages(payload.session_id, [
        {"role": "user", "content": payload.question},
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "rewritten": result["rewritten_query"],
            "original_question": payload.question,
        },
    ], docs=[
        {"content_hash": d.get("content_hash"), "filename": d.get("filename")}
        for d in doc_registry.values() if d.get("content_hash")
    ])

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


@app.get("/documents/cached")
def get_cached_documents():
    return {"documents": cache.list_cached_documents()}


@app.post("/documents/restore")
def restore_document(payload: RestorePayload):
    try:
        doc_info = ingest.restore_from_cache(payload.content_hash, vector_store, doc_registry)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return doc_info


@app.delete("/documents/cache/{content_hash}")
def delete_cached_document(content_hash: str):
    deleted = cache.delete_cached_document(content_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found.")
    return {"status": "deleted", "content_hash": content_hash}


@app.get("/conversations")
def list_saved_conversations():
    return {"conversations": conversations.list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    result = conversations.load_conversation(conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation_id": conversation_id,
        "messages": result["messages"],
        "docs": result["docs"],
    }


@app.delete("/conversations/{conversation_id}")
def delete_saved_conversation(conversation_id: str):
    deleted = conversations.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "Session cleared", "session_id": session_id}
