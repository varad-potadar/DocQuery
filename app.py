"""
app.py — Pure Streamlit frontend + backend for DocQuery

A proper chat assistant UI with:
  - Sidebar: upload + document list
  - Main: full chat conversation with source attribution
  - Rewritten query shown as a subtle tooltip
  
No FastAPI dependency - everything runs inside Streamlit.
"""

import uuid
import streamlit as st
from typing import List, Dict

# Import all services directly
from services.pdf_extractor import extract_text_from_pdf
from services.chunker import chunk_text
from services.embedder import embed_chunks
from services.vector_store import VectorStore
from services.qa_engine import answer_question
from services.memory import get_history, append_turn, clear_session, get_last_n_turns

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — clean, simple theme with guaranteed contrast
# ------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
[data-testid="stAppViewContainer"] > .main {
    background: #f8fafc;
}

/* Sidebar - clean slate */
[data-testid="stSidebar"] {
    background: #1e293b;
}

/* Sidebar text */
[data-testid="stSidebar"],
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stInfo p {
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Document cards - always readable */
.doc-pill {
    background: #334155;
    border-left: 4px solid #818cf8;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    color: #f1f5f9;
}

.doc-pill b,
.doc-pill strong {
    color: #ffffff;
}

.doc-pill span {
    color: #cbd5e1;
}

/* Uploaded file names */
[data-testid="stSidebar"] .stFileUploaderFile {
    background: #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px;
    padding: 8px;
}

/* Headers in main */
h1, h2, h3 {
    color: #1e293b;
    font-weight: 700;
}

/* Chat cards */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Input box */
[data-testid="stChatInput"] textarea {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
}

/* Source badges */
.source-badge {
    display: inline-block;
    background: #e0e7ff;
    color: #4338ca;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 8px;
}

/* Rewrite note */
.rewrite-note {
    color: #64748b;
    font-size: 12px;
    margin-top: 8px;
}

/* Buttons */
.stButton button {
    border-radius: 12px;
    font-weight: 600;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------------------------
# Helper functions for document processing
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


def index_document(file_content: bytes, filename: str, vector_store: VectorStore, doc_registry: Dict) -> Dict:
    """
    Index a single PDF document and add to vector store.
    Returns metadata about the indexed document.
    """
    # Save temporarily (PyMuPDF needs a file path)
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name
    
    try:
        # Extract text
        text, metadata = extract_text_from_pdf(tmp_path)
        
        # Chunk
        chunks = chunk_text(text, chunk_size=600, overlap=120)
        chunks = [c for c in chunks if is_useful_chunk(c["text"])]
        
        if not chunks:
            raise ValueError("No usable text extracted from PDF.")
        
        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = embed_chunks(texts)
        
        # Index
        vector_store.add(embeddings, chunks, filename)
        
        # Registry
        doc_info = {
            **metadata,
            "num_chunks": len(chunks),
            "doc_id": filename,
            "filename": filename,
        }
        doc_registry[filename] = doc_info
        
        return doc_info
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Session state bootstrap
# ------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str,
    #                "sources": list, "rewritten": str}
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # list of dicts

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(dim=384)

if "doc_registry" not in st.session_state:
    st.session_state.doc_registry = {}

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📄 DocQuery")
    st.caption("Chat with your documents — grounded answers only.")
    st.divider()

    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚙️ Index Documents", use_container_width=True, type="primary"):
            for uf in uploaded_files:
                if any(d["filename"] == uf.name for d in st.session_state.uploaded_docs):
                    st.warning(f"Already indexed: {uf.name}")
                    continue

                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        doc_info = index_document(
                            uf.getvalue(),
                            uf.name,
                            st.session_state.vector_store,
                            st.session_state.doc_registry
                        )
                        st.session_state.uploaded_docs.append(doc_info)
                        st.success(f"✅ {uf.name} — {doc_info['num_chunks']} chunks")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {str(e)}")

    st.divider()
    st.markdown("### Loaded Documents")

    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(
                f"<div class='doc-pill'>📄 <b>{doc['filename']}</b><br>"
                f"<span style='font-size:11px;color:#5a6480'>"
                f"{doc['num_pages']} pages · {doc['num_chunks']} chunks</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            # Clear memory service as well
            clear_session(st.session_state.session_id)
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.uploaded_docs = []
            st.session_state.vector_store = VectorStore(dim=384)
            st.session_state.doc_registry = {}
            clear_session(st.session_state.session_id)
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.markdown("""
<div style='text-align:center;padding:10px 0 20px 0'>
<h1 style='color:#760031'>🤖 DocQuery</h1>
<p style='color:#64748b'>
Conversational Document Intelligence Platform
</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.uploaded_docs:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #4a5070;'>
        <div style='font-size: 48px; margin-bottom: 16px;'>📄</div>
        <div style='font-size: 18px; font-weight: 500; color: #6a7090;'>No documents loaded</div>
        <div style='font-size: 14px; margin-top: 8px;'>Upload PDFs from the sidebar to begin chatting.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Sources (only on assistant messages)
        if msg["role"] == "assistant" and msg.get("sources"):
            badges = "".join(
                f"<span class='source-badge'>📄 {src}</span>"
                for src in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{badges}</div>", unsafe_allow_html=True)

        # Rewritten query (subtle, only when it differs)
        if msg["role"] == "assistant" and msg.get("rewritten"):
            orig = msg.get("original_question", "")
            rw = msg["rewritten"]
            if orig and rw.lower().strip() != orig.lower().strip():
                st.markdown(
                    f"<div class='rewrite-note'>🔁 Interpreted as: <i>{rw}</i></div>",
                    unsafe_allow_html=True,
                )

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend (now directly calling services)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                # Get conversation history
                history = get_last_n_turns(st.session_state.session_id, n=4)
                
                # Call QA engine directly
                result = answer_question(
                    question=prompt,
                    vector_store=st.session_state.vector_store,
                    history=history,
                )
                
                answer = result["answer"]
                sources = result["sources"]
                rewritten = result["rewritten_query"]
                
                # Store in memory service
                append_turn(
                    st.session_state.session_id,
                    prompt,
                    answer,
                    sources,
                )
                
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                sources = []
                rewritten = ""

        st.markdown(answer)

        if sources:
            badges = "".join(
                f"<span class='source-badge'>📄 {src}</span>"
                for src in sources
            )
            st.markdown(f"<div style='margin-top:6px'>{badges}</div>", unsafe_allow_html=True)

        if rewritten and rewritten.lower().strip() != prompt.lower().strip():
            st.markdown(
                f"<div class='rewrite-note'>🔁 Interpreted as: <i>{rewritten}</i></div>",
                unsafe_allow_html=True,
            )

    # Persist to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "rewritten": rewritten,
        "original_question": prompt,
    })
