"""
app.py  — Streamlit frontend for DocQuery

A proper chat assistant UI with:
  - Sidebar: upload + document list
  - Main: full chat conversation with source attribution
  - Rewritten query shown as a subtle tooltip
"""

import uuid
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

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
# Custom CSS — clean, professional dark-accent theme
# ------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}

/* ---- main area ---- */
[data-testid="stAppViewContainer"] > .main {
    background: #13151f;
}

/* ---- chat messages ---- */
[data-testid="stChatMessage"] {
    background: #1a1d2e !important;
    border: 1px solid #252840 !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}

/* ---- source badge ---- */
.source-badge {
    display: inline-block;
    background: #1e2a4a;
    color: #7eb8f7 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 4px;
    margin-top: 6px;
    border: 1px solid #2e4070;
}

/* ---- rewritten query ---- */
.rewrite-note {
    color: #5a6480 !important;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 4px;
}

/* ---- doc pill ---- */
.doc-pill {
    background: #1a1d2e;
    border: 1px solid #2a2f4a;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #a0aac0 !important;
}

/* ---- header ---- */
h1, h2, h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #e8eaf6 !important;
}

/* ---- input ---- */
[data-testid="stChatInput"] textarea {
    background: #1a1d2e !important;
    color: #e8eaf6 !important;
    border: 1px solid #2a2f4a !important;
}
</style>
""", unsafe_allow_html=True)

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
    st.session_state.uploaded_docs = []  # list of dicts from /upload response

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
                        resp = requests.post(
                            f"{BACKEND_URL}/upload",
                            files={"file": (uf.name, uf.read(), "application/pdf")},
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.uploaded_docs.append({
                                "filename": uf.name,
                                "num_chunks": data.get("num_chunks", "?"),
                                "num_pages": data.get("num_pages", "?"),
                                "title": data.get("title", uf.name),
                            })
                            st.success(f"✅ {uf.name} — {data.get('num_chunks', '?')} chunks")
                        else:
                            detail = resp.json().get("detail", resp.text)
                            st.error(f"❌ {uf.name}: {detail}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend (port 8000). Is FastAPI running?")
                        break
                    except requests.exceptions.Timeout:
                        st.error(f"❌ {uf.name}: Indexing timed out.")

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
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.uploaded_docs = []
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.markdown("## 💬 Document Chat")

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

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": prompt,
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    rewritten = data.get("rewritten_query", "")
                else:
                    answer = f"❌ Backend error (HTTP {resp.status_code}): {resp.text}"
                    sources = []
                    rewritten = ""

            except requests.exceptions.ConnectionError:
                answer = "❌ Cannot reach the backend on port 8000. Make sure FastAPI is running."
                sources = []
                rewritten = ""
            except requests.exceptions.Timeout:
                answer = "❌ Request timed out. The document may be very large."
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
