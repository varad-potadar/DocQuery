"""
app.py — Streamlit frontend for DocQuery

A chat assistant UI with:
  - Sidebar: chat history (persisted, resumable — see
    services/conversations.py), upload (PDF/DOCX/XLSX/TXT/MD/CSV/HTML/
    images), active document list, and a browser for previously
    processed (cached) documents
  - Main: full chat conversation with source attribution
  - Custom file upload display (no CSS override issues)

All document processing goes through services/ingest.py, shared with
main.py (the FastAPI backend), so both stay in sync.
"""

import uuid
import os
import streamlit as st
from typing import List, Dict
# Map the Streamlit secret directly to the environment variable
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
# Import all services directly
from services.vector_store import VectorStore
from services.qa_engine import answer_question
from services.memory import get_history, append_turn, clear_session, get_last_n_turns
from services.embedder import EMBEDDING_DIM
from services.loaders import SUPPORTED_EXTENSIONS
from services import ingest, cache, conversations

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="DocQuery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — theme-adaptive tokens (see comment below), document/
# citation-inspired styling on top of Streamlit's own layout
# ------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@450;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ==================================================================
   Design tokens.
   Light values are the default on :root; the dark block only
   overrides what changes. This is driven by the OS/browser
   prefers-color-scheme setting rather than any Streamlit-internal
   mechanism, on purpose: it's plain, standard CSS that won't break
   across Streamlit versions, and it matches Streamlit's own default
   "use system setting" behavior (nothing here touches
   .streamlit/config.toml, so that default stays intact).
   ================================================================== */
:root {
    --bg:            #F3F5F9;
    --surface:       #FFFFFF;
    --border:        #E2E6ED;
    --text:          #1E2433;
    --text-muted:    #5B6478;

    --sidebar-bg:    #EAEEF5;
    --sidebar-text:  #1E2433;

    --ink:           #26314D;
    --ink-strong:    #16203A;
    --on-ink:        #F3F5F9;

    --amber:         #A5720F;
    --amber-bg:      #FBF0DC;
    --amber-border:  #E9C783;

    --sage:          #3F7355;
    --sage-bg:       #E4EFE7;
    --sage-border:   #B7D6C2;

    --danger:        #AE3C2F;
    --danger-bg:     #FBEAE7;
    --danger-border: #EFC3BB;

    --shadow: 0 1px 2px rgba(23,27,46,0.05), 0 4px 14px rgba(23,27,46,0.06);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg:            #10141F;
        --surface:       #1A2032;
        --border:        #2A3346;
        --text:          #E7EAF3;
        --text-muted:    #9AA4C0;

        --sidebar-bg:    #151B2B;
        --sidebar-text:  #E7EAF3;

        --ink:           #AEBBE0;
        --ink-strong:    #D7DEF2;
        --on-ink:        #10141F;

        --amber:         #E0B25C;
        --amber-bg:      #372B12;
        --amber-border:  #5C481E;

        --sage:          #8FC6A4;
        --sage-bg:       #17281D;
        --sage-border:   #2D4A38;

        --danger:        #E19286;
        --danger-bg:     #33201D;
        --danger-border: #593029;

        --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.35);
    }
}

/* ==================================================================
   Base
   ================================================================== */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--bg);
}

code { font-family: 'IBM Plex Mono', monospace; }

/* ==================================================================
   Sidebar — treated as the document index/catalog, visually a shade
   apart from the reading pane.
   ================================================================== */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: var(--sidebar-text) !important;
}

.app-title {
    font-family: 'Fraunces', serif;
    font-size: 26px;
    font-weight: 600;
    color: var(--sidebar-text);
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.app-title span { color: var(--amber); }

.app-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin: 2px 0 4px 0;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: var(--sidebar-text) !important;
    font-size: 16px;
}

/* ==================================================================
   Chat history list. Buttons can't be nested inside markdown-emitted
   HTML (each st.button is its own element), so these are targeted via
   Streamlit's documented .st-key-<key> class instead of a wrapper div
   — [class*=...] catches every conv_<id> key with one rule since the
   id suffix varies per conversation.
   ================================================================== */
[class*="st-key-conv_"] button {
    background: transparent;
    border: 1px solid transparent;
    text-align: left;
    justify-content: flex-start;
    font-weight: 400;
    color: var(--sidebar-text);
    padding: 6px 10px;
}
[class*="st-key-conv_"] button:hover {
    background: var(--surface);
    border-color: var(--border);
    color: var(--sidebar-text);
}

/* ==================================================================
   Document cards — active + previously processed
   ================================================================== */
.doc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--sage);
    border-radius: 8px;
    padding: 9px 11px;
    margin-bottom: 7px;
    box-shadow: var(--shadow);
}

.doc-card--dormant {
    border-left-color: var(--border);
    box-shadow: none;
}

.doc-card__name {
    font-weight: 500;
    font-size: 13.5px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.doc-card__meta {
    margin-top: 5px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}

.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: transparent;
    white-space: nowrap;
}

.tag--ocr    { color: var(--amber); border-color: var(--amber-border); background: var(--amber-bg); }
.tag--cache  { color: var(--sage);  border-color: var(--sage-border);  background: var(--sage-bg); }

/* ==================================================================
   Pending file list (selected, not yet processed)
   ================================================================== */
.custom-file-item {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 9px 11px;
    margin-bottom: 7px;
}

.custom-file-name {
    color: var(--text) !important;
    font-size: 13.5px;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ==================================================================
   Chat
   ================================================================== */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
}

[data-testid="stChatInput"] textarea {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* Source badges — the one signature detail. Shaped like a small
   index-card tab (flat left accent, mono type) rather than a generic
   pill, so a citation reads as a precise reference, not decoration. */
.source-badge {
    display: inline-block;
    background: var(--amber-bg);
    color: var(--amber);
    border: 1px solid var(--amber-border);
    border-left: 3px solid var(--amber);
    border-radius: 4px;
    padding: 3px 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 8px;
}

.hint-text {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 8px;
}

/* ==================================================================
   Buttons — one quiet default (most buttons here are routine: remove,
   restore, delete, clear), with two specific exceptions styled via
   their Streamlit `key` (see docs.streamlit.io "Component theming":
   a keyed widget gets a matching .st-key-<key> class) rather than any
   internal/undocumented attribute, so this keeps working across
   Streamlit versions.
   ================================================================== */
.stButton button {
    border-radius: 8px;
    font-weight: 500;
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
}

.stButton button:hover {
    border-color: var(--danger);
    color: var(--danger);
    background: var(--danger-bg);
}

.st-key-process_documents_btn button,
.st-key-new_chat_btn button {
    background: var(--ink);
    color: var(--on-ink);
    border-color: var(--ink);
}
.st-key-process_documents_btn button:hover,
.st-key-new_chat_btn button:hover {
    background: var(--ink-strong);
    border-color: var(--ink-strong);
}

.st-key-delete_all_cache_btn button {
    background: transparent;
    color: var(--danger);
    border: 1px solid var(--danger-border);
}
.st-key-delete_all_cache_btn button:hover {
    background: var(--danger);
    color: var(--on-ink);
    border-color: var(--danger);
}

@media (prefers-reduced-motion: no-preference) {
    .stButton button, .doc-card {
        transition: transform 120ms ease, box-shadow 120ms ease,
                    background 120ms ease, border-color 120ms ease, color 120ms ease;
    }
    .stButton button:hover { transform: translateY(-1px); }
}

/* Hide default file uploader per-file details — we render our own list */
[data-testid="stSidebar"] .stFileUploaderFile {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def doc_card_html(doc: Dict, dormant: bool = False) -> str:
    """Renders one document as a styled card with metadata tags."""
    tags = []
    if doc.get("num_pages"):
        tags.append(f"<span class='tag'>{doc['num_pages']} pages</span>")
    tags.append(f"<span class='tag'>{doc['num_chunks']} chunks</span>")
    if doc.get("ocr_used"):
        tags.append("<span class='tag tag--ocr'>OCR</span>")
    if not dormant and doc.get("source") == "cache":
        tags.append("<span class='tag tag--cache'>from cache</span>")
    card_class = "doc-card doc-card--dormant" if dormant else "doc-card"
    return (
        f"<div class='{card_class}'>"
        f"<div class='doc-card__name'>{doc['filename']}</div>"
        f"<div class='doc-card__meta'>{''.join(tags)}</div>"
        f"</div>"
    )


def _current_doc_refs() -> List[Dict]:
    """The {content_hash, filename} pairs for whatever documents are
    active right now — what gets attached to the conversation on save."""
    return [
        {"content_hash": d["content_hash"], "filename": d["filename"]}
        for d in st.session_state.uploaded_docs
        if d.get("content_hash")
    ]


def _autosave_conversation():
    """Persists the current conversation (messages + which documents
    are attached to it), if it has at least one message yet. Called
    after every chat turn AND after every document add/remove, so a
    document added mid-conversation is remembered even if the user
    switches away before asking another question."""
    if st.session_state.messages:
        conversations.save_conversation(
            st.session_state.session_id,
            st.session_state.messages,
            docs=_current_doc_refs(),
        )


def start_new_chat():
    """Abandons the current chat view and starts a completely fresh one
    — like ChatGPT's "New Chat": no carried-over documents, no
    carried-over messages. The old conversation isn't deleted — it's
    already saved on disk (if it had any messages) and stays in the
    sidebar list."""
    clear_session(st.session_state.session_id)  # drop old short-term rewriting context
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.uploaded_docs = []
    st.session_state.pending_files = []
    st.session_state.vector_store = VectorStore(dim=EMBEDDING_DIM)
    st.session_state.doc_registry = {}


def switch_to_conversation(conversation_id: str):
    """Loads a previously saved conversation — including silently
    restoring whichever of its documents are still in the on-disk
    cache, with no re-upload and no reprocessing. This is what makes
    "resume an old chat" actually work rather than just look like it
    works: without this, the message history would reappear but the
    active document set would still be whatever's already loaded (or
    empty, after a restart), so a follow-up question would search the
    wrong documents or none at all.

    A document that's no longer in the cache (its cached copy was
    deleted) is skipped rather than failing the whole restore; the
    caller is notified via st.session_state["pending_notice"] so it can
    be shown after the rerun this triggers."""
    saved = conversations.load_conversation(conversation_id)
    if saved is None:
        st.session_state["pending_notice"] = ("error", "That conversation is no longer available.")
        return

    st.session_state.session_id = conversation_id
    st.session_state.messages = saved["messages"]

    st.session_state.vector_store = VectorStore(dim=EMBEDDING_DIM)
    st.session_state.doc_registry = {}
    st.session_state.uploaded_docs = []

    missing_names = []
    for ref in saved["docs"]:
        try:
            doc_info = ingest.restore_from_cache(
                ref["content_hash"], st.session_state.vector_store, st.session_state.doc_registry
            )
            st.session_state.uploaded_docs.append(doc_info)
        except ValueError:
            missing_names.append(ref.get("filename") or "a document")

    if missing_names:
        names = ", ".join(missing_names)
        was_were = "was" if len(missing_names) == 1 else "were"
        it_them = "it" if len(missing_names) == 1 else "them"
        st.session_state["pending_notice"] = (
            "warning",
            f"{names} {was_were} used in this conversation but no longer available in the "
            f"cache — questions relying on {it_them} may not work until re-uploaded.",
        )

    # Replay into the short-term rewriting memory so an immediate
    # follow-up question has context right away, not just after the
    # next turn.
    clear_session(conversation_id)
    messages = saved["messages"]
    for i in range(0, len(messages) - 1, 2):
        if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
            append_turn(
                conversation_id,
                messages[i].get("content", ""),
                messages[i + 1].get("content", ""),
                messages[i + 1].get("sources", []),
            )


# ------------------------------------------------------------------
# Session state bootstrap
# ------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "pending_files" not in st.session_state:
    st.session_state.pending_files = []  # Files to upload but not yet indexed

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(dim=EMBEDDING_DIM)

if "doc_registry" not in st.session_state:
    st.session_state.doc_registry = {}


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div class='app-title'>DocQuery<span>.</span></div>"
        "<div class='app-subtitle'>Grounded answers from your own documents.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Chats ---------------------------------------------------
    if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True):
        start_new_chat()
        st.rerun()

    recent_chats = conversations.list_conversations()
    if recent_chats:
        active_id = st.session_state.session_id
        if any(c["conversation_id"] == active_id for c in recent_chats):
            # Highlight whichever row is currently open. Injected per-render
            # (not a static rule) since only this code knows which id is
            # active right now — same .st-key-<key> hook as everywhere else.
            st.markdown(
                f"<style>[class*='st-key-conv_{active_id}'] button {{"
                f"background: var(--surface); border-color: var(--sage-border);"
                f"border-left: 3px solid var(--sage); font-weight: 500; }}</style>",
                unsafe_allow_html=True,
            )
        for c in recent_chats:
            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"{c['title']}  ·  {c['relative_time']}"
                if st.button(label, key=f"conv_{c['conversation_id']}", use_container_width=True):
                    switch_to_conversation(c["conversation_id"])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_conv_{c['conversation_id']}", help="Delete this conversation"):
                    conversations.delete_conversation(c["conversation_id"])
                    if c["conversation_id"] == st.session_state.session_id:
                        start_new_chat()
                    st.rerun()

    st.divider()

    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Select documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption("PDF · DOCX · XLSX · TXT · Markdown · CSV · HTML · PNG/JPG — scanned pages and photos use OCR automatically.")

    # Store selected files in pending state. Identity is content-based
    # (a hash), not just the filename, so a modified file re-uploaded
    # under its old name is correctly treated as new content.
    if uploaded_files:
        for uf in uploaded_files:
            content = uf.getvalue()
            content_hash = cache.compute_content_hash(content)
            existing = next((d for d in st.session_state.uploaded_docs if d["filename"] == uf.name), None)

            if existing and existing.get("content_hash") == content_hash:
                continue  # identical file is already active — nothing to do

            already_pending = any(
                p["name"] == uf.name and p["hash"] == content_hash
                for p in st.session_state.pending_files
            )
            if not already_pending:
                st.session_state.pending_files.append({
                    "name": uf.name,
                    "content": content,
                    "hash": content_hash,
                    "size": len(content),
                    "replaces_existing": existing is not None,
                })

    # Show pending files
    if st.session_state.pending_files:
        st.markdown("##### Ready to process:")
        for idx, pf in enumerate(st.session_state.pending_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                label = f"📄 {pf['name']} ({(pf['size']/1024):.1f} KB)"
                if pf.get("replaces_existing"):
                    label += " · replaces active version"
                st.markdown(label)
            with col2:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.pending_files.pop(idx)
                    st.rerun()

        if st.button("✅ Process Documents", key="process_documents_btn", use_container_width=True, type="primary"):
            for pf in st.session_state.pending_files[:]:
                with st.spinner(f"Processing {pf['name']}…"):
                    try:
                        if pf.get("replaces_existing"):
                            st.session_state.vector_store.remove_doc(pf["name"])
                            st.session_state.uploaded_docs = [
                                d for d in st.session_state.uploaded_docs if d["filename"] != pf["name"]
                            ]
                            st.session_state.doc_registry.pop(pf["name"], None)

                        doc_info = ingest.process_upload(
                            pf["content"],
                            pf["name"],
                            st.session_state.vector_store,
                            st.session_state.doc_registry,
                        )
                        st.session_state.uploaded_docs.append(doc_info)
                        st.session_state.pending_files.remove(pf)

                        if doc_info["source"] == "cache":
                            st.success(f"⚡ {pf['name']} — loaded from cache")
                        else:
                            note = " (OCR used)" if doc_info.get("ocr_used") else ""
                            st.success(f"✅ {pf['name']} — processed{note}")
                    except Exception as e:
                        st.error(f"❌ {pf['name']}: {str(e)}")
            _autosave_conversation()
            st.rerun()

    st.divider()
    st.markdown("### Active Documents")

    if st.session_state.uploaded_docs:
        for idx, doc in enumerate(st.session_state.uploaded_docs):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(doc_card_html(doc), unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"remove_doc_{idx}", help="Remove from this session — any cached copy on disk is kept"):
                    st.session_state.vector_store.remove_doc(doc["filename"])
                    st.session_state.uploaded_docs.pop(idx)
                    st.session_state.doc_registry.pop(doc["filename"], None)
                    _autosave_conversation()
                    st.rerun()
    else:
        st.info("No documents active yet.")

    st.divider()

    active_hashes = {d.get("content_hash") for d in st.session_state.uploaded_docs}
    cached_docs = [c for c in cache.list_cached_documents() if c.get("compatible", True)]

    with st.expander(f"📦 Previously processed documents ({len(cached_docs)})"):
        if not cached_docs:
            st.caption("Nothing cached yet — processed documents will appear here for reuse.")
        else:
            for c in cached_docs:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(doc_card_html(c, dormant=True), unsafe_allow_html=True)
                with col2:
                    if c["content_hash"] in active_hashes:
                        st.caption("active")
                    elif st.button("↺", key=f"restore_{c['content_hash']}", help="Restore into this session"):
                        try:
                            doc_info = ingest.restore_from_cache(
                                c["content_hash"],
                                st.session_state.vector_store,
                                st.session_state.doc_registry,
                            )
                            st.session_state.uploaded_docs.append(doc_info)
                            restored = True
                        except Exception as e:
                            st.error(str(e))
                            restored = False
                        if restored:
                            _autosave_conversation()
                            st.rerun()
                with col3:
                    if st.button("🗑️", key=f"delete_cache_{c['content_hash']}", help="Delete this cached copy from disk"):
                        cache.delete_cached_document(c["content_hash"])
                        st.rerun()

    st.divider()

    with st.expander("⚠️ Danger zone"):
        st.caption("Permanently deletes every cached document from disk. Anything still active in this session keeps working until you start a new chat or restart the app, but nothing can be restored from cache afterward.")
        if st.button("Delete ALL cached data", key="delete_all_cache_btn"):
            cache.clear_all_cache()
            st.success("Cache cleared.")
            st.rerun()

        st.caption("Permanently deletes every saved chat conversation. The chat currently open keeps working, but it won't be saved or reappear in this list once you leave it.")
        if st.button("Delete ALL chat history", key="delete_all_chats_btn"):
            conversations.clear_all_conversations()
            st.success("Chat history cleared.")
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

if "pending_notice" in st.session_state:
    kind, text = st.session_state.pop("pending_notice")
    getattr(st, kind)(text)

if not st.session_state.uploaded_docs:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <div style='font-size: 40px; margin-bottom: 16px;'>📄</div>
        <div style='font-family:"Fraunces",serif; font-size: 20px; font-weight: 600; color: var(--text);'>No documents loaded</div>
        <div style='font-size: 14px; margin-top: 8px; color: var(--text-muted);'>Upload documents from the sidebar to begin chatting.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            badges = "".join(
                f"<span class='source-badge'>{src}</span>"
                for src in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{badges}</div>", unsafe_allow_html=True)

        if msg["role"] == "assistant" and msg.get("rewritten"):
            orig = msg.get("original_question", "")
            rw = msg["rewritten"]
            if orig and rw.lower().strip() != orig.lower().strip():
                st.markdown(
                    f"<div class='hint-text'>🔁 Interpreted as: <i>{rw}</i></div>",
                    unsafe_allow_html=True,
                )

        if msg["role"] == "assistant" and msg.get("confidence"):
            st.markdown(
                f"<div class='hint-text'>🎯 Confidence: {msg['confidence']}</div>",
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                history = get_last_n_turns(st.session_state.session_id, n=4)
                
                result = answer_question(
                    question=prompt,
                    vector_store=st.session_state.vector_store,
                    history=history,
                )
                
                answer = result["answer"]
                sources = result["sources"]
                rewritten = result["rewritten_query"]
                confidence = result["confidence"]
                
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
                confidence = ""

        st.markdown(answer)

        if sources:
            badges = "".join(
                f"<span class='source-badge'>{src}</span>"
                for src in sources
            )
            st.markdown(f"<div style='margin-top:6px'>{badges}</div>", unsafe_allow_html=True)

        if rewritten and rewritten.lower().strip() != prompt.lower().strip():
            st.markdown(
                f"<div class='hint-text'>🔁 Interpreted as: <i>{rewritten}</i></div>",
                unsafe_allow_html=True,
            )

        if confidence:
            st.markdown(
                f"<div class='hint-text'>🎯 Confidence: {confidence}</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "rewritten": rewritten,
        "confidence": confidence,
        "original_question": prompt,
    })

    _autosave_conversation()
