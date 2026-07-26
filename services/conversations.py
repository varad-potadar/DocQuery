"""
services/conversations.py

On-disk persistence for chat conversations, so past chats survive an
app restart and can be resumed later — the same idea as
services/cache.py, applied to chat threads instead of documents.

Each conversation is one small folder:

    data/conversations/<conversation_id>/meta.json      title, timestamps, message count, docs
    data/conversations/<conversation_id>/messages.json  the message list shown in the chat UI

A conversation is only written to disk once it has at least one
message — opening a new chat and never sending anything doesn't create
a sidebar entry, matching how most chat apps behave.

Each conversation remembers which documents (by content hash — see
services/cache.py) were active while it was in use. This is the
detail that makes "resume an old chat" actually work rather than just
looking like it works: switching to a saved conversation restores its
documents from the on-disk cache with no re-upload and no
reprocessing, even after a full app restart, because the cache and the
conversation record both key off the same content hash. A document
that's since been removed from the cache is skipped rather than
failing the whole restore — see app.py's switch_to_conversation().
"""

import json
import os
import shutil
import time
import uuid
from typing import Dict, List, Optional

from services.config import MAX_CONVERSATIONS_LISTED, CONVERSATION_TITLE_MAX_LEN

CONV_DIR = os.path.join("data", "conversations")


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def _entry_dir(conversation_id: str) -> str:
    return os.path.join(CONV_DIR, conversation_id)


def _make_title(messages: List[Dict]) -> str:
    """Auto-titles from the first user message, like most chat apps do
    before you rename one — this app doesn't support manual rename yet."""
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            text = " ".join(m["content"].split())  # collapse whitespace/newlines
            if len(text) > CONVERSATION_TITLE_MAX_LEN:
                return text[:CONVERSATION_TITLE_MAX_LEN].rstrip() + "…"
            return text
    return "New conversation"


def _format_relative_time(timestamp: float) -> str:
    """Small, dependency-free relative-time label for the sidebar."""
    delta = max(0, time.time() - timestamp)
    if delta < 60:
        return "just now"
    minutes = delta / 60
    if minutes < 60:
        return f"{int(minutes)} min ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hr ago"
    days = hours / 24
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{int(days)} days ago"
    return time.strftime("%b %d", time.localtime(timestamp))


def save_conversation(
    conversation_id: str,
    messages: List[Dict],
    docs: Optional[List[Dict]] = None,
) -> None:
    """
    Persists the full message list for a conversation, plus which
    documents it uses. Meant to be called after every turn (the caller
    already holds the full message list in memory), so a conversation
    is never left half-saved. Does nothing if messages is empty — an
    empty chat isn't persisted.

    docs: [{"content_hash": str, "filename": str}, ...] — the
    documents active in the session when this was saved. Restoring
    them later only needs the hash (see services.cache /
    services.ingest.restore_from_cache); filename is kept alongside
    purely so a missing document can be named in a warning rather than
    just counted.
    """
    if not messages:
        return

    entry = _entry_dir(conversation_id)
    os.makedirs(entry, exist_ok=True)

    meta_path = os.path.join(entry, "meta.json")
    created_at = time.time()
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                created_at = json.load(f).get("created_at", created_at)
        except Exception:
            pass

    meta = {
        "conversation_id": conversation_id,
        "title": _make_title(messages),
        "created_at": created_at,
        "updated_at": time.time(),
        "message_count": len(messages),
        "docs": docs or [],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(entry, "messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)


def append_messages(
    conversation_id: str,
    new_messages: List[Dict],
    docs: Optional[List[Dict]] = None,
) -> None:
    """
    Loads the existing conversation (if any), appends new_messages, and
    saves. For callers (like the FastAPI endpoint) that don't hold the
    full message list themselves the way the Streamlit UI does.

    If docs isn't passed, whatever docs were already recorded for this
    conversation are kept as-is rather than being cleared.
    """
    existing = load_conversation(conversation_id)
    prior_messages = existing["messages"] if existing else []
    prior_docs = existing["docs"] if existing else []
    save_conversation(
        conversation_id,
        prior_messages + new_messages,
        docs=docs if docs is not None else prior_docs,
    )


def load_conversation(conversation_id: str) -> Optional[Dict]:
    """Returns {"messages": [...], "docs": [{"content_hash","filename"}, ...]},
    or None if this conversation doesn't exist / can't be read."""
    entry = _entry_dir(conversation_id)
    messages_path = os.path.join(entry, "messages.json")
    if not os.path.exists(messages_path):
        return None
    try:
        with open(messages_path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        docs = []
        meta_path = os.path.join(entry, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                docs = json.load(f).get("docs", [])

        return {"messages": messages, "docs": docs}
    except Exception as e:
        print(f"[conversations] Failed to load {conversation_id[:8]}: {e}")
        return None


def list_conversations(limit: int = MAX_CONVERSATIONS_LISTED) -> List[Dict]:
    """Most recently updated first, with a friendly relative-time label
    already attached — for the sidebar list."""
    if not os.path.isdir(CONV_DIR):
        return []

    entries = []
    for conversation_id in os.listdir(CONV_DIR):
        meta_path = os.path.join(_entry_dir(conversation_id), "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["relative_time"] = _format_relative_time(meta.get("updated_at", 0))
            meta["doc_count"] = len(meta.get("docs", []))
            entries.append(meta)
        except Exception:
            continue

    entries.sort(key=lambda m: m.get("updated_at", 0), reverse=True)
    return entries[:limit]


def delete_conversation(conversation_id: str) -> bool:
    entry = _entry_dir(conversation_id)
    if os.path.isdir(entry):
        shutil.rmtree(entry)
        return True
    return False


def clear_all_conversations() -> None:
    if os.path.isdir(CONV_DIR):
        shutil.rmtree(CONV_DIR)
    os.makedirs(CONV_DIR, exist_ok=True)
