"""
services/memory.py

In-process session store.
Stores full turn history per session_id so qa_engine can use it for
query rewriting and context injection.
"""

from typing import List, Dict

_sessions: Dict[str, List[Dict]] = {}


def get_history(session_id: str) -> List[Dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str, sources: List[str] = None):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": "user",      "content": question})
    _sessions[session_id].append({"role": "assistant",  "content": answer,
                                   "sources": sources or []})


def clear_session(session_id: str):
    _sessions[session_id] = []


def get_last_n_turns(session_id: str, n: int = 3) -> List[Dict]:
    """Returns last n complete turns (user+assistant pairs)."""
    history = get_history(session_id)
    # Each turn = 2 messages; return last n*2 messages
    return history[-(n * 2):]
