"""
services/query_rewriter.py  (NEW FILE)

Rewrites a user question to be self-contained, using recent conversation
history to resolve pronouns and references.

Example:
    History:  "What method is proposed?" → "CNN-based approach"
    Question: "What are its limitations?"
    Rewritten: "What are the limitations of the CNN-based approach?"

Uses Groq with a lightweight, fast model. Falls back to the original
question if the call fails or the question is already self-contained.
"""

import os
import re
from groq import Groq
from typing import List, Dict

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _needs_rewriting(question: str, history: List[Dict]) -> bool:
    """
    Quick heuristic: only rewrite if there IS prior history AND the question
    contains likely referential expressions.
    """
    if not history:
        return False

    q = question.lower()
    referential_signals = [
        r"\bit\b", r"\bits\b", r"\bthey\b", r"\btheir\b", r"\bthis\b",
        r"\bthat\b", r"\bthese\b", r"\bthose\b", r"\bhe\b", r"\bshe\b",
        r"\bthe method\b", r"\bthe approach\b", r"\bthe system\b",
        r"\bthe model\b", r"\bthe paper\b", r"\bthe document\b",
        r"\bsame\b", r"\babove\b", r"\bmentioned\b", r"\bprevious\b",
    ]
    return any(re.search(pat, q) for pat in referential_signals)


def rewrite_query(question: str, history: List[Dict]) -> str:
    """
    Returns a rewritten, self-contained version of the question.
    If rewriting is not needed or fails, returns original question unchanged.
    """
    if not _needs_rewriting(question, history):
        return question

    # Build a short history summary for the prompt (last 3 turns)
    recent = history[-6:]
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in recent
    )

    prompt = f"""You are a question rewriter. Your only job is to rewrite the FOLLOW-UP QUESTION to be fully self-contained, resolving any pronouns or references using the conversation history.

Rules:
- Output ONLY the rewritten question, nothing else.
- Do NOT answer the question.
- If the question is already self-contained, output it unchanged.
- Keep the rewritten question concise.

Conversation history:
{history_text}

Follow-up question: {question}

Rewritten question:"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()

        # Sanity check: result should be a question, not an answer
        if len(rewritten) > 10 and len(rewritten) < 300:
            return rewritten
        return question

    except Exception as e:
        print(f"[query_rewriter] Rewrite failed: {e}. Using original.")
        return question
