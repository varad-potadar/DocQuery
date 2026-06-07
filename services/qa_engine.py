"""
services/qa_engine.py

Redesigned QA engine:
  1. Rewrite question to be self-contained (resolves pronouns / references)
  2. Embed rewritten query
  3. Hybrid retrieval (semantic + keyword) from vector store
  4. Build context with source attribution
  5. Call Groq LLM with grounding prompt
  6. Return answer + sources
"""

import os
import re
from groq import Groq
from typing import List, Dict, Tuple

from services.embedder import embed_chunks
from services.query_rewriter import rewrite_query

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _clean_chunk(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _build_context_block(chunks: List[Dict]) -> Tuple[str, List[str]]:
    """
    Formats retrieved chunks into a numbered context block.
    Returns (context_string, list_of_source_doc_ids).
    """
    parts = []
    sources_seen = []

    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc_id", "unknown")
        text = _clean_chunk(chunk["text"])
        parts.append(f"[{i}] (Source: {doc})\n{text}")

        if doc not in sources_seen:
            sources_seen.append(doc)

    return "\n\n".join(parts), sources_seen


def _build_history_text(history: List[Dict]) -> str:
    if not history:
        return ""
    return "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def answer_question(
    question: str,
    vector_store,
    history: List[Dict] = None,
) -> Dict:
    """
    Returns:
        {
            "answer":          str,
            "sources":         List[str],   # doc_ids used
            "rewritten_query": str,         # for transparency / debugging
        }
    """
    if history is None:
        history = []

    # 1. Rewrite question (resolve pronouns / references)
    rewritten = rewrite_query(question, history)
    print(f"\n[qa_engine] Original:  {question}")
    print(f"[qa_engine] Rewritten: {rewritten}")

    # 2. Embed rewritten query
    q_embedding = embed_chunks([rewritten])

    # 3. Determine retrieval depth
    q_lower = rewritten.lower()
    broad_keywords = ["summary", "overview", "explain", "describe",
                      "limitation", "challenge", "problem", "contribution",
                      "method", "approach", "compare", "difference",
                      "future", "conclusion", "finding"]
    k = 20 if any(kw in q_lower for kw in broad_keywords) else 12

    # 4. Hybrid retrieval
    retrieved = vector_store.search(q_embedding, query_text=rewritten, k=k)

    if not retrieved:
        return {
            "answer": "No documents have been indexed yet. Please upload a document first.",
            "sources": [],
            "rewritten_query": rewritten,
        }

    print(f"[qa_engine] Retrieved {len(retrieved)} chunks from: "
          f"{list(dict.fromkeys(c['doc_id'] for c in retrieved))}")

    # 5. Build context
    context, sources = _build_context_block(retrieved)

    # 6. Build conversation history text (last 3 turns)
    history_text = _build_history_text(history[-6:])

    # 7. Prompt
    prompt = f"""You are a document-grounded assistant.

Your job is to answer the user's question strictly using the provided context.

Rules:
- Answer ONLY from the context. Do not use outside knowledge.
- If the context contains the answer, answer confidently and completely.
- If information spans multiple sources, combine it and mention the sources by name.
- If the answer is genuinely not in the context, say exactly: "This information was not found in the uploaded documents."
- Use conversation history only to understand what pronouns like "it", "they", "its" refer to.
- Do not repeat the question back.
- Be direct. Start with the answer.

Context (numbered chunks with source document):
{context}

Conversation history:
{history_text}

Question: {question}

Answer:"""

    # 8. LLM call
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise, document-grounded assistant. "
                        "Answer strictly from the provided context. "
                        "Never fabricate information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"LLM call failed: {e}"
        print(f"[qa_engine] ERROR: {e}")

    print(f"[qa_engine] Answer (first 200 chars): {answer[:200]}")

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
    }
