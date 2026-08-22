"""
services/qa_engine.py

Redesigned QA engine:
  1. Rewrite question to be self-contained (resolves pronouns / references)
  2. Embed rewritten query
  3. Hybrid retrieval (semantic + keyword) from vector store
  4. Rerank the retrieved chunks with a cross-encoder
  5. Build context with source attribution
  6. Call Gemini LLM with grounding prompt
  7. Return answer + sources + a rough confidence label
"""

import math
import os
import re
import time
from google import genai
from google.genai import types
from typing import List, Dict, Tuple

from services.embedder import embed_chunks
from services.query_rewriter import rewrite_query
from services.reranker import rerank_chunks

_client = None

GEMINI_MODEL = "gemini-3.6-flash"

# ------------------------------------------------------------------
# Token budget
# ------------------------------------------------------------------
# Gemini's free tier for gemini-2.5-flash caps requests well under a
# generous tokens/minute limit, but we still stay under a conservative
# budget so a single call never gets rejected with a rate-limit error.
#
# ~4 chars/token is a standard rough estimate for English text and is
# good enough for budgeting purposes (we don't need exact tokenization).
MODEL_TPM_LIMIT = 6000
RESPONSE_TOKENS = 800          # max_tokens we request back
SAFETY_MARGIN = 400            # buffer for system prompt + instructions + rounding
MAX_CONTEXT_TOKENS = MODEL_TPM_LIMIT - RESPONSE_TOKENS - SAFETY_MARGIN  # ~4800
CHARS_PER_TOKEN = 4

NOT_FOUND_SENTINEL = "This information was not found in the uploaded documents."

# Rough buckets for the confidence label -- ms-marco-MiniLM isn't trained
# as a calibrated classifier, so these thresholds (applied to a sigmoid of
# its raw cross-encoder score) are a useful signal, not a real probability.
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.5


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _confidence_label(top_rerank_score: float) -> str:
    """
    Turns the top reranked chunk's raw cross-encoder score into a rough
    "high" / "medium" / "low" label via a sigmoid + thresholds. This is a
    heuristic to flag "the best evidence we found was weak", not a
    calibrated probability -- treat it as a hint, not a guarantee.
    """
    p = 1.0 / (1.0 + math.exp(-top_rerank_score))
    if p >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if p >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _clean_chunk(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _build_context_block(
    chunks: List[Dict],
    max_tokens: int = None,
) -> Tuple[str, List[str], int]:
    """
    Formats retrieved chunks into a numbered context block.

    chunks is expected to already be ordered most-relevant-first (true for
    both retrieval modes in vector_store.search). If max_tokens is given,
    chunks are added in that order until the budget would be exceeded,
    then the rest are dropped -- so a broad query against a large indexed
    set degrades to "the most relevant chunks that fit" instead of
    blowing the LLM provider's per-request token limit.

    Returns (context_string, list_of_source_doc_ids, chunks_used_count).
    """
    parts = []
    sources_seen = []
    running_tokens = 0
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc_id", "unknown")
        page = chunk.get("page")
        heading = chunk.get("heading")
        location = doc
        if page:
            location += f", p.{page}"
        elif heading:
            location += f", section: {heading}"

        text = _clean_chunk(chunk["text"])
        piece = f"[{i}] (Source: {location})\n{text}"

        if max_tokens is not None:
            piece_tokens = _estimate_tokens(piece)
            if running_tokens + piece_tokens > max_tokens and parts:
                break
            running_tokens += piece_tokens

        parts.append(piece)
        used += 1

        if doc not in sources_seen:
            sources_seen.append(doc)

    return "\n\n".join(parts), sources_seen, used


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
            "confidence":      str,         # "high" / "medium" / "low" -- see
                                             # _confidence_label(); heuristic, not
                                             # a calibrated probability
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
            "confidence": "low",
        }

    print(f"[qa_engine] Retrieved {len(retrieved)} chunks from: "
          f"{list(dict.fromkeys(c['doc_id'] for c in retrieved))}")

    # 4b. Rerank: a cross-encoder scores each retrieved chunk jointly
    #     against the query -- a meaningfully better relevance signal
    #     than the fused dense+BM25 rank used to build the candidate set
    #     above -- and keeps only the strongest few (rerank_chunks
    #     defaults to the top 8).
    reranked = rerank_chunks(rewritten, retrieved)
    print(f"[qa_engine] Reranked {len(retrieved)} -> {len(reranked)} chunks")

    confidence = _confidence_label(reranked[0].get("rerank_score", 0.0))

    # 5. Build context, capped to a token budget so the request can never
    #    exceed the LLM provider's tokens-per-minute limit regardless of
    #    how many chunks were retrieved. `reranked` is relevance-ordered,
    #    so trimming drops the least-relevant chunks first.
    history_reserve = _estimate_tokens(_build_history_text(history[-4:])) if history else 0
    context_budget = max(500, MAX_CONTEXT_TOKENS - history_reserve)
    context, sources, chunks_used = _build_context_block(reranked, max_tokens=context_budget)

    if chunks_used < len(retrieved):
        print(f"[qa_engine] Context budget reached: using {chunks_used}/{len(retrieved)} "
              f"retrieved chunks (~{context_budget} token budget)")

    # 6. Build conversation history text (last 2 turns -- kept short to
    #    leave more of the budget for document context, which matters more)
    history_text = _build_history_text(history[-4:])

    # 7. Prompt
    prompt = f"""You are a document-grounded assistant.

Your job is to answer the user's question strictly using the provided context.

Rules:
- Answer ONLY from the context. Do not use outside knowledge.
- If the context contains the answer, answer confidently and completely.
- If information spans multiple sources, combine it and mention the sources by name.
- If the answer is genuinely not in the context, say exactly: "{NOT_FOUND_SENTINEL}"
- Use conversation history only to understand what pronouns like "it", "they", "its" refer to.
- Do not repeat the question back.
- Be direct. Start with the answer.

Context (numbered chunks with source document):
{context}

Conversation history:
{history_text}

Question: {question}

Answer:"""

    # 8. LLM call, with exponential-backoff retry for transient
    #    rate-limit errors (429 from Gemini's rate limiter).
    max_retries = 3
    answer = None
    for attempt in range(max_retries):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a precise, document-grounded assistant. "
                        "Answer strictly from the provided context. "
                        "Never fabricate information."
                    ),
                    temperature=0.0,
                    max_output_tokens=RESPONSE_TOKENS,
                ),
            )
            answer = response.text.strip()
            break

        except Exception as e:
            is_rate_limit = "rate_limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[qa_engine] Rate limited (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait}s: {e}")
                time.sleep(wait)
                continue
            answer = f"LLM call failed: {e}"
            print(f"[qa_engine] ERROR: {e}")
            break

    if answer.strip() == NOT_FOUND_SENTINEL:
        confidence = "low"

    print(f"[qa_engine] Answer (first 200 chars): {answer[:200]}")
    print(f"[qa_engine] Confidence: {confidence}")

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
        "confidence": confidence,
    }
