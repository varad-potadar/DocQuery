"""
services/qa_engine.py

Redesigned QA engine:
  1. Rewrite question to be self-contained (resolves pronouns / references)
  2. Embed rewritten query
  3. Hybrid retrieval (semantic + keyword) from vector store, fetched
     wider than needed so reranking (next) has real candidates to pick from
  4. Cross-encoder rerank -- reorders by true relevance, drops nothing
  5. Build context with source attribution
  6. Call Gemini LLM with grounding prompt
  7. Return answer + sources
"""

import os
import re
import time
from google import genai
from google.genai import types
from typing import List, Dict, Tuple

from services.embedder import embed_chunks
from services.query_rewriter import rewrite_query
from services.reranker import rerank

_client = None

GEMINI_MODEL = "gemini-2.5-flash"

# ------------------------------------------------------------------
# Token budget
# ------------------------------------------------------------------
# These were sized for Groq's free-tier limit (6,000 TPM total) from
# before this app switched to Gemini -- that's the main reason answers
# were getting cut off mid-sentence: RESPONSE_TOKENS=800 was a hard
# ceiling on the model's OUTPUT, not just a rate-limit safety margin.
#
# There's a second, less obvious way that same ceiling got hit even
# earlier than 800 tokens of visible text: gemini-2.5-flash "thinks"
# before answering by default, and that invisible reasoning is billed
# against the SAME max_output_tokens ceiling as the visible answer --
# so the budget could be gone before the model writes a word the user
# sees. Thinking is turned off below (see the LLM call) since this is
# a "answer from the given context" task, not one that benefits much
# from extended reasoning -- RESPONSE_TOKENS is still sized generously
# here as a second layer of safety in case that's ever not honored.
#
# Gemini 2.5 Flash's free tier is far larger than Groq's was (250,000+
# tokens/minute, a 1M-token context window), so there's no rate-limit
# reason to keep these numbers small -- they're chosen for a complete
# answer and a generous retrieval context, not because Gemini would
# reject anything bigger.
#
# ~4 chars/token is a standard rough estimate for English text and is
# good enough for budgeting purposes (we don't need exact tokenization).
RESPONSE_TOKENS = 4096      # max output tokens -- room for a full answer
MAX_CONTEXT_TOKENS = 12000  # retrieved-chunk budget for the prompt
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


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
    final_k = 20 if any(kw in q_lower for kw in broad_keywords) else 12
    # Fetch wider than final_k so reranking below has real candidates to
    # choose from instead of just whatever dense+BM25 fusion ranked first.
    # (No effect on vector_store's small-corpus mode -- that path already
    # returns every chunk regardless of k.)
    fetch_k = min(final_k * 3, 60)

    # 4. Hybrid retrieval
    retrieved = vector_store.search(q_embedding, query_text=rewritten, k=fetch_k)

    if not retrieved:
        return {
            "answer": "No documents have been indexed yet. Please upload a document first.",
            "sources": [],
            "rewritten_query": rewritten,
        }

    print(f"[qa_engine] Retrieved {len(retrieved)} chunks from: "
          f"{list(dict.fromkeys(c['doc_id'] for c in retrieved))}")

    # 4b. Cross-encoder rerank -- reorders by true relevance to the
    #     (rewritten) query. Reorders only (top_k=None); the token-budget
    #     trim below is still what actually drops a chunk.
    retrieved = rerank(rewritten, retrieved)

    # 5. Build context, capped to a token budget so the request can never
    #    exceed the LLM provider's tokens-per-minute limit regardless of
    #    how many chunks were retrieved. `retrieved` is relevance-ordered,
    #    so trimming drops the least-relevant chunks first.
    history_reserve = _estimate_tokens(_build_history_text(history[-4:])) if history else 0
    context_budget = max(500, MAX_CONTEXT_TOKENS - history_reserve)
    context, sources, chunks_used = _build_context_block(retrieved, max_tokens=context_budget)

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
                    # Off, on purpose (see the token-budget comment above):
                    # this is grounded extraction/synthesis, not the kind of
                    # multi-step problem thinking mode is for, and leaving
                    # it on was silently consuming the same token budget as
                    # the visible answer.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            candidate = response.candidates[0] if response.candidates else None
            finish_reason = (
                candidate.finish_reason.name
                if candidate and candidate.finish_reason else None
            )
            try:
                answer = (response.text or "").strip()
            except Exception:
                # The SDK can raise here instead of returning "" when a
                # response has no text part at all -- seen when even
                # "thinking" (before it's disabled, or on a retry with a
                # different model) ate the whole token budget.
                answer = ""

            if finish_reason == "MAX_TOKENS":
                print(f"[qa_engine] WARNING: hit MAX_TOKENS at RESPONSE_TOKENS="
                      f"{RESPONSE_TOKENS} with thinking disabled -- this "
                      f"question's answer needs more room than that.")
                answer = (
                    answer + "\n\n*(This answer was cut short by a length "
                    "limit -- ask me to continue.)*"
                ) if answer else (
                    "The model ran out of room before it could write an "
                    "answer. Try asking a narrower or more specific question."
                )
            elif not answer:
                answer = f"The model returned an empty response (finish_reason: {finish_reason})."

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

    print(f"[qa_engine] Answer (first 200 chars): {answer[:200]}")

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
    }
