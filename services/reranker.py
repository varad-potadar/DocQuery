"""
services/reranker.py

Cross-encoder reranking. Hybrid retrieval (dense + BM25) is fast but
scores query and chunk separately, then compares vectors -- a
cross-encoder reads the query and chunk together in one pass, which is
slower but noticeably more accurate at judging "is this chunk actually
relevant to this question."

This module only RE-ORDERS what retrieval already found; it never drops
a chunk. That matters because vector_store.search()'s small-corpus mode
deliberately returns every chunk so nothing is excluded before the LLM
sees it (see that module's docstring) -- the token-budget trim in
qa_engine.py is still the only place a chunk actually gets cut, now
acting on a much better-ordered list.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 -- small (~80MB), fast on
CPU, free, and light enough to fit alongside everything else already
loaded on a resource-capped host like Streamlit Community Cloud's free
tier (~1GB RAM). If you're hosting somewhere with more headroom and
want higher accuracy, BAAI/bge-reranker-v2-m3 scores better -- just
swap RERANKER_MODEL_NAME (it's a larger download and slower per call).
"""

from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank(query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
    """
    Re-scores `chunks` against `query` with a cross-encoder and returns
    them re-ordered, most relevant first.

    Adds a "rerank_score" field to each returned chunk. The retriever's
    own "score" field (dense/BM25/RRF, whatever it was) is left alone,
    in case it's useful for debugging.

    top_k: pass a number to also truncate to the best N chunks. Pass
    None (the default) to reorder only -- this is what qa_engine.py
    uses, so token-budget trimming downstream stays the single place
    that removes a chunk from the final context.

    Falls back to returning `chunks` in their original order if the
    model can't be loaded or scoring fails (e.g. no network on first
    run to fetch the model), so a broken/missing dependency degrades to
    "no reranking" rather than breaking the app.
    """
    if not chunks:
        return chunks

    try:
        model = _get_model()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)
    except Exception as e:
        print(f"[reranker] Failed ({e}); falling back to retrieval order.")
        return chunks[:top_k] if top_k is not None else chunks

    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    return scored[:top_k] if top_k is not None else scored
