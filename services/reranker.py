"""
services/reranker.py

Cross-encoder reranking. vector_store.search() already casts a wide net
(the k=12/20 split in qa_engine, widened further by search()'s own
fetch_k = k * 4); a cross-encoder scores the query and each candidate
chunk together in a single forward pass, which is a meaningfully better
relevance judgment than the fused dense+BM25 rank used to build that
candidate set -- at the cost of being too slow to run over a whole
corpus, which is exactly why it only runs on the shortlist here rather
than replacing search() as the primary retrieval step.

Reuses chunk["index_text"] when present -- the same doc-title +
heading-prefixed text used for embedding/BM25 (see
chunker.contextual_text) -- falling back to the clean chunk["text"] for
any chunk that predates that field. Returns a new, shorter list of new
dicts (never mutates the input), each carrying a "rerank_score" field --
the raw cross-encoder score, used by qa_engine to derive a rough
confidence label.
"""

from typing import Dict, List

from sentence_transformers import CrossEncoder

# ~90MB, CPU-friendly -- fits alongside the embedding model within
# Streamlit Community Cloud's free-tier RAM. BAAI/bge-reranker-v2-m3
# scores meaningfully higher if this ever moves to a host with more
# headroom -- same .predict() interface, just swap the model name below.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 8

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    if not chunks:
        return chunks

    model = _get_model()
    pairs = [(query, c.get("index_text", c["text"])) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**c, "rerank_score": float(score)} for c, score in ranked[:top_k]]
