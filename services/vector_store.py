"""
services/vector_store.py

Hybrid vector store:
  - FAISS (flat L2) for dense semantic search
  - BM25-style keyword scoring via rank_bm25
  - Results are combined with Reciprocal Rank Fusion (RRF)

Each stored item:
    {
        "text":       str,
        "doc_id":     str,
        "chunk_index": int,
        "char_start": int,
    }
"""

import numpy as np
import faiss
from typing import List, Dict, Optional

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _rrf(ranks: List[List[int]], k: int = 60) -> List[float]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    Each list contains item indices (0-based into self.data).
    Returns a fused score per index position.
    """
    scores: Dict[int, float] = {}
    for ranked_list in ranks:
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.data: List[Dict] = []
        self._bm25: Optional[object] = None   # rebuilt on every add()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunks: List[Dict], doc_id: str):
        """
        chunks: list of dicts from chunker.chunk_text()
            Each must have at least {"text": str, "chunk_index": int}
        """
        self.index.add(embeddings)

        for chunk in chunks:
            self.data.append({
                "text":        chunk["text"],
                "doc_id":      doc_id,
                "chunk_index": chunk["chunk_index"],
                "char_start":  chunk.get("char_start", 0),
            })

        self._rebuild_bm25()

    def _rebuild_bm25(self):
        if not HAS_BM25 or not self.data:
            self._bm25 = None
            return
        corpus = [_tokenize(item["text"]) for item in self.data]
        self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, query_text: str = "", k: int = 10) -> List[Dict]:
        """
        Hybrid search: dense (FAISS) + sparse (BM25) fused with RRF.
        Falls back to dense-only if BM25 not available.
        """
        total = self.index.ntotal
        if total == 0:
            return []

        fetch_k = min(total, k * 4)

        # --- Dense ---
        distances, indices = self.index.search(query_embedding, fetch_k)
        dense_ranked = [int(i) for i in indices[0] if i < total]

        # --- Sparse (BM25) ---
        sparse_ranked: List[int] = []
        if HAS_BM25 and self._bm25 and query_text:
            tokens = _tokenize(query_text)
            bm25_scores = self._bm25.get_scores(tokens)
            sparse_ranked = list(np.argsort(bm25_scores)[::-1][:fetch_k])

        # --- Fuse ---
        if sparse_ranked:
            fused_scores = _rrf([dense_ranked, sparse_ranked])
        else:
            fused_scores = {idx: 1.0 / (r + 1) for r, idx in enumerate(dense_ranked)}

        sorted_indices = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)

        # --- Balance across docs ---
        seen_docs: Dict[str, int] = {}
        final: List[Dict] = []

        for idx in sorted_indices:
            item = self.data[idx]
            doc = item["doc_id"]
            seen_docs[doc] = seen_docs.get(doc, 0)

            if seen_docs[doc] < 3:          # max 3 chunks per doc per query
                final.append({**item, "score": fused_scores[idx]})
                seen_docs[doc] += 1

            if len(final) >= k:
                break

        return final

    def get_all_for_doc(self, doc_id: str) -> List[Dict]:
        return [item for item in self.data if item["doc_id"] == doc_id]

    def list_docs(self) -> List[str]:
        return list(dict.fromkeys(item["doc_id"] for item in self.data))

    def debug_count(self) -> int:
        return self.index.ntotal
