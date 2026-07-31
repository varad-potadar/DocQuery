"""
services/vector_store.py

Hybrid vector store:
  - FAISS (flat L2) for dense semantic search
  - BM25-style keyword scoring via rank_bm25
  - Results are combined with Reciprocal Rank Fusion (RRF)

Retrieval has two modes (see search() docstring for the full reasoning):
  - Small active corpus (a handful of short documents — this app's
    main use case): every chunk is returned, no ranking/cutoff at all,
    so nothing relevant can be left out of an answer.
  - Large active corpus (many documents loaded): falls back to ranked
    hybrid search with a per-document cap generous enough that a
    question focused on one document among many still gets enough of
    it to answer completely, not just enough to "represent" it.

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

from services.config import EMBEDDING_DIM

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
    def __init__(self, dim: int = None):
        self.dim = dim or EMBEDDING_DIM
        self.index = faiss.IndexFlatL2(self.dim)
        self.data: List[Dict] = []
        self._embeddings = np.zeros((0, self.dim), dtype="float32")  # mirrors self.data, row-for-row
        self._bm25: Optional[object] = None   # rebuilt on every add()/remove_doc()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunks: List[Dict], doc_id: str):
        """
        chunks: list of dicts from chunker.chunk_text() or chunk_sections()
            Each must have at least {"text": str, "chunk_index": int}.
            "page" and "heading" are carried through when present.
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self._embeddings = np.vstack([self._embeddings, embeddings])

        for chunk in chunks:
            self.data.append({
                "text":        chunk["text"],
                "doc_id":      doc_id,
                "chunk_index": chunk["chunk_index"],
                "char_start":  chunk.get("char_start", 0),
                "page":        chunk.get("page"),
                "heading":     chunk.get("heading"),
            })

        self._rebuild_bm25()

    def remove_doc(self, doc_id: str) -> bool:
        """
        Removes every chunk belonging to doc_id from the live index and
        rebuilds the (flat) FAISS index + BM25 from what remains.

        A full rebuild is the simplest correct way to remove from a flat
        FAISS index — no ID-mapping bookkeeping to get wrong — and is
        cheap at the document counts this app is meant for. Returns True
        if anything was actually removed.
        """
        keep = [i for i, item in enumerate(self.data) if item["doc_id"] != doc_id]
        if len(keep) == len(self.data):
            return False

        self.data = [self.data[i] for i in keep]
        self._embeddings = self._embeddings[keep] if keep else np.zeros((0, self.dim), dtype="float32")

        self.index = faiss.IndexFlatL2(self.dim)
        if len(self._embeddings):
            self.index.add(self._embeddings)

        self._rebuild_bm25()
        return True

    def _rebuild_bm25(self):
        if not HAS_BM25 or not self.data:
            self._bm25 = None
            return
        corpus = [_tokenize(item["text"]) for item in self.data]
        self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        k: int = 10,
        small_corpus_char_budget: int = 20000,
    ) -> List[Dict]:
        """
        Retrieval, in two modes:

        1. Small corpus (the common case this app is built for — a
           handful of short documents): skip ranked retrieval entirely
           and return every chunk currently indexed. No similarity-search
           or per-document cap can guarantee it won't rank a genuinely
           relevant chunk just outside the cutoff — the only way to
           *guarantee* nothing gets left out of the answer is to not
           leave anything out of the context. Five 5-7 page documents is
           still only a few thousand tokens, well within what the model
           handles comfortably in one call.

        2. Large corpus (many documents active, past the budget above):
           hybrid dense (FAISS) + sparse (BM25) search fused with RRF,
           same as before — but with a much higher per-document cap than
           this used to have. A cap of 3 meant a 45-chunk document could
           only ever contribute 3 chunks to ANY answer, regardless of
           how much of it was actually relevant; a focused question
           about one specific document among many would silently lose
           over 90% of that document's content. The cap here still
           exists (so one huge document can't crowd out everything
           else when many documents are loaded), but it scales with k
           instead of being an arbitrary small constant.
        """
        total = self.index.ntotal
        if total == 0:
            return []

        total_chars = sum(len(item["text"]) for item in self.data)
        if total_chars <= small_corpus_char_budget:
            # Still return every chunk (nothing is silently excluded from
            # *retrieval*), but order by actual relevance to the query
            # instead of doc/chunk order. This matters because the caller
            # (qa_engine) may need to truncate the tail to stay under the
            # LLM provider's token-per-minute limit -- if that happens, we
            # want the least-relevant chunks cut, not just "whatever came
            # last alphabetically".
            distances, indices = self.index.search(query_embedding, total)
            dense_order = [int(i) for i in indices[0] if 0 <= i < total]
            missing = [i for i in range(total) if i not in set(dense_order)]
            ordered = dense_order + missing
            return [
                {**self.data[i], "score": 1.0 / (rank + 1)}
                for rank, i in enumerate(ordered)
            ]

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

        # --- Balance across docs (generous cap — see docstring above) ---
        per_doc_cap = max(8, k)
        seen_docs: Dict[str, int] = {}
        final: List[Dict] = []

        for idx in sorted_indices:
            item = self.data[idx]
            doc = item["doc_id"]
            seen_docs[doc] = seen_docs.get(doc, 0)

            if seen_docs[doc] < per_doc_cap:
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
