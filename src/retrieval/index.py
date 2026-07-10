"""
Production ANN index — the Day-4 champion (faiss HNSW, inner product).

Day 4 swept IVF_FLAT vs HNSW vs exact Flat on the same held-out eval. HNSW
Pareto-dominated the shipped IVF_FLAT: exact-quality NDCG@10 at ~4x lower p95
latency. This wraps a faiss HNSW index (with an exact Flat fallback for the
reproduction check) over the champion e5 catalog embeddings.

pymilvus does not import in this environment (a protobuf runtime clash), and
Milvus itself is a Docker service used by the Streamlit app. The offline faiss
index here serves the FastAPI path identically without requiring the container,
so the API is runnable in an interview without `docker compose up`.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise ImportError("faiss-cpu is required for the production index") from e

# Day-4 champion ANN parameters.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


class MovieIndex:
    """Wraps catalog embeddings with both an exact and an HNSW search path.

    kind="hnsw" (default) is the shipped champion; kind="flat" is the exact index
    used by the validation harness to prove the HNSW approximation costs ~nothing
    on the end task.
    """

    def __init__(self, embeddings: np.ndarray, kind: str = "hnsw",
                 ef_search: int = HNSW_EF_SEARCH):
        self.embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.n, self.dim = self.embeddings.shape
        self.kind = kind
        if kind == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, HNSW_M,
                                             faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch = ef_search
            self.index.add(self.embeddings)
        else:
            raise ValueError(f"unknown index kind: {kind!r}")

    def search(self, query_vec: np.ndarray, k: int, exclude=None):
        """Return (indices, scores) for the top-k most similar catalog items.

        `exclude` is an optional set/collection of catalog indices to drop from
        the results (e.g. the query item itself, or a user's already-seen items).
        A small over-fetch keeps k results after exclusion.
        """
        q = np.ascontiguousarray(query_vec.astype(np.float32).reshape(1, -1))
        exclude = set(exclude) if exclude else set()
        fetch = k + len(exclude) + 8
        scores, idx = self.index.search(q, min(fetch, self.n))
        out_idx, out_scores = [], []
        for i, s in zip(idx[0], scores[0]):
            if i < 0 or i in exclude:
                continue
            out_idx.append(int(i))
            out_scores.append(float(s))
            if len(out_idx) >= k:
                break
        return out_idx, out_scores

    def more_like_this(self, item_index: int, k: int):
        """'More like this' — use the item's own embedding as the query."""
        return self.search(self.embeddings[item_index], k, exclude={item_index})
