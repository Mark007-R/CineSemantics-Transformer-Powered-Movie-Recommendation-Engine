"""
Production recommender — the personalization layer the name always claimed.

For four days "CineSemantics — Transformer-Powered Recommendation Engine" had no
recommender at all: it was semantic search + a substring filter. Day 3 measured
six CF systems on a per-user temporal held-out split; ItemKNN (item-item cosine
on the binary interaction matrix) won at NDCG@10 0.1059, +47% over popularity.

This wraps that champion for production. Two entry points:
  * recommend(user_id)        — warm personalization for a known user.
  * recommend_from_likes(...)  — cold-start from a handful of liked catalog items
                                 via a content centroid over the champion e5
                                 embeddings (Day-3/5 fallback, honest NDCG 0.029).

Every returned item is a real catalog index -> the recommendations are 100%
catalog-valid by construction (the Day-8 anti-hallucination baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"
MODELS = ROOT / "models"
NEG_INF = -1e9


class Recommender:
    """ItemKNN collaborative filter with a content cold-start fallback."""

    def __init__(self, train: dict, item_universe: list,
                 catalog_embeddings: np.ndarray | None = None):
        self.train = {int(u): list(v) for u, v in train.items()}
        self.universe = list(item_universe)                 # catalog indices
        self.col = {cat: j for j, cat in enumerate(self.universe)}
        self.users = sorted(self.train)
        self.urow = {u: i for i, u in enumerate(self.users)}
        self.n_users = len(self.users)
        self.n_items = len(self.universe)
        self.catalog_embeddings = catalog_embeddings

        rows, cols = [], []
        for u in self.users:
            for it in self.train[u]:
                if it in self.col:
                    rows.append(self.urow[u]); cols.append(self.col[it])
        self.R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                            shape=(self.n_users, self.n_items))
        self.train_pop = np.asarray(self.R.sum(axis=0)).ravel()
        # item-item cosine similarity (Day-3 ItemKNN champion)
        self.S = cosine_similarity(self.R.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(self.S, 0.0)

    # ---- construction ----
    @classmethod
    def from_split(cls, catalog_embeddings=None):
        with open(EVAL / "cf_split.json") as f:
            split = json.load(f)
        return cls(split["train"], split["item_universe"], catalog_embeddings)

    def has_user(self, user_id) -> bool:
        return int(user_id) in self.urow

    # ---- scoring ----
    def _user_scores(self, user_id: int) -> np.ndarray:
        row = self.R[self.urow[int(user_id)]]
        return np.asarray(row @ self.S).ravel()

    def _rank(self, scores: np.ndarray, seen_cols, k: int):
        scores = scores.copy()
        if seen_cols:
            scores[list(seen_cols)] = NEG_INF
        top_cols = np.argsort(-scores)[:k]
        return [self.universe[c] for c in top_cols], [float(scores[c]) for c in top_cols]

    def recommend(self, user_id, k: int = 10):
        """Warm personalized recommendations (catalog indices + scores)."""
        uid = int(user_id)
        if not self.has_user(uid):
            raise KeyError(f"unknown user_id {uid}")
        seen = {self.col[it] for it in self.train[uid] if it in self.col}
        return self._rank(self._user_scores(uid), seen, k)

    def recommend_from_likes(self, liked_catalog_indices, k: int = 10):
        """Cold-start: rank by ItemKNN if the liked items are in-universe,
        otherwise fall back to a content centroid over the e5 embeddings."""
        liked = [int(i) for i in liked_catalog_indices]
        in_univ = [self.col[i] for i in liked if i in self.col]
        seen_cols = set(in_univ)
        if in_univ:
            pseudo = np.zeros(self.n_items, np.float32)
            pseudo[in_univ] = 1.0
            scores = pseudo @ self.S
            return self._rank(scores, seen_cols, k)
        # pure cold-start -> content fallback
        if self.catalog_embeddings is None:
            raise ValueError("content fallback needs catalog_embeddings")
        E = self.catalog_embeddings
        centroid = E[liked].mean(axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm > 0:
            centroid = centroid / nrm
        sims = E @ centroid
        sims[liked] = NEG_INF
        # restrict to the in-universe candidate set for comparability
        univ_arr = np.array(self.universe)
        univ_scores = sims[univ_arr]
        top = np.argsort(-univ_scores)[:k]
        return [int(univ_arr[t]) for t in top], [float(univ_scores[t]) for t in top]

    def save(self, path: Path = None):
        """Persist the interaction matrix + universe (models/ is gitignored)."""
        path = path or MODELS / "cf_interactions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, data=self.R.data, indices=self.R.indices,
                 indptr=self.R.indptr, shape=np.array(self.R.shape),
                 universe=np.array(self.universe), users=np.array(self.users))
        meta = {"n_users": self.n_users, "n_items": self.n_items,
                "champion": "ItemKNN (item-item cosine)", "day3_ndcg@10": 0.1059}
        (path.parent / "cf_meta.json").write_text(json.dumps(meta, indent=2))
        return path
