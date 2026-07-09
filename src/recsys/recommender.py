"""
CineSemantics — champion recommender (Day-5 Phase-3 integration).

This is the recommendation layer the project's name ("...Recommendation Engine")
always promised but never had: through Day 1 the app was semantic search + a
substring genre filter, with ZERO personalization and ZERO recsys metrics.

Day-3 CF bake-off (src/recsys/cf_compare.py) on a per-user TEMPORAL held-out split:

    system        NDCG@10   note
    ItemKNN       0.1059    <- champion, +47% over popularity
    ALS           0.0808
    PureSVD       0.0724
    Popularity    0.0721    non-personalized baseline
    Hybrid        0.0699
    Content       0.0122    MiniLM centroid (weakest)

ItemKNN (item-item cosine on the binary interaction matrix) wins, so it is the
production ranker. Because pure ItemKNN can only score items a user's history
connects to, we add a CONTENT cold-start fallback (centroid of the user's liked
catalog embeddings) for brand-new users / items with no interaction neighbours —
the genuine personalization + cold-start capability an LLM has no history for.

Interactions come from public MovieLens aligned to the TMDB catalog (Day-1
build_cf_eval); item indices are catalog row indices in data/9000plus.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = _ROOT / "models"
NEG_INF = -1e9


class ItemKNNRecommender:
    """Item-item collaborative filtering (Day-3 champion) with content cold-start.

    `item_universe` maps matrix columns -> catalog row indices, so recommendations
    are returned as catalog indices directly usable by the retrieval layer / UI.
    """

    def __init__(self):
        self.item_universe: list[int] = []      # col -> catalog idx
        self._col: dict[int, int] = {}          # catalog idx -> col
        self.R: csr_matrix | None = None        # users x items (binary)
        self.S: np.ndarray | None = None        # item-item cosine (cols x cols)
        self.item_pop: np.ndarray | None = None  # per-item train popularity
        self.emb: np.ndarray | None = None      # optional catalog embeddings (cold-start)

    # ------------------------------------------------------------------ fit
    def fit(self, train: dict, item_universe: list[int]) -> "ItemKNNRecommender":
        """train: {user_id: [catalog_idx, ...]} ; item_universe: [catalog_idx,...]"""
        self.item_universe = list(item_universe)
        self._col = {c: j for j, c in enumerate(self.item_universe)}
        users = sorted(train)
        n_items = len(self.item_universe)
        rows, cols = [], []
        for ui, u in enumerate(users):
            for it in train[u]:
                if it in self._col:
                    rows.append(ui)
                    cols.append(self._col[it])
        R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                       shape=(len(users), n_items))
        self.R = R
        S = cosine_similarity(R.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(S, 0.0)
        self.S = S
        self.item_pop = np.asarray(R.sum(axis=0)).ravel()
        return self

    def attach_embeddings(self, catalog_embeddings: np.ndarray):
        """Provide full-catalog embeddings so cold-start users get content recs."""
        self.emb = catalog_embeddings.astype(np.float32)
        return self

    # -------------------------------------------------------------- predict
    def recommend(self, liked_catalog_idx, top_k: int = 10,
                  exclude_seen: bool = True) -> list[dict]:
        """Rank items for a user given the catalog indices they liked."""
        liked = [int(i) for i in liked_catalog_idx]
        in_universe = [i for i in liked if i in self._col]
        method = "itemknn"
        if in_universe and self.S is not None:
            cols = [self._col[i] for i in in_universe]
            scores = self.S[cols].sum(axis=0)             # over item_universe
            ranked_cols = np.argsort(-scores)
            seen_cols = set(cols) if exclude_seen else set()
            recs = []
            for c in ranked_cols:
                if c in seen_cols:
                    continue
                recs.append((self.item_universe[int(c)], float(scores[c])))
                if len(recs) >= top_k:
                    break
        else:
            method = "content_coldstart"
            recs = self._content_recommend(liked, top_k)
        return [{"index": i, "score": round(s, 4), "method": method} for i, s in recs]

    def _content_recommend(self, liked, top_k):
        """Cold-start: centroid of liked embeddings over the whole catalog."""
        if self.emb is None or not liked:
            # last resort: most popular items in the training universe
            if self.item_pop is None:
                return []
            order = np.argsort(-self.item_pop)[:top_k]
            return [(self.item_universe[int(c)], float(self.item_pop[c])) for c in order]
        v = self.emb[liked].mean(axis=0)
        n = np.linalg.norm(v)
        v = v / n if n > 0 else v
        sims = self.emb @ v
        sims[liked] = NEG_INF
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]

    # ------------------------------------------------------------- persist
    def save(self, path: Path | str | None = None):
        d = Path(path) if path else _MODELS
        d.mkdir(parents=True, exist_ok=True)
        save_npz(d / "cf_interactions.npz", self.R)
        (d / "cf_meta.json").write_text(json.dumps({"item_universe": self.item_universe}))
        return d

    @classmethod
    def load(cls, path: Path | str | None = None,
             catalog_embeddings: np.ndarray | None = None) -> "ItemKNNRecommender":
        d = Path(path) if path else _MODELS
        meta = json.loads((d / "cf_meta.json").read_text())
        obj = cls()
        obj.item_universe = [int(x) for x in meta["item_universe"]]
        obj._col = {c: j for j, c in enumerate(obj.item_universe)}
        obj.R = load_npz(d / "cf_interactions.npz")
        S = cosine_similarity(obj.R.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(S, 0.0)
        obj.S = S
        obj.item_pop = np.asarray(obj.R.sum(axis=0)).ravel()
        if catalog_embeddings is not None:
            obj.attach_embeddings(catalog_embeddings)
        return obj
