"""
CineSemantics Day-3 (Phase 2b): collaborative-filtering bake-off.

The repo was always called a "recommendation engine" but had NO personalization
and NO recsys metrics (Day 1). Day 2 improved the content-retrieval embedding.
Day 3 finally adds the recommendation layer and measures it honestly on a
per-user TEMPORAL held-out split (src/eval/build_cf_eval.py).

Systems compared (all trained on TRAIN interactions only, ranked over the shared
train-item universe, seen items excluded):
  1. Popularity        non-personalized train-popularity (the baseline that beat
                       semantic search 7x on Day 1)
  2. ItemKNN           item-item cosine on the binary interaction matrix
  3. ALS               implicit-feedback matrix factorization (implicit lib)
  4. PureSVD           truncated SVD of the interaction matrix (Cremonesi 2010).
                       [scikit-surprise's SVD DLL is blocked by this machine's
                        Application-Control policy, so PureSVD via scipy.svds is
                        the SVD-family competitor -- a standard, stronger CF SVD.]
  5. Content           centroid of the user's liked catalog embeddings
                       (all-MiniLM-L6-v2, the shipped encoder) -- content-based CF
  6. Hybrid            per-user z-score fusion of ALS + Content

Metrics on the held-out likes: precision@10, recall@20, NDCG@10, MAP@20,
catalog coverage, intra-list diversity@10 (comparable to Day-1 leaderboard) and
novelty@10 (self-information of recommended items).

Outputs:
  results/phase2b_cf.csv            leaderboard
  results/phase2b_metrics.json      headline dict (append-friendly)
  results/samples/cf_reco_*.json    sample per-user recommendations
  results/figures/phase2b_cf_ndcg.png
"""
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "emb_cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_NDCG = 10
K_RECALL = 20
K_MAP = 20
K_PREC = 10
K_LIST = 10
SEED = 42
NEG_INF = -1e9

# ---------- metrics ----------
def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(ranked, relevant, k):
    rel = [1.0 if i in relevant else 0.0 for i in ranked[:k]]
    idcg = dcg([1.0] * min(len(relevant), k))
    return dcg(rel) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def ap_at_k(ranked, relevant, k):
    hits, s = 0, 0.0
    for i, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            s += hits / (i + 1)
    denom = min(len(relevant), k)
    return s / denom if denom else 0.0


def genre_set(g):
    return set(x.strip().lower() for x in re.split(r"[,/|]", str(g)) if x.strip())


def build_catalog_text(row):
    """Mirror utils/text_embedder.embed_csv / Day-1 baseline exactly."""
    title = str(row.get("Title", ""))
    overview = str(row.get("Overview", ""))
    genre = str(row.get("Genre", ""))
    rd = str(row.get("Release_Date", ""))
    year = rd[:4] if len(rd) >= 4 else ""
    parts = [title]
    if genre:
        parts.append(f"Genre: {genre}")
    if year:
        parts.append(f"Released: {year}")
    if overview:
        parts.append(overview)
    return ". ".join(parts).strip()


def get_content_embeddings(catalog, universe):
    """MiniLM embeddings for the candidate universe, cached by catalog index."""
    cache = CACHE / "minilm_universe.npz"
    if cache.exists():
        d = np.load(cache)
        if list(d["universe"]) == list(universe):
            print("[content] using cached MiniLM embeddings")
            return d["emb"].astype(np.float32)
    from sentence_transformers import SentenceTransformer
    print(f"[content] encoding {len(universe)} candidate movies with {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [build_catalog_text(catalog.loc[i]) for i in universe]
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    emb = emb.astype(np.float32)
    np.savez(cache, emb=emb, universe=np.array(universe))
    return emb


def rank_from_scores(scores, seen_cols):
    """Argsort a score row descending, dropping seen columns (already -inf)."""
    return np.argsort(-scores)


def main():
    with open(EVAL / "cf_split.json") as f:
        split = json.load(f)
    train = {int(u): v for u, v in split["train"].items()}
    test = {int(u): set(v) for u, v in split["test"].items()}
    universe = split["item_universe"]                 # catalog indices
    n_users = len(train)
    n_items = len(universe)
    col = {cat: j for j, cat in enumerate(universe)}   # catalog idx -> matrix col
    users = sorted(train)
    urow = {u: i for i, u in enumerate(users)}
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n_catalog = len(catalog)
    genres = {cat: genre_set(catalog.loc[cat, "Genre"]) for cat in universe}
    print(f"[data] {n_users} users x {n_items} candidate items")

    # ---- train interaction matrix (binary) ----
    rows, cols = [], []
    for u in users:
        for it in train[u]:
            rows.append(urow[u]); cols.append(col[it])
    R = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                   shape=(n_users, n_items))

    train_pop = np.asarray(R.sum(axis=0)).ravel()      # item popularity (train)
    pop_prob = train_pop / n_users
    self_info = -np.log2(np.clip(pop_prob, 1e-9, None))  # novelty weight per item

    seen_cols = {u: [col[it] for it in train[u]] for u in users}

    # ---- score matrices per system (n_users x n_items) ----
    scores = {}
    timing = {}

    # 1. Popularity (broadcast)
    t = time.time()
    scores["Popularity"] = np.tile(train_pop, (n_users, 1)).astype(np.float32)
    timing["Popularity"] = time.time() - t

    # 2. ItemKNN (item-item cosine)
    t = time.time()
    S = cosine_similarity(R.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    scores["ItemKNN"] = (R @ S).astype(np.float32)
    timing["ItemKNN"] = time.time() - t

    # 3. ALS (implicit feedback MF)
    t = time.time()
    from implicit.als import AlternatingLeastSquares
    alpha = 15.0
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20,
                                  random_state=SEED, use_gpu=False)
    als.fit((R * alpha).tocsr(), show_progress=False)
    scores["ALS"] = (als.user_factors @ als.item_factors.T).astype(np.float32)
    timing["ALS"] = time.time() - t

    # 4. PureSVD (truncated SVD reconstruction)
    t = time.time()
    k_svd = 64
    U, sig, Vt = svds(R.astype(np.float32), k=k_svd)
    scores["PureSVD"] = ((U * sig) @ Vt).astype(np.float32)
    timing["PureSVD"] = time.time() - t

    # 5. Content (centroid of liked embeddings)
    t = time.time()
    E = get_content_embeddings(catalog, universe)      # n_items x d, normalized
    profiles = np.zeros((n_users, E.shape[1]), dtype=np.float32)
    for u in users:
        v = E[[col[it] for it in train[u]]].mean(axis=0)
        nrm = np.linalg.norm(v)
        profiles[urow[u]] = v / nrm if nrm > 0 else v
    scores["Content"] = (profiles @ E.T).astype(np.float32)
    timing["Content"] = time.time() - t

    # 6. Hybrid (per-user z-score fusion of ALS + Content)
    t = time.time()
    def zrows(M):
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True) + 1e-9
        return (M - mu) / sd
    scores["Hybrid"] = (zrows(scores["ALS"]) + zrows(scores["Content"])).astype(np.float32)
    timing["Hybrid"] = time.time() - t

    # ---- evaluate ----
    order = ["Popularity", "ItemKNN", "ALS", "PureSVD", "Content", "Hybrid"]
    rows_out = []
    coverage = {s: set() for s in order}
    samples = {}
    per_user_ndcg = {s: [] for s in order}
    for s in order:
        M = scores[s].copy()
        acc = {"ndcg": [], "recall": [], "map": [], "prec": [], "ild": [],
               "nov": []}
        for u in users:
            row = M[urow[u]].copy()
            row[seen_cols[u]] = NEG_INF
            ranked_cols = np.argsort(-row)[:max(K_RECALL, K_LIST)]
            ranked_cat = [universe[c] for c in ranked_cols]
            rel = test[u]
            acc["ndcg"].append(ndcg_at_k(ranked_cat, rel, K_NDCG))
            acc["recall"].append(recall_at_k(ranked_cat, rel, K_RECALL))
            acc["map"].append(ap_at_k(ranked_cat, rel, K_MAP))
            acc["prec"].append(precision_at_k(ranked_cat, rel, K_PREC))
            topl_cols = ranked_cols[:K_LIST]
            topl = [universe[c] for c in topl_cols]
            coverage[s].update(topl)
            acc["nov"].append(float(np.mean(self_info[topl_cols])))
            ds = []
            for a in range(len(topl)):
                for b in range(a + 1, len(topl)):
                    ga, gb = genres[topl[a]], genres[topl[b]]
                    un = ga | gb
                    ds.append(1 - (len(ga & gb) / len(un) if un else 0.0))
            acc["ild"].append(float(np.mean(ds)) if ds else 0.0)
        per_user_ndcg[s] = acc["ndcg"]
        rows_out.append({
            "system": s,
            "ndcg@10": round(float(np.mean(acc["ndcg"])), 4),
            "recall@20": round(float(np.mean(acc["recall"])), 4),
            "map@20": round(float(np.mean(acc["map"])), 4),
            "precision@10": round(float(np.mean(acc["prec"])), 4),
            "catalog_coverage": round(len(coverage[s]) / n_catalog, 4),
            "candidate_coverage": round(len(coverage[s]) / n_items, 4),
            "intra_list_diversity@10": round(float(np.mean(acc["ild"])), 4),
            "novelty@10": round(float(np.mean(acc["nov"])), 3),
            "fit_seconds": round(timing[s], 2),
        })

        # samples (first 3 users) for the champion-ish systems
        if s in ("Popularity", "ALS", "Content", "Hybrid"):
            srows = []
            for u in users[:3]:
                row = M[urow[u]].copy()
                row[seen_cols[u]] = NEG_INF
                top = np.argsort(-row)[:10]
                srows.append({
                    "user": u,
                    "liked_train_sample": [catalog.loc[c, "Title"] for c in train[u][:5]],
                    "held_out_test": [catalog.loc[c, "Title"] for c in list(test[u])[:8]],
                    "top10": [{"title": catalog.loc[universe[c], "Title"],
                               "genre": catalog.loc[universe[c], "Genre"],
                               "hit": bool(universe[c] in test[u])} for c in top],
                })
            samples[s] = srows

    lb = pd.DataFrame(rows_out).sort_values("ndcg@10", ascending=False)
    lb.to_csv(RESULTS / "phase2b_cf.csv", index=False)
    print("\n" + lb.to_string(index=False))

    # paired significance: champion vs Popularity (Wilcoxon on per-user NDCG)
    champ = lb.iloc[0]["system"]
    sig = {}
    try:
        from scipy.stats import wilcoxon
        for s in order:
            if s == "Popularity":
                continue
            stat, p = wilcoxon(per_user_ndcg[s], per_user_ndcg["Popularity"])
            sig[s] = {"vs_popularity_ndcg_wilcoxon_p": float(p)}
    except Exception as e:
        sig = {"error": str(e)}

    headline = {
        "day": 3, "project": "CineSemantics", "phase": "Phase 2b - CF bake-off",
        "date": "2026-07-07",
        "eval": {
            "protocol": "per-user temporal leave-last-20% (no leakage)",
            "n_users": n_users, "candidate_items": n_items,
            "test_interactions": int(sum(len(v) for v in test.values())),
            "cold_test_ceiling_recall": 0.941,
        },
        "leaderboard": {r["system"]: {k: r[k] for k in r if k != "system"}
                        for r in rows_out},
        "champion": champ,
        "day1_reference": {"semantic_ndcg@10": 0.0295, "popularity_ndcg@10": 0.2148},
        "significance": sig,
    }
    with open(RESULTS / "phase2b_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)
    with open(RESULTS / "samples" / "cf_reco_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = lb.sort_values("ndcg@10")
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        colors = ["#3b7dd8" if s == champ else "#9bb7d4" for s in d["system"]]
        bars = ax.barh(d["system"], d["ndcg@10"], color=colors)
        ax.axvline(0.0295, color="#c0392b", ls="--", lw=1)
        ax.text(0.0295, -0.4, "Day-1 semantic 0.0295", color="#c0392b", fontsize=8)
        ax.set_xlabel("NDCG@10 (personalized, held-out users)")
        ax.set_title("CineSemantics Day-3: collaborative-filtering bake-off\n"
                     f"per-user temporal split, {n_users} users")
        for b, v in zip(bars, d["ndcg@10"]):
            ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "phase2b_cf_ndcg.png", dpi=130)
        print(f"[fig] saved {RESULTS/'figures'/'phase2b_cf_ndcg.png'}")
    except Exception as e:
        print(f"[fig] skipped: {e}")

    print(f"\n[done] champion = {champ}; wrote results/phase2b_cf.csv")


if __name__ == "__main__":
    main()
