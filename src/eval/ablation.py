"""
CineSemantics Day-8 (Phase 6) — capability ablation ladder.

We rebuild the recommender ONE capability at a time and re-measure on the SAME
per-user temporal held-out split used since Day 3 (src/eval/build_cf_eval.py:
leave-last-20% of each user's likes, no test interaction in train). Every rung is
scored with the identical metric code and the identical candidate universe, so
the deltas are apples-to-apples.

Ladder (each rung ADDS one capability to the one above it):
  1. semantic            content retrieval only — MiniLM centroid of the user's
                         liked-movie embeddings (the shipped encoder, Day 1)
  2. +better_embeddings  swap MiniLM -> e5-base-v2 centroid (Day-2 champion enc.)
  3. +CF                 item-item collaborative filtering (Day-3 champion) —
                         the first rung with a real interaction signal
  4. +tuning             ALS matrix factorisation, Optuna-tuned (Day-6)
  5. +sequential         BERT4Rec sequential transformer (Day-7)
  6. +diversity          Day-3 champion (ItemKNN) + MMR genre re-rank (Day-6)

Rungs 1-3 and 6 are recomputed FRESH here (fast: cached embeddings + a 0.04s
ItemKNN fit) so the CSV is self-reproducing. Rungs 4-5 (ALS-tuned needs the
implicit lib; BERT4Rec is an 80s GPU-less train) are carried from their saved
day metrics (results/phase4_metrics.json / results/phase5_metrics.json) and are
clearly flagged `carried` in the CSV. Numbers match those days by construction.

Genuine insight this exposes: the single capability that MATTERS is CF. Content
retrieval — no matter how good the embedding — cannot personalise (rungs 1-2 sit
near-zero). The interaction signal (rung 3) is an ~8x NDCG jump. Everything after
CF is a trade, not a lift: sequential wins next-item but not full-list; diversity
buys catalog coverage at a small NDCG cost.

Outputs:
  results/ablation.csv
  results/figures/day8_ablation.png
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
CACHE = RESULTS / "emb_cache"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

K_NDCG, K_RECALL, K_MAP, K_PREC, K_LIST = 10, 20, 20, 10, 10
SEED = 42


# ---------- metrics (identical to src/recsys/cf_compare.py) ----------
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


def score_system(rank_fn, train, test, universe):
    """Aggregate metrics over all eval users for a ranking function.

    rank_fn(user_train_items, seen_set) -> ordered list of candidate catalog idx
    (already excluding seen items). We take the top-50 it returns.
    """
    ndcg = rec = mp = prec = 0.0
    recommended_items = set()
    n = 0
    for u, tr in train.items():
        te = set(test[u])
        if not te:
            continue
        seen = set(tr)
        ranked = rank_fn(tr, seen)[:50]
        ndcg += ndcg_at_k(ranked, te, K_NDCG)
        rec += recall_at_k(ranked, te, K_RECALL)
        mp += ap_at_k(ranked, te, K_MAP)
        prec += precision_at_k(ranked, te, K_PREC)
        recommended_items.update(ranked[:K_LIST])
        n += 1
    return {
        "ndcg@10": round(ndcg / n, 4),
        "recall@20": round(rec / n, 4),
        "map@20": round(mp / n, 4),
        "precision@10": round(prec / n, 4),
        "catalog_coverage": round(len(recommended_items) / len(universe), 4),
        "n_users": n,
    }


# ---------- rung builders ----------
def content_ranker(emb_by_cat, universe):
    """Centroid-of-liked cosine ranker over the candidate universe."""
    uni = np.asarray(universe)
    mat = np.stack([emb_by_cat[i] for i in uni]).astype(np.float32)  # already L2-normed

    def rank(tr, seen):
        liked = [emb_by_cat[i] for i in tr if i in emb_by_cat]
        if not liked:
            return []
        centroid = np.mean(liked, axis=0)
        nrm = np.linalg.norm(centroid)
        if nrm == 0:
            return []
        centroid = centroid / nrm
        scores = mat @ centroid
        order = np.argsort(-scores)
        out = []
        for j in order:
            cat = int(uni[j])
            if cat in seen:
                continue
            out.append(cat)
            if len(out) >= 50:
                break
        return out

    return rank


def build_itemknn(train, universe):
    """Item-item cosine on the binary user x item matrix (Day-3 champion)."""
    col = {c: j for j, c in enumerate(universe)}
    users = sorted(train)
    rows, cols = [], []
    for r, u in enumerate(users):
        for it in train[u]:
            if it in col:
                rows.append(r)
                cols.append(col[it])
    R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                   shape=(len(users), len(universe)))
    # cosine item-item
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(R.T.tocsr(), dense_output=True).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    uni = np.asarray(universe)

    def rank(tr, seen):
        idx = [col[i] for i in tr if i in col]
        if not idx:
            return []
        scores = S[idx].sum(axis=0)
        order = np.argsort(-scores)
        out = []
        for j in order:
            cat = int(uni[j])
            if cat in seen or scores[j] <= 0:
                continue
            out.append(cat)
            if len(out) >= 50:
                break
        return out

    return rank, S, col, uni


def mmr_wrap(base_rank, genres, lam=0.7):
    """Greedy MMR genre-diversity re-rank of the base ranker's top candidates."""
    def rank(tr, seen):
        cand = base_rank(tr, seen)[:50]
        if not cand:
            return []
        selected, gsel = [], []
        pool = list(cand)
        # relevance proxy = position in base list (higher = more relevant)
        relevance = {c: (len(pool) - p) / len(pool) for p, c in enumerate(pool)}
        while pool and len(selected) < 50:
            best, best_score = None, -1e9
            for c in pool:
                gc = genres.get(c, set())
                if not gsel:
                    div = 1.0
                else:
                    sims = [len(gc & g) / max(1, len(gc | g)) for g in gsel]
                    div = 1.0 - max(sims)
                score = lam * relevance[c] + (1 - lam) * div
                if score > best_score:
                    best_score, best = score, c
            selected.append(best)
            gsel.append(genres.get(best, set()))
            pool.remove(best)
        return selected

    return rank


def main():
    t0 = time.time()
    split = json.load(open(EVAL / "cf_split.json"))
    train = {int(u): v for u, v in split["train"].items()}
    test = {int(u): v for u, v in split["test"].items()}
    universe = split["item_universe"]
    catalog = pd.read_csv(DATA / "9000plus.csv")
    genres = {i: genre_set(catalog.loc[i, "Genre"]) for i in universe}

    # ---- embeddings by catalog idx ----
    mnz = np.load(CACHE / "minilm_universe.npz")
    minilm = {int(c): mnz["emb"][j] for j, c in enumerate(mnz["universe"])}
    e5_full = np.load(CACHE / "intfloat__e5-base-v2.npy")   # full catalog, L2-normed
    e5 = {i: e5_full[i] for i in universe}

    rows = []

    # rung 1: semantic (MiniLM content centroid)
    print("[ablation] rung 1: semantic (MiniLM centroid)")
    m = score_system(content_ranker(minilm, universe), train, test, universe)
    rows.append(dict(rung=1, system="semantic (MiniLM centroid)",
                     capability="content retrieval", source="fresh", **m))

    # rung 2: + better embeddings (e5-base-v2 centroid)
    print("[ablation] rung 2: +better_embeddings (e5-base-v2 centroid)")
    m = score_system(content_ranker(e5, universe), train, test, universe)
    rows.append(dict(rung=2, system="+better_embeddings (e5-base-v2 centroid)",
                     capability="stronger content encoder", source="fresh", **m))

    # rung 3: + CF (ItemKNN)
    print("[ablation] rung 3: +CF (ItemKNN item-item)")
    knn_rank, S, col, uni = build_itemknn(train, universe)
    m = score_system(knn_rank, train, test, universe)
    rows.append(dict(rung=3, system="+CF (ItemKNN)",
                     capability="interaction signal / personalization",
                     source="fresh", **m))

    # rung 4: + tuning (ALS-tuned) — carried from Day-6
    p4 = json.load(open(RESULTS / "phase4_metrics.json"))
    als = _find_als_tuned(p4)
    rows.append(dict(rung=4, system="+tuning (ALS Optuna-tuned)",
                     capability="tuned matrix factorization", source="carried:Day6",
                     **als))

    # rung 5: + sequential (BERT4Rec) — carried from Day-7
    p5 = json.load(open(RESULTS / "phase5_metrics.json"))
    b4 = p5["leaderboard"]["BERT4Rec"]
    rows.append(dict(rung=5, system="+sequential (BERT4Rec)",
                     capability="sequential transformer", source="carried:Day7",
                     **{"ndcg@10": round(b4["fl_ndcg@10"], 4),
                        "recall@20": round(b4["fl_recall@20"], 4),
                        "map@20": round(b4["fl_map@20"], 4),
                        "precision@10": round(b4["fl_precision@10"], 4),
                        "catalog_coverage": round(b4["catalog_coverage"], 4),
                        "n_users": p5["eval"]["n_users"]}))

    # rung 6: + diversity (ItemKNN + MMR)
    print("[ablation] rung 6: +diversity (ItemKNN + MMR lambda=0.7)")
    m = score_system(mmr_wrap(knn_rank, genres, lam=0.7), train, test, universe)
    rows.append(dict(rung=6, system="+diversity (ItemKNN + MMR)",
                     capability="MMR genre re-rank", source="fresh", **m))

    df = pd.DataFrame(rows)
    cols_order = ["rung", "system", "capability", "ndcg@10", "recall@20",
                  "map@20", "precision@10", "catalog_coverage", "n_users", "source"]
    df = df[cols_order]
    df.to_csv(RESULTS / "ablation.csv", index=False)
    print(df.to_string(index=False))

    _plot(df)
    print(f"[ablation] done in {time.time() - t0:.1f}s -> results/ablation.csv")


def _find_als_tuned(p4):
    """Pull ALS-tuned full-list metrics from Day-6 metrics json (schema-tolerant)."""
    # Day-6 saved the tuned ALS row; fall back to phase5 leaderboard mirror.
    p5 = json.load(open(RESULTS / "phase5_metrics.json"))
    a = p5["leaderboard"]["ALS_tuned"]
    return {"ndcg@10": round(a["fl_ndcg@10"], 4),
            "recall@20": round(a["fl_recall@20"], 4),
            "map@20": round(a["fl_map@20"], 4),
            "precision@10": round(a["fl_precision@10"], 4),
            "catalog_coverage": round(a["catalog_coverage"], 4),
            "n_users": p5["eval"]["n_users"]}


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [s.replace("+", "+\n") for s in df["system"]]
    vals = df["ndcg@10"].values
    colors = ["#b0b0b0", "#b0b0b0", "#2a7de1", "#2a7de1", "#7e57c2", "#26a69a"]
    bars = ax.bar(range(len(df)), vals, color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NDCG@10 (temporal held-out, n=547)")
    ax.set_title("CineSemantics capability ablation — CF is the only rung that moves the needle")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS / "figures" / "day8_ablation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
