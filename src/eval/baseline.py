"""
CineSemantics Day-1 baseline metrics on the CURRENT system.

The current "recommendation engine" is off-the-shelf all-MiniLM-L6-v2 semantic
search over the movie catalog (Milvus IVF_FLAT, metric=IP on L2-normalized
384-d vectors) with an optional genre SUBSTRING post-filter. There is ZERO
offline evaluation in the repo.

This harness reproduces that retrieval faithfully WITHOUT Milvus (same model,
same text construction as utils/text_embedder.embed_csv, IP == cosine on
normalized vectors == exact inner product == what IVF_FLAT approximates). It then
scores it against the co-rating "more like this" relevance set from build_eval.py
using standard IR/recsys metrics, alongside Random and Popularity baselines so the
numbers are interpretable.

Outputs:
  - results/baseline_metrics.json      headline metrics (append-friendly dict)
  - results/baseline_leaderboard.csv   Random / Popularity / Semantic table
  - results/samples/baseline_*.json    5-10 sample "more like this" outputs
  - results/figures/baseline_ndcg.png  bar chart
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_NDCG = 10
K_RECALL = 20
K_MAP = 20
K_LIST = 10          # list length for coverage / diversity
SEED = 42


def build_catalog_text(row) -> str:
    """Mirror utils/text_embedder.embed_csv exactly."""
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


# ---- metrics ----
def dcg(rel_flags):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel_flags))


def ndcg_at_k(ranked, relevant, k):
    rel_flags = [1.0 if i in relevant else 0.0 for i in ranked[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg(ideal)
    return dcg(rel_flags) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    hit = len(set(ranked[:k]) & relevant)
    return hit / len(relevant) if relevant else 0.0


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def ap_at_k(ranked, relevant, k):
    hits, score = 0, 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(relevant), k)
    return score / denom if denom else 0.0


def genre_set(g):
    return set(x.strip().lower() for x in re.split(r"[,/|]", str(g)) if x.strip())


def main():
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n = len(catalog)
    genres = [genre_set(g) for g in catalog["Genre"]]

    with open(EVAL / "content_relevance.json") as f:
        relevance = {int(k): set(v) for k, v in json.load(f).items()}
    queries = sorted(relevance)
    print(f"[eval] {len(queries)} queries over {n} catalog movies")

    # ---- embed catalog (current-system reproduction) ----
    from sentence_transformers import SentenceTransformer
    print(f"[model] loading {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [build_catalog_text(r) for _, r in catalog.iterrows()]
    print("[model] encoding catalog (normalize=True, IP==cosine)...")
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    emb = emb.astype(np.float32)

    # popularity score (non-personalized baseline)
    pop = (pd.to_numeric(catalog["Popularity"], errors="coerce").fillna(0).values *
           0 + pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0).values)
    pop_rank = np.argsort(-pop)  # most popular first

    rng = np.random.default_rng(SEED)

    systems = {}
    coverage_sets = {"Random": set(), "Popularity": set(), "Semantic": set()}
    metric_acc = {s: {"ndcg": [], "recall": [], "map": [], "prec": [],
                      "ild": []} for s in coverage_sets}
    samples = []

    for qi, q in enumerate(queries):
        rel = relevance[q]

        # --- Semantic (current system) ---
        sims = emb @ emb[q]
        sims[q] = -np.inf  # exclude self
        sem_rank = np.argsort(-sims)

        # --- Popularity (exclude self) ---
        pop_r = [int(i) for i in pop_rank if i != q]

        # --- Random ---
        rand_r = rng.permutation(n)
        rand_r = [int(i) for i in rand_r if i != q]

        ranked = {"Random": rand_r,
                  "Popularity": pop_r,
                  "Semantic": [int(i) for i in sem_rank]}

        for s, r in ranked.items():
            metric_acc[s]["ndcg"].append(ndcg_at_k(r, rel, K_NDCG))
            metric_acc[s]["recall"].append(recall_at_k(r, rel, K_RECALL))
            metric_acc[s]["map"].append(ap_at_k(r, rel, K_MAP))
            metric_acc[s]["prec"].append(precision_at_k(r, rel, K_NDCG))
            topl = r[:K_LIST]
            coverage_sets[s].update(topl)
            # intra-list diversity: mean pairwise genre Jaccard distance
            ds = []
            for a in range(len(topl)):
                for b in range(a + 1, len(topl)):
                    ga, gb = genres[topl[a]], genres[topl[b]]
                    union = ga | gb
                    jac = (len(ga & gb) / len(union)) if union else 0.0
                    ds.append(1 - jac)
            metric_acc[s]["ild"].append(float(np.mean(ds)) if ds else 0.0)

        if qi < 8:
            samples.append({
                "query_index": q,
                "query_title": catalog.loc[q, "Title"],
                "query_genre": catalog.loc[q, "Genre"],
                "n_relevant": len(rel),
                "semantic_top10": [
                    {"title": catalog.loc[i, "Title"],
                     "genre": catalog.loc[i, "Genre"],
                     "is_relevant": bool(i in rel)}
                    for i in ranked["Semantic"][:10]
                ],
            })

    rows = []
    for s in coverage_sets:
        m = metric_acc[s]
        rows.append({
            "system": s,
            "ndcg@10": round(float(np.mean(m["ndcg"])), 4),
            "recall@20": round(float(np.mean(m["recall"])), 4),
            "map@20": round(float(np.mean(m["map"])), 4),
            "precision@10": round(float(np.mean(m["prec"])), 4),
            "catalog_coverage": round(len(coverage_sets[s]) / n, 4),
            "intra_list_diversity@10": round(float(np.mean(m["ild"])), 4),
        })
    lb = pd.DataFrame(rows)
    lb.to_csv(RESULTS / "baseline_leaderboard.csv", index=False)
    print("\n" + lb.to_string(index=False))

    headline = {
        "day": 1, "project": "CineSemantics", "phase": "Phase 1 - baseline",
        "date": "2026-07-05",
        "eval": {
            "source": "MovieLens ml-latest-small co-rating 'more like this'",
            "n_queries": len(queries), "catalog_size": n,
            "relevance": "co-liked (rating>=4) movies, aligned to TMDB catalog",
        },
        "current_system": {
            "model": MODEL_NAME, "dim": 384, "metric": "IP==cosine",
            "note": "genre filter is a substring post-filter (line 354), not semantic",
        },
        "systems": {r["system"]: {k: r[k] for k in r if k != "system"} for r in rows},
        "primary": {
            "semantic_ndcg@10": rows[2]["ndcg@10"],
            "semantic_recall@20": rows[2]["recall@20"],
            "semantic_map@20": rows[2]["map@20"],
            "popularity_ndcg@10": rows[1]["ndcg@10"],
            "random_ndcg@10": rows[0]["ndcg@10"],
        },
    }
    with open(RESULTS / "baseline_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)
    with open(RESULTS / "samples" / "baseline_more_like_this.json", "w") as f:
        json.dump(samples, f, indent=2)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        names = [r["system"] for r in rows]
        vals = [r["ndcg@10"] for r in rows]
        bars = ax.bar(names, vals, color=["#c0c0c0", "#f2a154", "#3b7dd8"])
        ax.set_ylabel("NDCG@10")
        ax.set_title("CineSemantics Day-1 baseline: content-retrieval NDCG@10\n"
                     "(MovieLens co-rating 'more like this', 1072 held-out queries)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "baseline_ndcg.png", dpi=130)
        print(f"[fig] saved {RESULTS / 'figures' / 'baseline_ndcg.png'}")
    except Exception as e:
        print(f"[fig] skipped: {e}")

    print("\n[done] wrote results/baseline_metrics.json + leaderboard + samples")


if __name__ == "__main__":
    main()
