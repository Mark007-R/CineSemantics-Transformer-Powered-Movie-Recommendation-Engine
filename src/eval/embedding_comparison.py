"""
CineSemantics Day-2 (Phase 2a) — embedding-model bake-off for content retrieval.

Builds directly on the Day-1 harness (src/eval/baseline.py): SAME catalog text
construction, SAME co-rating "more like this" relevance set, SAME IR/recsys
metrics. The only thing that changes is the embedding model. Every model is
scored item-item over the full 9.8K catalog against the 1,072 held-out queries,
so the numbers are directly comparable to the Day-1 baseline (MiniLM = 0.0295).

Models compared (all CPU, sentence-transformers):
  - sentence-transformers/all-MiniLM-L6-v2   384   (current / Day-1 baseline)
  - BAAI/bge-small-en-v1.5                    384   (fast small)
  - sentence-transformers/all-mpnet-base-v2   768
  - intfloat/e5-base-v2                       768
  - BAAI/bge-base-en-v1.5                     768

Each model's text is prefixed per the authors' convention for SYMMETRIC
similarity (E5: "query: " on both sides; BGE: no instruction for symmetric s2s;
MiniLM/MPNet: none) so no model is unfairly penalised.

Experiment 2 (metadata-aware retrieval) replaces the current substring genre
filter (utils/milvus_vectordb.py:354) with a proper metadata-aware re-ranker on
top of the champion embedding: score = cosine + a*genre_Jaccard + b*pop_prior.
The blend weights (a, b) are tuned on a DEV half of the queries and reported on a
disjoint TEST half, so the reported lift is not fit on its own test set.

Outputs:
  - results/phase2a_embeddings.csv         model bake-off table
  - results/phase2a_metadata_rerank.csv    substring-filter vs metadata-aware
  - results/phase2a_metrics.json           append-friendly headline dict
  - results/figures/phase2a_ndcg.png       bar chart across models
  - results/samples/phase2a_champion_more_like_this.json
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
CACHE = RESULTS / "emb_cache"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

K_NDCG = 10
K_RECALL = 20
K_MAP = 20
K_LIST = 10
SEED = 42

# name -> (hf_id, dim, symmetric-similarity prefix)
MODELS = [
    ("MiniLM-L6-v2 (current)", "sentence-transformers/all-MiniLM-L6-v2", 384, ""),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", 384, ""),
    ("mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", 768, ""),
    ("e5-base-v2", "intfloat/e5-base-v2", 768, "query: "),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 768, ""),
]


def build_catalog_text(row) -> str:
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


# ---- metrics (identical to Day-1 baseline.py) ----
def dcg(rel_flags):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel_flags))


def ndcg_at_k(ranked, relevant, k):
    rel_flags = [1.0 if i in relevant else 0.0 for i in ranked[:k]]
    idcg = dcg([1.0] * min(len(relevant), k))
    return dcg(rel_flags) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    return (len(set(ranked[:k]) & relevant) / len(relevant)) if relevant else 0.0


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


def encode_catalog(hf_id, prefix, texts):
    """Encode (cached to disk by model id) and return (emb, encode_seconds)."""
    safe = hf_id.replace("/", "__")
    cache_f = CACHE / f"{safe}.npy"
    meta_f = CACHE / f"{safe}.time.json"
    if cache_f.exists() and meta_f.exists():
        emb = np.load(cache_f)
        enc_s = json.loads(meta_f.read_text())["encode_seconds"]
        print(f"[cache] {hf_id}: {emb.shape} ({enc_s:.1f}s original)")
        return emb.astype(np.float32), enc_s

    from sentence_transformers import SentenceTransformer
    print(f"[model] loading {hf_id}")
    model = SentenceTransformer(hf_id)
    inp = [prefix + t for t in texts] if prefix else texts
    t0 = time.perf_counter()
    emb = model.encode(inp, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    enc_s = time.perf_counter() - t0
    emb = emb.astype(np.float32)
    np.save(cache_f, emb)
    meta_f.write_text(json.dumps({"encode_seconds": enc_s, "dim": int(emb.shape[1])}))
    print(f"[model] {hf_id} encoded {emb.shape} in {enc_s:.1f}s")
    return emb, enc_s


def eval_embedding(emb, queries, relevance, genres, n):
    """Item-item 'more like this' eval over the full catalog. Returns metric dict
    plus per-query semantic ranks (for the champion reranking experiment)."""
    acc = {"ndcg": [], "recall": [], "map": [], "prec": [], "ild": []}
    coverage = set()
    search_times = []
    ranks = {}
    for q in queries:
        rel = relevance[q]
        t0 = time.perf_counter()
        sims = emb @ emb[q]
        sims[q] = -np.inf
        order = np.argsort(-sims)
        search_times.append(time.perf_counter() - t0)
        r = [int(i) for i in order]
        ranks[q] = (r[:64], sims)  # keep head + sims for reranking reuse
        acc["ndcg"].append(ndcg_at_k(r, rel, K_NDCG))
        acc["recall"].append(recall_at_k(r, rel, K_RECALL))
        acc["map"].append(ap_at_k(r, rel, K_MAP))
        acc["prec"].append(precision_at_k(r, rel, K_NDCG))
        topl = r[:K_LIST]
        coverage.update(topl)
        ds = []
        for a in range(len(topl)):
            for b in range(a + 1, len(topl)):
                ga, gb = genres[topl[a]], genres[topl[b]]
                union = ga | gb
                ds.append(1 - ((len(ga & gb) / len(union)) if union else 0.0))
        acc["ild"].append(float(np.mean(ds)) if ds else 0.0)
    return {
        "ndcg@10": round(float(np.mean(acc["ndcg"])), 4),
        "recall@20": round(float(np.mean(acc["recall"])), 4),
        "map@20": round(float(np.mean(acc["map"])), 4),
        "precision@10": round(float(np.mean(acc["prec"])), 4),
        "catalog_coverage": round(len(coverage) / n, 4),
        "intra_list_diversity@10": round(float(np.mean(acc["ild"])), 4),
        "search_ms_per_query": round(float(np.mean(search_times)) * 1000, 3),
    }, ranks


def substring_genre_rank(q, emb, genres_str, n):
    """Reproduce the CURRENT substring genre filter as a HARD pre-filter, then
    rank the survivors by semantic similarity. Query genre = the movie's first
    listed genre (utils/milvus_vectordb.py:354 does `filter.lower() in genre.lower()`)."""
    qg = re.split(r"[,/|]", str(genres_str[q]))
    qg = qg[0].strip().lower() if qg and qg[0].strip() else ""
    sims = emb @ emb[q]
    sims[q] = -np.inf
    if qg:
        mask = np.array([qg in str(genres_str[i]).lower() for i in range(n)])
        sims = np.where(mask, sims, -np.inf)
    return [int(i) for i in np.argsort(-sims)]


def metadata_rerank(q, emb, genres, pop_norm, a, b, cand_pool=200):
    """Metadata-aware re-ranker: take the top `cand_pool` by cosine, then rescore
    score = cosine + a*genre_Jaccard(query,cand) + b*pop_prior. Soft, not a hard
    substring filter -> keeps recall while adding metadata signal."""
    sims = emb @ emb[q]
    sims[q] = -np.inf
    pool = np.argsort(-sims)[:cand_pool]
    gq = genres[q]
    rescored = []
    for i in pool:
        gi = genres[i]
        union = gq | gi
        jac = (len(gq & gi) / len(union)) if union else 0.0
        rescored.append((int(i), float(sims[i]) + a * jac + b * float(pop_norm[i])))
    rescored.sort(key=lambda x: x[1], reverse=True)
    ranked = [i for i, _ in rescored]
    # append the rest of the semantic order beyond the pool (for recall@20 fairness)
    tail = [int(i) for i in np.argsort(-sims) if int(i) not in set(ranked)]
    return ranked + tail


def main():
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n = len(catalog)
    genres = [genre_set(g) for g in catalog["Genre"]]
    genres_str = catalog["Genre"].astype(str).tolist()
    texts = [build_catalog_text(r) for _, r in catalog.iterrows()]

    with open(EVAL / "content_relevance.json") as f:
        relevance = {int(k): set(v) for k, v in json.load(f).items()}
    queries = sorted(relevance)
    print(f"[eval] {len(queries)} queries over {n} catalog movies\n")

    # popularity prior (min-max normalised vote_count), for metadata reranker
    vc = pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0).values.astype(float)
    pop_norm = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)

    # ---------- Experiment 1: embedding bake-off ----------
    rows = []
    champion = None  # (name, emb, ndcg)
    for name, hf_id, dim, prefix in MODELS:
        emb, enc_s = encode_catalog(hf_id, prefix, texts)
        metrics, ranks = eval_embedding(emb, queries, relevance, genres, n)
        row = {"model": name, "hf_id": hf_id, "dim": emb.shape[1],
               "encode_sec_9837docs": round(enc_s, 1),
               "encode_ms_per_doc": round(enc_s / n * 1000, 3), **metrics}
        rows.append(row)
        print(f"  -> {name}: NDCG@10={metrics['ndcg@10']} recall@20={metrics['recall@20']}")
        if champion is None or metrics["ndcg@10"] > champion[2]:
            champion = (name, emb, metrics["ndcg@10"], ranks, hf_id)

    lb = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    lb.to_csv(RESULTS / "phase2a_embeddings.csv", index=False)
    print("\n=== Phase 2a embedding bake-off (content retrieval) ===")
    print(lb.to_string(index=False))

    champ_name, champ_emb, champ_ndcg, _, champ_id = champion
    print(f"\n[champion] {champ_name} (NDCG@10={champ_ndcg})")

    # ---------- Experiment 2: metadata-aware retrieval vs substring filter ----------
    # dev/test 50-50 split of queries to tune (a,b) on dev, report on test
    rng = np.random.default_rng(SEED)
    qperm = rng.permutation(queries)
    dev = sorted(int(x) for x in qperm[: len(qperm) // 2])
    test = sorted(int(x) for x in qperm[len(qperm) // 2:])

    def score_on(qset, rank_fn):
        nd, rc = [], []
        for q in qset:
            r = rank_fn(q)
            nd.append(ndcg_at_k(r, relevance[q], K_NDCG))
            rc.append(recall_at_k(r, relevance[q], K_RECALL))
        return float(np.mean(nd)), float(np.mean(rc))

    # tune (a,b) on dev
    grid = [(a, b) for a in [0.0, 0.05, 0.1, 0.2, 0.4] for b in [0.0, 0.02, 0.05, 0.1]]
    best, best_nd = (0.0, 0.0), -1.0
    for a, b in grid:
        nd, _ = score_on(dev, lambda q, a=a, b=b:
                         metadata_rerank(q, champ_emb, genres, pop_norm, a, b))
        if nd > best_nd:
            best_nd, best = nd, (a, b)
    a_star, b_star = best
    print(f"[metadata] tuned on dev: a(genre)={a_star}, b(pop)={b_star} -> dev NDCG@10={best_nd:.4f}")

    def pure(q):
        sims = champ_emb @ champ_emb[q]
        sims[q] = -np.inf
        return [int(i) for i in np.argsort(-sims)]

    rerank_rows = []
    for label, fn in [
        ("substring_genre_filter (current)",
         lambda q: substring_genre_rank(q, champ_emb, genres_str, n)),
        ("champion_pure_semantic", pure),
        (f"metadata_aware_rerank (a={a_star},b={b_star})",
         lambda q: metadata_rerank(q, champ_emb, genres, pop_norm, a_star, b_star)),
    ]:
        nd, rc = score_on(test, fn)
        rerank_rows.append({"variant": label,
                            "ndcg@10_test": round(nd, 4),
                            "recall@20_test": round(rc, 4)})
        print(f"  [test] {label}: NDCG@10={nd:.4f} recall@20={rc:.4f}")

    rr = pd.DataFrame(rerank_rows)
    rr.to_csv(RESULTS / "phase2a_metadata_rerank.csv", index=False)

    # ---------- samples: champion 'more like this' ----------
    samples = []
    for q in queries[:8]:
        r = pure(q)
        rel = relevance[q]
        samples.append({
            "query_index": q, "query_title": catalog.loc[q, "Title"],
            "query_genre": catalog.loc[q, "Genre"], "n_relevant": len(rel),
            "champion_top10": [
                {"title": catalog.loc[i, "Title"], "genre": catalog.loc[i, "Genre"],
                 "is_relevant": bool(i in rel)} for i in r[:10]],
        })
    with open(RESULTS / "samples" / "phase2a_champion_more_like_this.json", "w") as f:
        json.dump(samples, f, indent=2)

    # ---------- headline json ----------
    headline = {
        "day": 2, "project": "CineSemantics", "phase": "Phase 2a - embedding bake-off",
        "date": "2026-07-06",
        "eval": {"n_queries": len(queries), "catalog_size": n,
                 "relevance": "MovieLens co-rating 'more like this' (Day-1 set)"},
        "models": {r["model"]: {k: r[k] for k in r if k not in ("model", "hf_id")}
                   for r in rows},
        "champion": {"model": champ_name, "hf_id": champ_id, "ndcg@10": champ_ndcg},
        "day1_baseline_minilm_ndcg@10": 0.0295,
        "popularity_ndcg@10": 0.2148,
        "metadata_rerank": {
            "tuned_weights": {"genre": a_star, "pop": b_star},
            "test_results": rerank_rows,
        },
    }
    with open(RESULTS / "phase2a_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = lb.sort_values("ndcg@10")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        colors = ["#3b7dd8" if m != "MiniLM-L6-v2 (current)" else "#888888"
                  for m in order["model"]]
        bars = ax.barh(order["model"], order["ndcg@10"], color=colors)
        ax.axvline(0.2148, ls="--", color="#f2a154", lw=1.5, label="Popularity 0.2148")
        ax.set_xlabel("NDCG@10 (content retrieval, 1,072 held-out queries)")
        ax.set_title("CineSemantics Day-2: embedding-model bake-off")
        for b, v in zip(bars, order["ndcg@10"]):
            ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                    va="center", fontsize=9)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "phase2a_ndcg.png", dpi=130)
        print(f"\n[fig] saved {RESULTS / 'figures' / 'phase2a_ndcg.png'}")
    except Exception as e:
        print(f"[fig] skipped: {e}")

    print("\n[done] wrote results/phase2a_embeddings.csv + metadata_rerank.csv + metrics.json")


if __name__ == "__main__":
    main()
