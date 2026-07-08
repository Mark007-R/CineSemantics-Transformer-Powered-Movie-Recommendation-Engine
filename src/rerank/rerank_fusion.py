"""
CineSemantics Day-4 (Phase 2c) — reranking + multimodal fusion + ANN index sweep.

Builds directly on the Day-1/Day-2 harness (src/eval/): SAME 9.8K catalog, SAME
1,072 held-out co-rating "more like this" queries, SAME IR/recsys metrics. The
retrieval backbone is the Day-2 champion embedding (intfloat/e5-base-v2, 768-dim).

Three experiments, three honest questions:

  A. ANN INDEX SWEEP (faiss). The production store is Milvus, whose IVF_FLAT and
     HNSW indexes are the SAME algorithms faiss exposes as IndexIVFFlat /
     IndexHNSWFlat. We characterise the recall-vs-latency-vs-memory frontier
     Docker-free and reproducibly here, then Day-7 carries the winning index_type
     into the Milvus collection. Every ANN config is scored on recall@10 vs an
     exact brute-force ground truth AND on the END-TASK NDCG@10, so we know how
     much approximation the recommender can tolerate before quality drops.

  B. CROSS-ENCODER RERANK. Take the top-K semantic candidates and rescore them
     with a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2). Honest caveat:
     that model is trained for QUERY->PASSAGE search relevance, not symmetric
     movie<->movie similarity, so this is a real test of whether an off-the-shelf
     reranker helps or hurts our task — reported as measured, with its latency cost.

  C. MULTIMODAL FUSION. Fuse the text signal with the CLIP poster signal
     (openai/clip-vit-base-patch32): score = a*text_cos + (1-a)*poster_cos, a
     tuned on a dev half and reported on a disjoint test half. Posters are pulled
     on demand from public TMDB thumbnails (the local posters/ mirror is absent on
     this machine) for a bounded, seeded query sample; text-only and fused are
     scored on the IDENTICAL candidate pool so the delta is the pure fusion effect.

Outputs:
  results/phase2c_rerank_fusion.csv   rerank variants, full 1,072-query eval
  results/phase2c_ann_sweep.csv       Flat vs IVF_FLAT vs HNSW frontier
  results/phase2c_fusion.csv          text-only vs poster-fused (bounded sample)
  results/phase2c_metrics.json        append-friendly headline dict
  results/figures/phase2c_*.png       charts
  results/samples/phase2c_rerank_examples.json
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
# reuse the EXACT Day-2 catalog text + metric functions so numbers stay comparable
from embedding_comparison import (  # noqa: E402
    build_catalog_text, ndcg_at_k, recall_at_k, precision_at_k, ap_at_k,
    genre_set, encode_catalog,
)

DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
CACHE = RESULTS / "emb_cache"
SCRATCH = Path(r"C:\Users\antho\AppData\Local\Temp\claude\C--Users-antho-OneDrive-Desktop-Mark-ALL-DATA-SCIENTIST\95ec0902-e425-414a-9f20-10c5bf2e37a8\scratchpad")
for d in [RESULTS / "samples", RESULTS / "figures", CACHE, SCRATCH]:
    d.mkdir(parents=True, exist_ok=True)

# Day-2 champion embedding = retrieval backbone for the whole day
CHAMPION_NAME = "e5-base-v2"
CHAMPION_ID = "intfloat/e5-base-v2"
CHAMPION_PREFIX = "query: "

K_NDCG, K_RECALL = 10, 20
SEED = 42


# ------------------------------------------------------------------ helpers
def load_champion(catalog):
    texts = [build_catalog_text(r) for _, r in catalog.iterrows()]
    emb, enc_s = encode_catalog(CHAMPION_ID, CHAMPION_PREFIX, texts)
    # ensure L2-normalised (encode_catalog normalises, but re-normalise defensively)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.clip(norms, 1e-9, None)).astype(np.float32)
    print(f"[champion] {CHAMPION_NAME} {emb.shape} (encode {enc_s:.1f}s)")
    return np.ascontiguousarray(emb)


def eval_ranking(rank_fn, queries, relevance):
    nd, rc, mp, pr = [], [], [], []
    for q in queries:
        r = rank_fn(q)
        rel = relevance[q]
        nd.append(ndcg_at_k(r, rel, K_NDCG))
        rc.append(recall_at_k(r, rel, K_RECALL))
        mp.append(ap_at_k(r, rel, K_RECALL))
        pr.append(precision_at_k(r, rel, K_NDCG))
    return {
        "ndcg@10": round(float(np.mean(nd)), 4),
        "recall@20": round(float(np.mean(rc)), 4),
        "map@20": round(float(np.mean(mp)), 4),
        "precision@10": round(float(np.mean(pr)), 4),
    }


# ============================================================ EXPERIMENT A
def experiment_ann(emb, queries, relevance):
    """faiss ANN sweep: Flat (exact) vs IVF_FLAT vs HNSW. Recall@10 vs exact,
    single-query p95 latency, on-disk index size, and END-TASK NDCG@10."""
    import faiss
    n, d = emb.shape
    print(f"\n=== Experiment A: ANN index sweep (n={n}, d={d}) ===")

    # ---- exact ground truth (brute-force inner product) ----
    flat = faiss.IndexFlatIP(d)
    flat.add(emb)
    KQ = 11  # top-10 + self
    gt_top10 = {}
    lat_flat = []
    for q in queries:
        t0 = time.perf_counter()
        _, I = flat.search(emb[q:q + 1], KQ)
        lat_flat.append(time.perf_counter() - t0)
        ids = [int(i) for i in I[0] if int(i) != q][:10]
        gt_top10[q] = ids

    def index_size_mb(index):
        p = SCRATCH / "tmp_faiss.index"
        faiss.write_index(index, str(p))
        mb = p.stat().st_size / 1e6
        p.unlink(missing_ok=True)
        return round(mb, 2)

    def bench(index, label, search_setup=None):
        """Return metrics dict for a built index."""
        lat = []
        recalls, ndcgs, recall20s = [], [], []
        for q in queries:
            t0 = time.perf_counter()
            _, I = index.search(emb[q:q + 1], KQ)
            lat.append(time.perf_counter() - t0)
            ids = [int(i) for i in I[0] if int(i) != q]
            top10 = ids[:10]
            recalls.append(len(set(top10) & set(gt_top10[q])) / 10.0)
            # end-task quality on the approx ranking
            ndcgs.append(ndcg_at_k(ids, relevance[q], K_NDCG))
            recall20s.append(recall_at_k(ids, relevance[q], K_RECALL))
        lat_ms = np.array(lat) * 1000
        return {
            "index": label,
            "recall@10_vs_exact": round(float(np.mean(recalls)), 4),
            "ndcg@10": round(float(np.mean(ndcgs)), 4),
            "recall@20": round(float(np.mean(recall20s)), 4),
            "lat_mean_ms": round(float(lat_ms.mean()), 4),
            "lat_p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
            "index_mb": index_size_mb(index),
        }

    rows = []
    # Flat exact reference (recall=1 by definition)
    lat_flat_ms = np.array(lat_flat) * 1000
    exact_ndcg = float(np.mean([ndcg_at_k(gt_top10[q] + [], relevance[q], K_NDCG) for q in queries]))
    # recompute exact full-order ndcg/recall properly with full ranking
    flat_full = []
    r20 = []
    for q in queries:
        _, I = flat.search(emb[q:q + 1], 64)
        ids = [int(i) for i in I[0] if int(i) != q]
        flat_full.append(ndcg_at_k(ids, relevance[q], K_NDCG))
        r20.append(recall_at_k(ids, relevance[q], K_RECALL))
    rows.append({
        "index": "Flat (exact / Milvus FLAT)",
        "recall@10_vs_exact": 1.0,
        "ndcg@10": round(float(np.mean(flat_full)), 4),
        "recall@20": round(float(np.mean(r20)), 4),
        "lat_mean_ms": round(float(lat_flat_ms.mean()), 4),
        "lat_p95_ms": round(float(np.percentile(lat_flat_ms, 95)), 4),
        "index_mb": index_size_mb(flat),
    })
    print(f"  Flat exact: NDCG@10={rows[0]['ndcg@10']} p95={rows[0]['lat_p95_ms']}ms {rows[0]['index_mb']}MB")

    # ---- IVF_FLAT sweep ----
    nlist = 128
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(emb)
    ivf.add(emb)
    for nprobe in [1, 4, 8, 16, 32, 64]:
        ivf.nprobe = nprobe
        m = bench(ivf, f"IVF_FLAT nlist={nlist} nprobe={nprobe}")
        rows.append(m)
        print(f"  {m['index']}: recall@10={m['recall@10_vs_exact']} NDCG@10={m['ndcg@10']} p95={m['lat_p95_ms']}ms")

    # ---- HNSW sweep ----
    M = 32
    hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(emb)
    for ef in [16, 32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef
        m = bench(hnsw, f"HNSW M={M} efSearch={ef}")
        rows.append(m)
        print(f"  {m['index']}: recall@10={m['recall@10_vs_exact']} NDCG@10={m['ndcg@10']} p95={m['lat_p95_ms']}ms")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase2c_ann_sweep.csv", index=False)
    return df


# ============================================================ EXPERIMENT B
def experiment_rerank(emb, catalog, queries, relevance, genres, pop_norm):
    """Cross-encoder rerank of the top-K semantic candidates, vs pure semantic and
    the Day-2 metadata reranker. Full 1,072-query eval + p95 latency per variant."""
    print("\n=== Experiment B: cross-encoder rerank ===")
    n = len(catalog)
    texts = [build_catalog_text(r) for _, r in catalog.iterrows()]

    def semantic_order(q, k=64):
        sims = emb @ emb[q]
        sims[q] = -np.inf
        return [int(i) for i in np.argsort(-sims)[:k]], sims

    # pure semantic + latency
    def timed(rank_fn, qs):
        lat = []
        out = {}
        for q in qs:
            t0 = time.perf_counter()
            out[q] = rank_fn(q)
            lat.append(time.perf_counter() - t0)
        lat_ms = np.array(lat) * 1000
        return out, round(float(np.percentile(lat_ms, 95)), 3)

    # --- pure semantic ---
    def pure(q):
        r, _ = semantic_order(q, 64)
        return r
    pure_ranks, pure_p95 = timed(pure, queries)
    m_pure = eval_ranking(lambda q: pure_ranks[q], queries, relevance)

    # --- Day-2 metadata reranker (carried, tuned a=0.2/b=0.05 region; re-tune small) ---
    def metadata_rank(q, a=0.2, b=0.05, pool=200):
        sims = emb @ emb[q]
        sims[q] = -np.inf
        cand = np.argsort(-sims)[:pool]
        gq = genres[q]
        rescored = []
        for i in cand:
            gi = genres[i]
            u = gq | gi
            jac = (len(gq & gi) / len(u)) if u else 0.0
            rescored.append((int(i), float(sims[i]) + a * jac + b * float(pop_norm[i])))
        rescored.sort(key=lambda x: x[1], reverse=True)
        ranked = [i for i, _ in rescored]
        tail = [int(i) for i in np.argsort(-sims) if int(i) not in set(ranked)]
        return ranked + tail
    meta_ranks, meta_p95 = timed(lambda q: metadata_rank(q), queries)
    m_meta = eval_ranking(lambda q: meta_ranks[q], queries, relevance)

    # --- cross-encoder ---
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    rows = [
        {"variant": f"champion_semantic ({CHAMPION_NAME})", **m_pure,
         "rerank_p95_ms": pure_p95, "notes": "Day-2 champion, pure cosine"},
        {"variant": "metadata_rerank (Day-2 carried)", **m_meta,
         "rerank_p95_ms": meta_p95, "notes": "cosine + genre-Jaccard + pop prior"},
    ]

    for pool_k in [15, 50]:
        # build all (query_text, cand_text) pairs for the top-pool_k semantic candidates
        pools = {q: pure_ranks[q][:pool_k] for q in queries}
        pairs, index_map = [], []
        for q in queries:
            for c in pools[q]:
                pairs.append([texts[q], texts[c]])
                index_map.append((q, c))
        t0 = time.perf_counter()
        scores = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        ce_sec = time.perf_counter() - t0
        per_pair_ms = ce_sec / len(pairs) * 1000
        # reassemble reranked order per query, keep semantic tail for recall fairness
        by_q = {q: [] for q in queries}
        for (q, c), s in zip(index_map, scores):
            by_q[q].append((c, float(s)))
        def ce_rank(q):
            ranked = [c for c, _ in sorted(by_q[q], key=lambda x: x[1], reverse=True)]
            pool_set = set(ranked)
            tail = [c for c in pure_ranks[q] if c not in pool_set]
            return ranked + tail
        m_ce = eval_ranking(ce_rank, queries, relevance)
        # end-to-end p95 = semantic search p95 + cross-encoder cost for pool_k pairs
        ce_p95 = round(pure_p95 + per_pair_ms * pool_k, 3)
        rows.append({
            "variant": f"cross_encoder_rerank top-{pool_k}", **m_ce,
            "rerank_p95_ms": ce_p95,
            "notes": f"ms-marco-MiniLM-L-6; {per_pair_ms:.2f}ms/pair, {len(pairs)} pairs"})
        print(f"  cross-encoder top-{pool_k}: NDCG@10={m_ce['ndcg@10']} "
              f"(pure {m_pure['ndcg@10']}) +{per_pair_ms:.2f}ms/pair")
        if pool_k == 15:
            ce_rank15 = ce_rank  # keep for samples

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase2c_rerank_fusion.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df, m_pure, pure_ranks, ce_rank15


# ============================================================ EXPERIMENT C
def _thumb_url(poster_url):
    # convert /t/p/original/<file> -> /t/p/w200/<file> for a small fast thumbnail
    if not isinstance(poster_url, str) or "/t/p/" not in poster_url:
        return None
    return poster_url.replace("/original/", "/w200/").replace("/w500/", "/w200/")


def _download_one(args):
    idx, url = args
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=15).read()
        return idx, data
    except Exception:
        return idx, None


def experiment_fusion(emb, catalog, queries, relevance, genres):
    """Multimodal fusion: text_cos vs a*text_cos + (1-a)*poster_cos on a bounded,
    seeded query sample. Posters pulled from public TMDB thumbnails on demand."""
    print("\n=== Experiment C: multimodal (text+poster) fusion ===")
    rng = np.random.default_rng(SEED)
    N_QUERIES = 300
    TOP_CAND = 25
    sample_q = sorted(int(x) for x in rng.choice(queries, size=min(N_QUERIES, len(queries)), replace=False))

    # candidate pools (text) for the sampled queries
    pools = {}
    need = set()
    for q in sample_q:
        sims = emb @ emb[q]
        sims[q] = -np.inf
        cand = [int(i) for i in np.argsort(-sims)[:TOP_CAND]]
        pools[q] = cand
        need.add(q)
        need.update(cand)
    need = sorted(need)
    print(f"  sampled {len(sample_q)} queries, need {len(need)} unique posters")

    # ---- CLIP poster embeddings (download + encode, cached) ----
    clip_cache = CACHE / "clip_posters_w200.npz"
    have = {}
    if clip_cache.exists():
        z = np.load(clip_cache)
        have = {int(k): z[k] for k in z.files}
        print(f"  [cache] {len(have)} poster embeddings loaded")
    todo = [i for i in need if i not in have]

    if todo:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        # download thumbnails in parallel
        dl_args = []
        for i in todo:
            u = _thumb_url(catalog.loc[i, "Poster_Url"])
            if u:
                dl_args.append((i, u))
        print(f"  downloading {len(dl_args)} posters (threaded)...")
        imgs = {}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=24) as ex:
            for idx, data in ex.map(_download_one, dl_args):
                if data is not None:
                    try:
                        imgs[idx] = Image.open(BytesIO(data)).convert("RGB")
                    except Exception:
                        pass
        print(f"  downloaded {len(imgs)}/{len(dl_args)} in {time.perf_counter()-t0:.1f}s; encoding CLIP...")

        # encode in batches
        idxs = list(imgs.keys())
        BS = 32
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(idxs), BS):
                chunk = idxs[s:s + BS]
                batch = [imgs[i] for i in chunk]
                inp = proc(images=batch, return_tensors="pt")
                feats = model.get_image_features(**inp)
                feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float32)
                for i, f in zip(chunk, feats):
                    have[i] = f
                if s % (BS * 20) == 0:
                    print(f"    encoded {s+len(chunk)}/{len(idxs)}")
        print(f"  CLIP encode done in {time.perf_counter()-t0:.1f}s")
        np.savez_compressed(clip_cache, **{str(k): v for k, v in have.items()})

    # ---- fuse on queries with posters + >=8 candidate posters ----
    poster_ok = set(have)
    eval_q = [q for q in sample_q if q in poster_ok and
              sum(c in poster_ok for c in pools[q]) >= 8]
    print(f"  {len(eval_q)} queries usable for fusion (poster coverage)")

    def restricted_text_rank(q):
        cand = [c for c in pools[q] if c in poster_ok]
        tsim = {c: float(emb[c] @ emb[q]) for c in cand}
        return sorted(cand, key=lambda c: tsim[c], reverse=True), tsim

    def fused_rank(q, a):
        cand = [c for c in pools[q] if c in poster_ok]
        pv = have[q]
        tsim = {c: float(emb[c] @ emb[q]) for c in cand}
        psim = {c: float(have[c] @ pv) for c in cand}
        return sorted(cand, key=lambda c: a * tsim[c] + (1 - a) * psim[c], reverse=True)

    # tune a on dev half, report test half
    perm = np.random.default_rng(SEED + 1).permutation(eval_q)
    dev = sorted(int(x) for x in perm[: len(perm) // 2])
    test = sorted(int(x) for x in perm[len(perm) // 2:])

    def ndcg_on(qs, rank_fn):
        return float(np.mean([ndcg_at_k(rank_fn(q), relevance[q], K_NDCG) for q in qs]))

    grid = [round(a, 2) for a in np.arange(0.5, 1.0001, 0.05)]
    best_a, best = 1.0, -1
    for a in grid:
        s = ndcg_on(dev, lambda q, a=a: fused_rank(q, a))
        if s > best:
            best, best_a = s, a
    text_only_test = ndcg_on(test, lambda q: restricted_text_rank(q)[0])
    fused_test = ndcg_on(test, lambda q: fused_rank(q, best_a))

    # intra-list diversity (genre) of top-5 for both, on test
    def ild(qs, rank_fn):
        out = []
        for q in qs:
            top = rank_fn(q)[:5]
            ds = []
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    ga, gb = genres[top[i]], genres[top[j]]
                    u = ga | gb
                    ds.append(1 - ((len(ga & gb) / len(u)) if u else 0.0))
            if ds:
                out.append(float(np.mean(ds)))
        return round(float(np.mean(out)), 4) if out else 0.0

    rows = [
        {"variant": "text_only (champion cosine)", "ndcg@10_test": round(text_only_test, 4),
         "ild@5_test": ild(test, lambda q: restricted_text_rank(q)[0]),
         "tuned_alpha_text": 1.0, "n_test_queries": len(test),
         "notes": "candidate pool restricted to items with posters"},
        {"variant": f"text+poster fusion (a={best_a})", "ndcg@10_test": round(fused_test, 4),
         "ild@5_test": ild(test, lambda q: fused_rank(q, best_a)),
         "tuned_alpha_text": best_a, "n_test_queries": len(test),
         "notes": f"a*text+(1-a)*CLIP; a tuned on dev ({len(dev)} q)"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase2c_fusion.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df, best_a, len(eval_q), have, pools, restricted_text_rank, fused_rank


# ============================================================ MAIN
def main():
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n = len(catalog)
    genres = [genre_set(g) for g in catalog["Genre"]]
    vc = pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0).values.astype(float)
    pop_norm = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)

    with open(EVAL / "content_relevance.json") as f:
        relevance = {int(k): set(v) for k, v in json.load(f).items()}
    queries = sorted(relevance)
    print(f"[eval] {len(queries)} queries over {n} catalog movies")

    emb = load_champion(catalog)

    ann_df = experiment_ann(emb, queries, relevance)
    rr_df, m_pure, pure_ranks, ce_rank15 = experiment_rerank(
        emb, catalog, queries, relevance, genres, pop_norm)
    fus_df, best_a, n_fuse, poster_emb, pools, txt_rank, fused_rank = experiment_fusion(
        emb, catalog, queries, relevance, genres)

    # ---- samples: semantic vs CE-rerank vs fused, a few queries ----
    samples = []
    fuse_ok = [q for q in pools if q in poster_emb]
    for q in (list(queries[:4]) + fuse_ok[:4]):
        rel = relevance[q]
        entry = {
            "query_index": q, "query_title": catalog.loc[q, "Title"],
            "query_genre": catalog.loc[q, "Genre"],
            "semantic_top5": [
                {"title": catalog.loc[i, "Title"], "genre": catalog.loc[i, "Genre"],
                 "relevant": bool(i in rel)} for i in pure_ranks[q][:5]],
            "cross_encoder_top5": [
                {"title": catalog.loc[i, "Title"], "genre": catalog.loc[i, "Genre"],
                 "relevant": bool(i in rel)} for i in ce_rank15(q)[:5]],
        }
        if q in fuse_ok:
            entry["poster_fused_top5"] = [
                {"title": catalog.loc[i, "Title"], "genre": catalog.loc[i, "Genre"],
                 "relevant": bool(i in rel)} for i in fused_rank(q, best_a)[:5]]
        samples.append(entry)
    with open(RESULTS / "samples" / "phase2c_rerank_examples.json", "w") as f:
        json.dump(samples, f, indent=2)

    # ---- headline json ----
    headline = {
        "day": 4, "project": "CineSemantics", "phase": "Phase 2c - rerank + fusion + ANN",
        "date": "2026-07-08", "backbone": {"name": CHAMPION_NAME, "hf_id": CHAMPION_ID},
        "eval": {"n_queries": len(queries), "catalog_size": n,
                 "relevance": "MovieLens co-rating 'more like this' (Day-1 set)"},
        "ann_sweep": ann_df.to_dict("records"),
        "rerank": rr_df.to_dict("records"),
        "fusion": {"rows": fus_df.to_dict("records"), "tuned_alpha_text": best_a,
                   "n_usable_queries": n_fuse},
        "champion_semantic_ndcg@10": m_pure["ndcg@10"],
    }
    with open(RESULTS / "phase2c_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)

    # ---- figures ----
    _figures(ann_df, rr_df, fus_df)
    print("\n[done] wrote phase2c_rerank_fusion.csv + ann_sweep.csv + fusion.csv + metrics.json")


def _figures(ann_df, rr_df, fus_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ANN recall vs p95 latency
        fig, ax = plt.subplots(figsize=(8, 5))
        for fam, color, marker in [("Flat", "#888888", "s"),
                                   ("IVF_FLAT", "#3b7dd8", "o"),
                                   ("HNSW", "#e07b39", "^")]:
            sub = ann_df[ann_df["index"].str.startswith(fam)]
            ax.plot(sub["lat_p95_ms"], sub["recall@10_vs_exact"], marker=marker,
                    color=color, label=fam, ms=8, lw=1.4)
        ax.set_xlabel("p95 single-query latency (ms)")
        ax.set_ylabel("recall@10 vs exact")
        ax.set_title("CineSemantics Day-4: ANN recall vs latency (Milvus index types)")
        ax.axhline(0.99, ls="--", color="#2a9d8f", lw=1, label="0.99 recall")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "phase2c_ann_recall_latency.png", dpi=130)
        plt.close(fig)

        # rerank NDCG bar
        fig, ax = plt.subplots(figsize=(8, 4.4))
        order = rr_df.sort_values("ndcg@10")
        colors = ["#888888" if "semantic" in v else "#3b7dd8" for v in order["variant"]]
        bars = ax.barh(order["variant"], order["ndcg@10"], color=colors)
        for b, v in zip(bars, order["ndcg@10"]):
            ax.text(v + 0.0003, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel("NDCG@10 (content retrieval, 1,072 held-out queries)")
        ax.set_title("CineSemantics Day-4: reranking variants")
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "phase2c_rerank_ndcg.png", dpi=130)
        plt.close(fig)

        # fusion bar
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        bars = ax.bar(fus_df["variant"], fus_df["ndcg@10_test"], color=["#888888", "#e07b39"])
        for b, v in zip(bars, fus_df["ndcg@10_test"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
        ax.set_ylabel("NDCG@10 (test half)")
        ax.set_title("CineSemantics Day-4: text vs text+poster fusion")
        plt.xticks(rotation=8, fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "figures" / "phase2c_fusion.png", dpi=130)
        plt.close(fig)
        print("[fig] saved 3 figures")
    except Exception as e:
        print(f"[fig] skipped: {e}")


if __name__ == "__main__":
    main()
