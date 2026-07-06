# CineSemantics Production Upgrade — Day 2 / 10

**Date:** 2026-07-06 · **Phase:** 2a — Embedding-model bake-off (content retrieval) ·
**Field:** Recommendation Systems / Multimodal Retrieval

## Resume gap progress
**Gap:** The engine ships an off-the-shelf `all-MiniLM-L6-v2` (384-d) text encoder
that was never benchmarked, plus a genre filter that is a raw **substring match**
(`utils/milvus_vectordb.py:354`) masquerading as metadata-aware retrieval.
**Today's contribution:** Ran the first head-to-head embedding benchmark the
project has ever had — 5 encoders on the exact Day-1 held-out eval — and replaced
the substring genre filter with a proper metadata-aware re-ranker, tuned on a dev
split and reported on a disjoint test split. Result: better embeddings lift
content-retrieval NDCG@10 by **+63% relative** (0.0295 → 0.0482), but the honest
headline is that *embeddings alone still lose ~4× to a popularity baseline* — the
gap closes only with the metadata/behavioural signal, motivating Day-3 CF.

## Files touched
- `src/eval/embedding_comparison.py` (new, 300 lines) — 5-model bake-off on the
  Day-1 relevance set (same text construction, same metrics), per-model prefix
  conventions (E5 `query: `, BGE none for symmetric s2s), disk-cached embeddings,
  latency/dim instrumentation, and the metadata-aware re-ranker with dev/test tuning.
- `.gitignore` — ignore `results/emb_cache/` (cached `.npy` embeddings, ~75 MB).
- No production code (`utils/`, `pages/`) modified — the champion swap + filter
  replacement land in the Day-5 Phase-3 integration; today measures which to swap.

## Setup
- **Compute:** CPU (venv, Python 3.11.9). Full run ≈ 27 min: each 768-d base model
  encodes 9,837 movies in ~7.9 min (~48 ms/doc) vs MiniLM's 72 s (~7 ms/doc).
- **Data:** existing `data/9000plus.csv` (9,837 TMDB movies) + Day-1 MovieLens
  co-rating relevance set (1,072 held-out queries, ~30 relevant each). Public-only.
- **Eval:** identical to Day-1 (`src/eval/baseline.py`) — item-item "more like this"
  over the full catalog, NDCG@10 / recall@20 / MAP@20 / precision@10 / coverage / ILD.

## Experiment 1 — Embedding-model bake-off (content retrieval)
- **Hypothesis:** A stronger sentence encoder than MiniLM raises content-retrieval
  quality on the same held-out behavioural ground truth.
- **Method:** Encode the catalog with each model (authors' symmetric-similarity
  prefix), rank the full catalog per query by cosine, score on the Day-1 set.
- **Result:**

| Model | Dim | NDCG@10 | Recall@20 | MAP@20 | Prec@10 | Coverage | ILD@10 | ms/doc |
|---|---|---|---|---|---|---|---|---|
| **e5-base-v2** (champion) | 768 | **0.0482** | 0.0205 | 0.0116 | 0.0410 | 0.472 | 0.556 | 49.4 |
| mpnet-base-v2 | 768 | 0.0480 | 0.0218 | 0.0119 | 0.0420 | 0.482 | 0.682 | 47.7 |
| bge-base-en-v1.5 | 768 | 0.0393 | 0.0166 | 0.0098 | 0.0320 | 0.505 | 0.594 | 48.4 |
| bge-small-en-v1.5 | 384 | 0.0376 | 0.0153 | 0.0097 | 0.0299 | 0.515 | 0.602 | 13.9 |
| MiniLM-L6-v2 (current) | 384 | 0.0295 | 0.0109 | 0.0072 | 0.0237 | 0.534 | 0.678 | 7.3 |

- **Interpretation:** e5-base-v2 and mpnet-base-v2 are a statistical tie at the top
  (+63% / +63% relative NDCG@10 over MiniLM). The best 384-d option, bge-small, gets
  **+27% for only 2× MiniLM's cost** — the real operating sweet spot, since the
  768-d models cost ~7× the encode time for a further ~28% relative lift. bge-*base*
  underperforms mpnet/e5 here, a reminder that leaderboard rank on generic STS does
  not transfer to a movie "more like this" task.

## Experiment 2 — Metadata-aware retrieval vs the substring genre filter
- **Hypothesis:** Replacing the current substring genre hard-filter with a proper
  metadata-aware re-ranker (cosine + genre-Jaccard + popularity prior) on the
  champion embeddings improves ranking without collapsing recall.
- **Method:** On champion (e5-base) embeddings, tune blend weights `a` (genre
  Jaccard) and `b` (popularity prior) on a **dev half** of queries, report on the
  **disjoint test half**. Baselines: the current substring filter, and pure semantic.
- **Result (test split, 536 queries):**

| Variant | NDCG@10 | Recall@20 |
|---|---|---|
| substring_genre_filter (current) | 0.0443 | 0.0202 |
| champion_pure_semantic | 0.0439 | 0.0192 |
| **metadata_aware_rerank** (a=0.0, b=0.1) | **0.1755** | **0.0677** |

- **Interpretation:** Two findings, one expected and one counterintuitive. (1) The
  current **substring genre filter adds essentially nothing** — 0.0443 vs 0.0439
  pure semantic (+0.0004). It is doing no real work as a "recommender." (2) The
  metadata-aware re-ranker nearly **4×'s NDCG@10** (0.0439 → 0.1755) and 3.5×'s
  recall — but the dev-tuned weights landed at **a=0.0, b=0.1**: the grid threw away
  genre Jaccard entirely and kept only the popularity prior. So the entire lift comes
  from re-ranking the semantically-plausible top-200 by popularity, not from genre.

## Head-to-Head Comparison (running leaderboard — content retrieval, NDCG@10)
| Rank | System | NDCG@10 | Notes |
|---|---|---|---|
| 1 | Popularity (non-personalized) | 0.2148 | Day-1 bar to beat; coverage 0.001 |
| 2 | **e5-base + metadata rerank (pop)** | **0.1755** | Day-2 best; still < popularity, but coverage-preserving |
| 3 | e5-base-v2 (pure semantic) | 0.0482 | Day-2 embedding champion (+63% vs MiniLM) |
| 4 | mpnet-base-v2 | 0.0480 | tied co-champion |
| 5 | bge-small-en-v1.5 | 0.0376 | best cost/quality (384-d, 2× MiniLM cost) |
| 6 | Semantic — MiniLM (Day-1 current) | 0.0295 | the live system |
| 7 | Random | 0.0034 | floor |

*Still to clear 0.2148:* Day-3 CF/hybrid and Day-7 SASRec (true personalization).

## Key findings (incl. what didn't work and why)
1. **GENUINE INSIGHT:** Upgrading the encoder is real but *bounded* — the best
   embedding (+63% relative) is still ~4.5× below a one-line popularity baseline.
   You cannot out-embed the missing personalization; content similarity ≠ co-watch.
   This quantifies exactly how much of the gap is "better encoder" (a little) vs
   "needs a user model" (most of it) — the case for Day-3 CF, in numbers.
2. **The substring genre filter is inert.** Reproduced faithfully, it moves NDCG@10
   by +0.0004 over pure semantic. The "genre-aware retrieval" claim was hollow.
3. **What didn't work — genre Jaccard.** The dev grid drove the genre weight to
   **zero**; only the popularity prior helped. Movies co-liked by users are *not*
   reliably same-genre, so genre overlap is a poor relevance signal on this ground
   truth. Honest negative result — I did not force a genre term in to tell a tidier
   story.
4. **bge-base > mpnet on generic benchmarks, but < mpnet here.** Off-the-shelf
   leaderboard rank did not predict task performance; measuring on *our* eval mattered.

## Sample outputs saved
- `results/phase2a_embeddings.csv` — 5-model bake-off table
- `results/phase2a_metadata_rerank.csv` — substring filter vs metadata-aware
- `results/phase2a_metrics.json` — headline dict (append-friendly)
- `results/figures/phase2a_ndcg.png` — NDCG@10 bar chart vs popularity line
- `results/samples/phase2a_champion_more_like_this.json` — 8 champion "more like this" lists

## Next day (Day 3)
Phase 2b — collaborative filtering bake-off: implicit ALS vs item-item kNN vs SVD
(`surprise`) vs content-centroid vs hybrid, on a held-out **user/time** split with
precision@k / recall@k / NDCG@k / MAP / coverage / novelty. This is where the
popularity bar (0.2148) should finally fall — personalization, not embeddings.

## Code changes
No production code modified (per the sprint rule: measure first, integrate the
champion on Day 5). New harness in `src/eval/embedding_comparison.py`; production
`utils/`/`pages/` untouched. The champion (e5-base-v2, with mpnet as the
prefix-free fallback) and the metadata-aware re-ranker are the Day-5 swap targets
for `utils/text_embedder.py:load_model` and `utils/milvus_vectordb.py:354`.
