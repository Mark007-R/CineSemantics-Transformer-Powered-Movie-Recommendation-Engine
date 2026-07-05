# CineSemantics Production Upgrade — Day 1 / 10

**Date:** 2026-07-05 · **Phase:** 1 — Audit + eval set + baseline ·
**Field:** Recommendation Systems / Multimodal Retrieval

## Resume gap progress
**Gap:** The repo is marketed as a "Transformer-Powered Movie Recommendation
Engine" but has (a) no personalization, (b) an off-the-shelf embedding stack, and
(c) — most damning — **zero offline evaluation**. There is no way to claim any
recommendation quality because nothing was ever measured.
**Today's contribution:** Wrote the honest component audit and built the first
reproducible offline eval harness the project has ever had — a model-agnostic
"more like this" relevance set from MovieLens co-rating, aligned to the TMDB
catalog — and produced the first real numbers for the current system against
Random and Popularity baselines. This is the starting line every later day must beat.

## Files touched
- `docs/RECSYS_AUDIT.md` (new) — full audit; documents semantic-search-not-a-recommender, substring genre filter (`utils/milvus_vectordb.py:354`), off-the-shelf models (`utils/config.py:7-11`), and the zero-evaluation gap.
- `src/eval/build_eval.py` (new) — download MovieLens `ml-latest-small`, align to `data/9000plus.csv` by (title, year), build co-rating relevance set.
- `src/eval/baseline.py` (new) — reproduce the live retrieval (same model + text construction as `utils/text_embedder.py:77-92`, IP==cosine) without Milvus; score NDCG@10 / recall@20 / MAP@20 / precision@10 / coverage / diversity vs Random + Popularity.
- New dirs: `docs/ data/eval/ results/ reports/ tests/ src/`.

## Setup
- **Compute:** CPU (venv, Python 3.11.9). Catalog embedding (9,837 movies, MiniLM) ~2–3 min.
- **Data:** existing `data/9000plus.csv` (9,837 TMDB movies) + MovieLens `ml-latest-small` (100k ratings, 610 users) for behavioral ground truth. Media-discipline compliant (public TMDB + public MovieLens).
- **Components measured:** current text retrieval (`all-MiniLM-L6-v2`, 384-d, IP).

## Experiment 1 — Build a model-agnostic relevance set
- **Hypothesis:** User co-rating behavior gives a fair "more like this" ground truth the embeddings never saw, so it does not favor any model.
- **Method:** Align MovieLens→TMDB by (normalized title, year); relevant(A) = movies most co-liked (rating ≥ 4) with A by the same users (≥ 3 co-likers, top-30).
- **Result:**

| Metric | Value |
|---|---|
| Catalog movies matched | 4,228 (3,907 exact-year, 321 fallback) |
| Ratings used | 75,070 |
| Usable queries | 1,072 |
| Avg relevant / query | 29.97 |

- **Interpretation:** Dense enough for stable NDCG/MAP; independent of embeddings, so a fair test bench for the whole sprint.

## Experiment 2 — Baseline metrics on the current system
- **Hypothesis:** Off-the-shelf semantic search is a weak recommender because content similarity ≠ user co-preference.
- **Method:** Reproduce live pipeline exactly, rank full catalog per query, score against the relevance set; compare to Random and a non-personalized Popularity ranker.
- **Result:**

| System | NDCG@10 | Recall@20 | MAP@20 | Precision@10 | Coverage | ILD@10 |
|---|---|---|---|---|---|---|
| Random | 0.0034 | 0.0023 | 0.0006 | 0.0036 | 0.659 | 0.858 |
| **Popularity** | **0.2148** | **0.1858** | **0.1042** | **0.2360** | 0.001 | 0.639 |
| **Semantic (current)** | **0.0295** | 0.0109 | 0.0072 | 0.0237 | 0.534 | 0.678 |

- **Interpretation:** Semantic beats Random ~9× (real signal) but loses to a one-line Popularity baseline ~7× on NDCG@10. The "engine" is a decent *content* retriever and a *poor recommender*.

## Head-to-Head Comparison (running leaderboard — content retrieval, NDCG@10)
| Rank | System | NDCG@10 | Notes |
|---|---|---|---|
| 1 | Popularity (non-personalized) | **0.2148** | trivial, but the number to beat; coverage 0.001 (popularity bias) |
| 2 | Semantic — MiniLM-L6-v2 (current) | 0.0295 | off-the-shelf; the live system |
| 3 | Random | 0.0034 | floor |

*Targets to beat this sprint:* Day-2 better embeddings > 0.0295; Day-3 CF/hybrid and Day-7 SASRec must clear **0.2148**.

## Key findings (incl. what didn't work and why)
1. **GENUINE INSIGHT:** The current semantic-search recommender is beaten ~7× by "recommend the most popular movies to everyone." Content-embedding similarity surfaces same-franchise/same-theme titles (e.g. "Batman" → other Batman films) but *not* what users co-like, which is broader and popularity-driven. This is the honest headline for the whole upgrade.
2. **Popularity bias is extreme** (coverage 0.001) — it serves ~10 blockbusters to all users. So beating it *and* keeping coverage/diversity is the real bar, not just beating its NDCG.
3. **Why semantic underperforms:** it optimizes textual similarity, a proxy that diverges from co-watch behavior; no user signal is used at all. Fix = add personalization (CF, Days 3/7), not just better embeddings.
4. **What I deliberately did NOT do:** compute "personalized reco metrics" for the current system — it has no user model, so those are uncomputable for it. Only content retrieval is measurable today; that honest limitation is documented.

## Sample outputs saved
- `results/baseline_metrics.json` — headline metrics + config
- `results/baseline_leaderboard.csv` — Random / Popularity / Semantic table
- `results/samples/baseline_more_like_this.json` — 8 sample "more like this" lists with relevance flags
- `results/figures/baseline_ndcg.png` — NDCG@10 bar chart
- `data/eval/{content_relevance.json, movielens_alignment.csv, eval_manifest.json}`

## Next day (Day 2)
Phase 2a — embedding-model bake-off for content retrieval: current MiniLM vs
`all-mpnet-base-v2` vs `e5-base-v2` vs `bge-base-en-v1.5` (and an API embedding),
on this exact eval. Report NDCG@10 / recall@20 / latency / dim, and replace the
substring genre filter with proper metadata-aware retrieval. Save to
`results/phase2a_embeddings.csv`.

## Code changes
No production code modified today (audit + eval only, per Hard Rule 11: the first
refactor commit must add the eval harness before any feature). New harness lives in
`src/eval/`; production files (`utils/`, `pages/`) untouched.
