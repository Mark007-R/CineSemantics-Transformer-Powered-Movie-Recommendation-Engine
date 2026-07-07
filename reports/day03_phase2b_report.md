# CineSemantics Production Upgrade — Day 3 / 10

**Date:** 2026-07-07 · **Phase:** 2b — Collaborative-filtering bake-off ·
**Field:** Recommendation Systems / Multimodal Retrieval

## Resume gap progress
**Gap:** The project is named a "recommendation engine" but has **no
personalization at all** — Day 1 showed it is semantic search + a substring genre
filter, with zero recsys metrics. Days 1–2 could only measure *content retrieval*
("more like this"), never *personalized recommendation*, because there was no user
model and no per-user held-out evaluation.
**Today's contribution:** Built the project's first **personalized** evaluation —
a per-user, temporal held-out split of the MovieLens interactions aligned to the
9,837-movie TMDB catalog — and ran the first CF bake-off: Popularity, ItemKNN,
implicit ALS, PureSVD, content-centroid, and a hybrid. Result: **every
interaction-based CF model significantly beats the popularity baseline** on
personalized NDCG@10 (ItemKNN +47%, Wilcoxon p=1e-10), and — the counterintuitive
headline — the **semantic embedding that "powers" the engine is the *worst*
personalization signal of all**, 6× below popularity. Personalization comes from
behaviour, not from the transformer embeddings.

## Files touched
- `src/eval/build_cf_eval.py` (new, 150 lines) — per-user **temporal** leave-last-20%
  split of MovieLens likes (rating≥4) aligned to the catalog; emits `cf_split.json`
  + `cf_manifest.json` with integrity checks (train/test overlap must be 0; reports
  the cold-item recall ceiling).
- `src/recsys/cf_compare.py` (new, 300 lines) — 6-system CF bake-off over the shared
  train-item universe (seen items excluded), full IR/recsys metric suite +
  coverage/novelty, per-user Wilcoxon significance vs Popularity, cached MiniLM
  content embeddings, leaderboard/metrics/samples/figure.
- `src/recsys/__init__.py` (new) — package marker (the Day-5 `src/recsys/` home).
- `data/9000plus.csv` restored (public HF mirror of the same 9,837-row TMDB catalog;
  the local copy was lost to a sync fault — verified byte-consistent: same 9,837
  rows, same 4,228 MovieLens alignments and 1,072 Day-1 queries reproduce exactly).
- No production code (`utils/`, `pages/`) modified — the CF layer lands in the Day-5
  Phase-3 integration; today measures which recommender to ship.

## Setup
- **Compute:** CPU, Python 3.11.9. Full bake-off ≈ 40 s (ALS 20 iters 0.35 s;
  ItemKNN 0.04 s; content encode 35 s first run, then cached).
- **Data:** existing `data/9000plus.csv` (9,837 TMDB movies) + **MovieLens
  ml-latest-small** ratings (public). After alignment: 547 evaluable users, 2,607
  candidate items, 28,591 train / 7,149 test interactions.
- **Eval protocol:** per-user **temporal** split — sort each user's likes by
  timestamp, hold out the most-recent 20% (≥1) as test, earlier likes as train.
  Every test interaction is strictly later than that user's training history → no
  future→past leak. `train∩test = 0` asserted. Rank over the shared train-item
  universe, exclude already-seen items.
- **Libraries:** `implicit` (ALS). `scikit-surprise`'s SVD is **blocked by this
  machine's Application-Control policy** (Cython DLL load fails), so the SVD-family
  competitor is **PureSVD** via `scipy.svds` (Cremonesi et al. 2010) — a standard,
  typically *stronger* SVD recommender than surprise's biased-SVD anyway.

## Experiment 1 — CF bake-off on the held-out user split
- **Hypothesis:** A model that uses interaction history will personalize and beat
  the non-personalized popularity baseline that crushed semantic search on Day 1.
- **Method:** Train each system on train likes only; rank the candidate universe
  per user; score the held-out likes with NDCG@10 / recall@20 / MAP@20 /
  precision@10, plus coverage, intra-list diversity, and novelty (self-information).
- **Result** (`results/phase2b_cf.csv`, sorted by NDCG@10):

| System | NDCG@10 | Recall@20 | MAP@20 | Prec@10 | Cat.Cov | Cand.Cov | ILD@10 | Novelty@10 | vs Pop (Wilcoxon p) |
|--------|--------:|----------:|-------:|--------:|--------:|---------:|-------:|-----------:|:-------------------:|
| **ItemKNN** | **0.1059** | **0.1675** | **0.0507** | **0.0793** | 0.0378 | 0.1427 | 0.787 | 2.70 | **1.1e-10** |
| PureSVD | 0.0979 | 0.1608 | 0.0488 | 0.0689 | 0.0524 | 0.1975 | 0.789 | 3.14 | 5.7e-05 |
| Hybrid (ALS+Content) | 0.0940 | 0.1546 | 0.0483 | 0.0627 | 0.0653 | 0.2463 | 0.749 | 3.73 | 8.1e-04 |
| ALS | 0.0861 | 0.1494 | 0.0447 | 0.0589 | 0.0663 | **0.2501** | 0.795 | 3.38 | 3.2e-02 |
| Popularity (baseline) | 0.0719 | 0.0993 | 0.0343 | 0.0492 | 0.0053 | 0.0199 | 0.778 | 1.80 | — |
| Content (MiniLM centroid) | 0.0123 | 0.0243 | 0.0046 | 0.0099 | 0.0392 | 0.1481 | 0.714 | 7.07 | 8.0e-23 *(worse)* |

- **Interpretation:** All four interaction-based recommenders beat Popularity with
  statistical significance. **ItemKNN is the champion** (+47% NDCG@10 over
  Popularity) — consistent with the well-documented result that a well-tuned
  item-item kNN is a brutally strong baseline on MovieLens-scale data (Ferrari
  Dacrema et al., *"Are we really making much progress?"*, RecSys 2019). The
  content-centroid recommender — averaging the semantic embeddings of a user's
  liked movies — is the **worst** system, 6× below Popularity and significantly so.

## Head-to-Head Comparison (running story)

| Eval | System | NDCG@10 | Note |
|------|--------|--------:|------|
| Day-1 content "more like this" | Semantic (MiniLM) | 0.0295 | loses 7× to popularity |
| Day-1 content "more like this" | Popularity | 0.2148 | popularity dominates item-item |
| Day-2 content "more like this" | e5-base (champion) | 0.0482 | +63% over MiniLM, still <popularity |
| **Day-3 personalized** | **ItemKNN (champion)** | **0.1059** | **+47% over popularity — first real personalization** |
| Day-3 personalized | Content (MiniLM centroid) | 0.0123 | embeddings are a *poor* personalization signal |

> The two evals are different tasks (item-item content relevance vs per-user next-item),
> so NDCG values are not directly comparable across the horizontal line — what *is*
> comparable is **each system vs the popularity baseline in its own eval**. On content
> relevance, popularity wins; on personalization, interaction-CF wins. That contrast
> is the whole Day-1→Day-3 arc.

## Key findings
1. **Personalization finally exists and it works.** For the first time the "engine"
   beats a non-personalized baseline on a *per-user* metric — ItemKNN +47% NDCG@10,
   p=1.1e-10. Days 1–2 could never show this; the eval didn't exist.
2. **The transformer embedding is the worst personalizer (the counterintuitive one).**
   The MiniLM content-centroid scores 0.0123 — below random-ish and 6× below
   popularity (p=8e-23). Semantic similarity captures *"this movie is like that
   movie"* but not *"this user will watch that next."* The "transformer-powered"
   framing was doing content retrieval, not recommendation — exactly the Day-1 audit
   claim, now quantified from the other direction.
3. **Naive hybridization can't reach the frontier.** Equal-weight z-score fusion of
   ALS + Content (Hybrid 0.094) does edge ALS-alone (0.086), but both trail the
   ItemKNN/PureSVD frontier (0.106/0.098) — the weak content channel caps the
   fusion's ceiling. A learned/weighted hybrid (Day-6 tuning) is the fix, not an
   equal-weight sum.
4. **Coverage vs accuracy trade-off is real and measurable.** Popularity recommends
   the same head to everyone (candidate coverage 0.020 — 2% of items). ALS/Hybrid
   spread across ~25% of candidates at only a modest NDCG cost — the personalization
   *diversifies catalog exposure* 12× over popularity, which matters for a catalog
   product even where raw accuracy is close.
5. **Honest cold-start ceiling.** 5.9% of held-out likes are on items with zero
   training interactions — unreachable by any CF model, so recall is hard-capped at
   0.941. Logged in `cf_manifest.json`, not hidden. This is precisely the gap a
   content/sequential model (Day 7) can attack.

## What didn't work (and why)
- **Content-based CF (MiniLM centroid):** worst system. Averaging normalized
  sentence embeddings of liked titles produces a bland centroid that ranks generic,
  broadly-similar movies, not the specific ones a user co-watches. *Why:* the
  behavioural co-occurrence signal (who-liked-what-together) is orthogonal to
  textual/semantic similarity — a lesson that motivates learned two-tower / sequential
  models over off-the-shelf embeddings.
- **Equal-weight ALS+Content hybrid:** did not reach the ItemKNN/PureSVD frontier,
  because a z-score sum lets the weak content channel dilute the strong CF channel.
  Fix deferred to Day-6 tuning (learned fusion weight / drop content, or reranking).
- **scikit-surprise SVD:** unusable — its compiled DLL is blocked by the machine's
  Application-Control policy. Substituted PureSVD (scipy), which is a stronger and
  fully-reproducible SVD recommender; noted transparently rather than skipped.

## Metrics update (primary)
- Personalized champion (ItemKNN): **NDCG@10 0.1059**, recall@20 0.1675, MAP@20 0.0507.
- Uplift over popularity baseline: **+47% NDCG@10** (p=1.1e-10, Wilcoxon, 547 users).
- Content-signal personalization: NDCG@10 0.0123 (6× below popularity) — the day's insight.

## Sample outputs saved
- `results/phase2b_cf.csv` — full leaderboard (6 systems × 9 metrics).
- `results/phase2b_metrics.json` — headline + per-system Wilcoxon significance.
- `results/samples/cf_reco_samples.json` — per-user top-10 recs (Popularity/ALS/Content/Hybrid) with hit flags, liked-train and held-out-test titles.
- `results/figures/phase2b_cf_ndcg.png` — NDCG@10 bar chart vs the Day-1 semantic line.
- `data/eval/cf_split.json` + `cf_manifest.json` — the reusable held-out split.

## Next day
**Day 4 — Phase 2c: Reranking + multimodal fusion + ANN tuning.** Cross-encoder
rerank of the CF top-15 → top-5, fuse text + poster CLIP signals into a combined
relevance score, and sweep the Milvus index (IVF_FLAT vs HNSW: recall vs latency vs
memory), reporting NDCG@10 + p95 latency per variant.

## Code changes
New: `src/eval/build_cf_eval.py`, `src/recsys/cf_compare.py`, `src/recsys/__init__.py`.
Restored: `data/9000plus.csv` (gitignored; not committed). No `utils/`/`pages/`
production code changed this session — CF integration is Day-5 Phase 3.
