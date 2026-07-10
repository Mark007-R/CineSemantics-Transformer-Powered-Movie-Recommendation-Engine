# CineSemantics Production Upgrade — Day 6 / 10

**Phase 4: CF hyperparameter tuning + error analysis + targeted diversity fix**
**Date:** 2026-07-10

---

## Resume gap progress

**Gap:** The project was shipped as a "recommendation engine" but through Day 1 had
zero recsys evaluation and no personalization. Days 1–5 built the eval harness, an
honest CF bake-off, and integrated ItemKNN as the production champion. The remaining
gap after Day 5: *"the CF is stood up but never tuned, and no one has looked at where
it actually fails."* A recruiter reads "collaborative filtering" and asks two things —
did you tune it, and do you know its failure modes?

**Today's contribution:** (1) A ≥40-trial Optuna study on ALS that closes a **+23%**
gap over the default hyperparameters and lifts ALS from 3rd-place CF to **co-champion**,
tuned on an **inner validation fold carved from train only** so the test split is never
touched (the classic HPO-leak, avoided and asserted in a test). (2) A structured error
analysis of the champion's 30 worst users that names the **dominant failure mode
(genre over-concentration, 21/30)**. (3) A targeted, honest fix — **MMR genre-diversity
reranking** wired into the production recommender — that trades **−0.3pp NDCG@10 for
+6.4pp intra-list diversity**, with a documented **negative result** (popularity
debiasing) that would have been the "obvious" choice and is catastrophic here.

---

## Files touched

| File | Change |
|------|--------|
| `src/recsys/tune_als.py` | **new** — Optuna ALS study (inner-val), champion error analysis, MMR/debias fix sweep, figures |
| `src/recsys/recommender.py` | added `attach_genres()` + MMR `diversity` path to `recommend()` (lines ~78–150); backward compatible (`diversity=None` → unchanged) |
| `tests/test_day6_tuning.py` | **new** — 7 tests: tuning-beats-default, HPO-used-inner-val guard, MMR backward-compat / fallback / de-concentration / rank-1-relevance |
| `results/phase4_*.csv`, `results/phase4_metrics.json`, `results/leaderboard.csv` | tuning log, error analysis, fix sweep, headline, running leaderboard |
| `results/figures/phase4_*.png` | tuning bars, failure distribution, accuracy-vs-coverage tradeoff |
| `results/samples/phase4_error_cases.json` | 8 worst-user cases with recs, held-out, failure label |

---

## Setup

- **Compute:** CPU (system Python 3.11 env; `implicit` 0.7.3, `optuna` 4.8.0). Full day
  runs end-to-end in ~30 s.
- **Data:** the Day-3 per-user **temporal** held-out split (`data/eval/cf_split.json`) —
  547 evaluable users, 2,607 candidate items, 7,149 test interactions. Public MovieLens
  ratings aligned to the 9.8K-movie TMDB catalog. No new data; media discipline respected.
- **HPO protocol:** an **inner validation fold** built from each user's last 20% of their
  *training* likes (temporal, leakage-free). Optuna tunes against inner-val NDCG@10; the
  winner is refit on full train and scored **once** on the real test split.

---

## Experiment 1 — Optuna tuning of ALS

**Hypothesis:** Day-3 ranked ALS 3rd among CF models (NDCG@10 0.086, below ItemKNN 0.106
and PureSVD 0.098). Was ALS genuinely weaker, or were its *defaults* (factors=64,
alpha=15) simply wrong for this sparse split?

**Method:** 40-trial TPE study over `factors∈[16,256] (log)`, `regularization∈[1e-3,1] (log)`,
`iterations∈[5,60]`, `alpha∈[1,40]`, scored on the inner-val fold; refit best on full
train; scored once on test. Paired Wilcoxon vs default on per-user NDCG.

| Model | factors | reg | iters | alpha | NDCG@10 | recall@20 | MAP@20 | prec@10 | coverage |
|-------|--------:|----:|------:|------:|--------:|----------:|-------:|--------:|---------:|
| ALS default | 64 | 0.05 | 20 | 15.0 | 0.0861 | 0.1494 | 0.0447 | 0.0589 | 0.0663 |
| **ALS Optuna** | **27** | **0.043** | **35** | **3.75** | **0.1058** | **0.1763** | **0.0537** | **0.0773** | 0.0495 |
| ItemKNN (champion) | — | — | — | — | 0.1059 | 0.1675 | 0.0507 | 0.0793 | 0.0378 |

**Result:** Tuned ALS **+23% NDCG@10** over default (0.0861 → 0.1058), statistically
significant (Wilcoxon p ≈ 0.005). It now **ties the ItemKNN champion on NDCG** and
actually **edges it on recall@20 (0.176 vs 0.168) and MAP@20 (0.054 vs 0.051)** while
covering more of the catalog.

**Interpretation:** The Day-3 "ALS is a weak CF" conclusion was an **artifact of default
hyperparameters, not the algorithm.** The recovered optimum is a *small, lightly-corrected*
model — the top-5 trials all converge on **factors ≈ 20–27** and **alpha ≈ 3.5** (vs the
default 64 / 15). On a 547-user / 2.6K-item split, a big-factor high-confidence ALS
over-fits; shrinking it is what unlocks the personalization signal. ALS is now a genuine
co-champion, so we keep ItemKNN in production (identical NDCG, simpler, no training) but
have a tuned, recall-stronger MF ready for the sequential-model comparison on Day 7.

---

## Experiment 2 — Error analysis on the champion's 30 worst users

**Hypothesis:** A single dominant failure mode drives the low-NDCG tail, and it is
addressable without retraining.

**Method:** Rank all 547 users by ItemKNN NDCG@10; take the 30 worst (all NDCG@10 = 0).
For each, compute recommended-item popularity percentile, top-10 single-genre share,
the user's own taste popularity, and how popular/available their held-out items are.
Classify each into one dominant bucket.

| Failure mode | Count (of 30) | Signature |
|--------------|--------------:|-----------|
| **Genre over-concentration** | **21** | top-10 dominated by one genre (share ≥ 0.6) |
| Popularity bias | 8 | recs are ≥ 0.85 popularity-percentile blockbusters |
| Niche / cold taste | 1 | user's or test items' popularity too low to hit |

**Result:** **Genre over-concentration is the dominant failure (21/30).** ItemKNN, summing
item-item cosine over a user's history, piles the top-10 into whatever single genre their
history leans toward — a user who logged action films gets ten action films, and their
actual next watch (a comedy, a drama) never surfaces in the window.

**Interpretation:** The tail isn't "the model is dumb," it's "the model is *monotonous*."
That's a ranking-diversity problem, not a relevance problem — which points at a rerank
fix, not a bigger model.

---

## Experiment 3 — Targeted fix: MMR diversity vs popularity debiasing

**Hypothesis:** Re-diversifying the top-10 by genre recovers some of the lost tail; the
"textbook" alternative (popularity debiasing) should also help.

**Method:** On the champion scores, sweep two reranks over the held-out split:
popularity debiasing (`score / pop^β`) and MMR (`λ·rel − (1−λ)·max genre-Jaccard`).

| Variant | NDCG@10 | recall@20 | coverage | ILD@10 | novelty@10 |
|---------|--------:|----------:|---------:|-------:|-----------:|
| ItemKNN (no fix) | 0.1059 | 0.1675 | 0.0378 | 0.7872 | 2.696 |
| debias β=0.25 | 0.0254 | 0.0608 | 0.0832 | 0.7838 | 7.451 |
| debias β=0.50 | 0.0044 | 0.0071 | 0.0478 | 0.7904 | 9.029 |
| debias β=0.75 | 0.0043 | 0.0058 | 0.0413 | 0.7995 | 9.082 |
| **MMR λ=0.7** | **0.1032** | **0.1649** | 0.0366 | **0.8508** | 2.742 |
| MMR λ=0.5 | 0.0978 | 0.1522 | 0.0362 | 0.8836 | 2.788 |
| MMR λ=0.3 | 0.0928 | 0.1399 | 0.0372 | 0.9059 | 2.860 |

**Result:** **MMR λ=0.7 is the recommended fix** — it lifts intra-list diversity from
0.787 → **0.851 (+6.4pp)** for only **−0.3pp NDCG@10 and −0.3pp recall@20**, directly
attacking the dominant genre-over-concentration mode. **Popularity debiasing is a
catastrophe** here: even the gentlest β=0.25 collapses NDCG@10 by 76% (0.106 → 0.025).

**Interpretation (the genuine insight):** On this catalog, *popular items ARE what
held-out users actually watch* — the popularity signal is real relevance, not just bias,
so dividing it out throws away the very thing the model needs. The "obvious" fairness
lever is exactly wrong; the failure was **within-list genre monotony, not popularity
skew**, and only the fix that targets the *diagnosed* mode (MMR) pays off. MMR λ=0.7 is
now available in the production recommender via `recommend(..., diversity=0.7)`.

---

## Running leaderboard (personalized, held-out users)

| System | NDCG@10 | recall@20 | Δ vs Day-3 champ | Verdict |
|--------|--------:|----------:|-----------------:|---------|
| Day-1 semantic search | 0.0295 | — | — | no personalization |
| ItemKNN (Day-3 champ) | 0.1059 | 0.1675 | — | production ranker |
| ALS default | 0.0861 | 0.1494 | −0.020 | mis-tuned |
| **ALS Optuna (Day-6)** | **0.1058** | **0.1763** | **≈ tie / +recall** | co-champion |
| ItemKNN + MMR λ=0.7 (Day-6) | 0.1032 | 0.1649 | −0.003 | diversity fix (+6.4pp ILD) |

---

## Key findings

1. **Default hyperparameters lied about ALS.** Tuning (+23% NDCG@10) turned a 3rd-place
   CF into a co-champion. The optimum is a *small* model (factors 27, alpha 3.75) — big
   MF over-fits this sparse split. Robust across the top-5 trials, not a seed fluke.
2. **The low-NDCG tail is a diversity problem, not a relevance problem.** 21/30 worst
   users fail from genre over-concentration; ItemKNN returns monotone single-genre lists.
3. **The obvious fix is the wrong fix (negative result).** Popularity debiasing collapses
   NDCG by 76% because popularity here is genuine signal. Only MMR — which targets the
   *diagnosed* failure — improves diversity at negligible accuracy cost.
4. **HPO leakage avoided by construction.** Tuning happened on an inner train-only fold;
   a regression test asserts the test split was never used to pick hyperparameters.

## What didn't work (and why)

- **Popularity debiasing (any β).** Destroys NDCG/recall. *Why:* in this MovieLens-derived
  split, held-out likes skew popular, so `score / pop^β` demotes exactly the correct items.
  Kept in the report as the instructive negative result.
- **Beating ItemKNN outright with ALS.** Tuned ALS ties on NDCG and wins on recall/MAP but
  doesn't dominate — so ItemKNN stays the default ranker (simpler, no fit). Not a failure;
  an honest tie that gives us a second strong ranker for Day 7.

## Sample outputs saved

- `results/phase4_als_tuning.csv`, `results/phase4_optuna_trials.csv`
- `results/phase4_error_analysis.csv` (30 worst users + labels)
- `results/phase4_diversity_fix.csv`
- `results/phase4_metrics.json`
- `results/samples/phase4_error_cases.json` (8 worst cases, recs vs held-out)
- `results/figures/phase4_als_tuning.png`, `phase4_failures.png`, `phase4_fix_tradeoff.png`

## Next day (Day 7 — Phase 5)

Sequential / transformer recommendation — the first *actual* transformer in this
"transformer-powered" project: train SASRec / BERT4Rec on the MovieLens interaction
sequences and compare next-item HR@10 / NDCG@10 against today's co-champions (ItemKNN and
tuned ALS). Add popularity debiasing's honest counterpart at the sequence level and
LLM-generated "because you liked X" explanations on the chosen ranker.

## Code changes

- `recommend(..., diversity=λ)` MMR path + `attach_genres()` in `ItemKNNRecommender`
  (backward compatible; `diversity=None` reproduces the exact Day-5 behaviour).
- New `src/recsys/tune_als.py`; new `tests/test_day6_tuning.py` (7 tests, all green;
  full Day-5 + Day-6 suite: 17 passed).
