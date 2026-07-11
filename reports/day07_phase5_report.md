# CineSemantics Production Upgrade — Day 7 (Phase 5)
### Sequential / transformer recommendation — the first *actual* transformer ranker
**Date:** 2026-07-11 · **Day 07 of 10** · Recommendation Systems / Multimodal Retrieval

---

## Resume gap progress

**Gap:** the project is called a "**Transformer-Powered** Movie Recommendation Engine,"
but through Day 6 the transformer only ever produced *embeddings for retrieval*
(MiniLM→e5, CLIP). The personalized **ranker** was co-occurrence CF (ItemKNN, Day-3
champion) and matrix factorization (tuned ALS, Day-6) — both **order-blind**: they
treat a user's likes as a *set*.

**Today's contribution:** added the first transformers that model a user's likes as an
ordered **sequence** and predict the next item — **SASRec** (causal self-attention) and
**BERT4Rec** (bidirectional Cloze) — and benchmarked them honestly against the CF
co-champions on the **same** per-user temporal split, under **two** protocols
(strict next-item leave-one-out *and* the Day-3 full-list). Net: the transformer finally
earns the project's name on the next-item task — but the honest verdict is a **tie**, not
a conquest, and that is the resume-grade finding.

---

## Files touched

| File | Lines | What |
|------|-------|------|
| `src/recsys/sequential.py` | new, 300 | SASRec, BERT4Rec (PyTorch, CPU), Markov baseline; finite-mask attention (NaN-safe left-padding) |
| `src/eval/build_sequential_eval.py` | new, 150 | derive next-item + dense-context eval from the Day-3 `cf_split.json` (zero split drift) |
| `src/recsys/seq_compare.py` | new, 320 | 7-system bake-off, 2 protocols, debiasing sweep, grounded explanations, figures |
| `tests/test_day7_sequential.py` | new, 90 | 5 regression tests (leakage, vocab, learns-order, leaderboard sanity) |

Reuses Day-3 metric functions verbatim (`src/recsys/cf_compare.py`: `ndcg_at_k`,
`recall_at_k`, `ap_at_k`, …) so the numbers are directly comparable to the CF leaderboard.

---

## Setup

- **Compute:** CPU only (torch 2.1.2+cpu, 10 threads). Training SASRec 61 s, BERT4Rec 80 s,
  SASRec-dense 54 s. Everything reproducible from `python -m src.recsys.seq_compare`.
- **Data:** public MovieLens `ml-latest-small` aligned to the 9.8K TMDB catalog (Day-1
  alignment). Split = the **identical** Day-3 per-user temporal leave-last-20%
  (547 evaluable users, 2607-item train-only vocab, no cold items).
- **Two eval protocols (same users, same vocab):**
  - **Next-item (leave-one-out):** rank the single *immediate-next* held-out like → **HR@10 / NDCG@10** (the canonical sequential-rec metric).
  - **Full-list (Day-3):** rank *all* held-out likes → NDCG@10 / recall@20 / MAP@20 / coverage / diversity / novelty.

---

## Experiments

### Exp 1 — SASRec / BERT4Rec vs the CF co-champions (headline)

**Hypothesis:** an order-aware transformer should beat order-blind CF, at least on the
strict next-item task where recency matters.

| System | NI-HR@10 | NI-NDCG@10 | FL-NDCG@10 | FL-recall@20 | FL-MAP@20 | Cov | Novelty | fit s |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Popularity | 0.0987 | 0.0549 | 0.0719 | 0.0993 | 0.0343 | 0.005 | 1.80 | 0.0 |
| Markov (1st-order) | 0.1042 | 0.0559 | 0.0718 | 0.0926 | 0.0342 | 0.128 | 4.50 | 0.2 |
| **ItemKNN** (Day-3 champ) | 0.1572 | 0.0778 | **0.1059** | 0.1675 | 0.0507 | 0.038 | 2.70 | 0.0 |
| **ALS_tuned** (Day-6) | **0.1664** | **0.0811** | 0.1058 | **0.1763** | **0.0537** | 0.050 | 2.94 | 0.4 |
| **SASRec** | 0.1444 | 0.0757 | 0.0915 | 0.1499 | 0.0488 | 0.094 | 4.16 | 61 |
| **BERT4Rec** | 0.1627 | 0.0797 | 0.0933 | 0.1688 | 0.0485 | 0.064 | 3.87 | 80 |
| SASRec_dense | 0.1316 | 0.0697 | 0.0738 | 0.1307 | 0.0386 | 0.100 | 4.56 | 54 |

**Paired Wilcoxon (per-user):**

| Comparison | Metric | p | Read |
|------------|--------|---|------|
| BERT4Rec vs ItemKNN | next-item NDCG | **0.903** | **tie** — transformer draws level, does not surpass |
| SASRec vs ItemKNN | next-item NDCG | 0.777 | tie |
| SASRec vs ItemKNN | full-list NDCG | **0.029** | ItemKNN significantly better on the broad list |
| BERT4Rec vs ItemKNN | full-list NDCG | 0.114 | n.s. (BERT4Rec closes the gap) |
| SASRec vs Markov | full-list NDCG | **0.013** | higher-order context beats last-item-only |
| SASRec_dense vs SASRec | full-list NDCG | **0.006** | denser context *significantly hurt* |

**Interpretation.** On the **next-item** task the transformer (BERT4Rec) matches the tuned
CF champions to a statistical dead heat (p≈0.90) — order modeling earns the project's name
*here*. But on the **full-list** metric, order-blind co-occurrence/MF still win: with only
547 users and ~52 likes each, the transformers are data-starved relative to CF's dense
item-item statistics. **BERT4Rec > SASRec** on this sparse data (bidirectional Cloze sees
context from both sides), matching the literature that BERT4Rec favors shorter/sparser
sequences.

### Exp 2 — Does denser input history help? (sensitivity)

**Hypothesis:** feeding *all* rated movies (not just ≥4-star likes) as input context gives
the transformer 90% more signal (avg history 99.5 vs 52.3 interactions) → better next-like
prediction.

**Result:** the opposite. `SASRec_dense` FL-NDCG@10 **0.0738 vs 0.0915** for likes-only
SASRec (p=0.006, significant). **Why:** sub-4-star ratings are *negative/lukewarm* signal;
mixing them into the history as if they were positives dilutes the "what this user
actually likes" trajectory. More data ≠ better when the extra data is noisier signal — a
clean, transferable lesson for the production feature store.

### Exp 3 — Popularity debiasing at the sequence level

**Hypothesis (from Day-6):** `score − β·log(pop)` trades accuracy for coverage/novelty.

| β | FL-NDCG@10 | Catalog cov | Novelty@10 |
|---|:---:|:---:|:---:|
| 0.0 | 0.0933 | 0.064 | 3.87 |
| 0.5 | 0.0792 | 0.078 | 4.57 |
| 1.0 | 0.0528 | 0.083 | 5.85 |
| 2.0 | 0.0113 | 0.050 | 8.29 |

**Interpretation.** Same honest trade-off as Day-6: β>0 monotonically destroys NDCG because
held-out likes on this MovieLens split genuinely skew popular, so demoting popular items
demotes the correct answers. Debiasing is a **coverage lever, not a free lunch** — kept as
an instructive negative and a tunable knob, off by default.

---

## Head-to-Head Comparison (running leaderboard, full-list NDCG@10)

| Day | System | FL-NDCG@10 | recall@20 | Note |
|-----|--------|:---:|:---:|------|
| 1 | Semantic search (MiniLM) | 0.0295 | — | pre-personalization |
| 3 | ItemKNN (CF champion) | **0.1059** | 0.1675 | order-blind |
| 6 | ALS tuned (Optuna) | 0.1058 | 0.1763 | order-blind |
| **7** | **BERT4Rec** (transformer) | 0.0933 | 0.1688 | **order-aware; ties on next-item** |
| **7** | **SASRec** (transformer) | 0.0915 | 0.1499 | order-aware |
| 7 | Markov (1st-order) | 0.0718 | 0.0926 | order-aware but memoryless |

---

## Key findings

1. **The first real transformer earns the project's name on next-item — as a tie.** BERT4Rec
   matches ItemKNN/ALS on next-item NDCG@10 (p≈0.90). The honest headline is *parity*, not a
   conquest: on 547 users, self-attention has no data advantage over dense co-occurrence
   statistics. Reporting the tie (with a Wilcoxon p-value) is the resume signal.
2. **BERT4Rec > SASRec on sparse data.** Bidirectional Cloze (0.0797 next-item NDCG) beats
   causal SASRec (0.0757); expected for short, sparse sequences.
3. **Denser context significantly *hurt* (p=0.006).** Adding lukewarm (<4★) ratings as
   history degraded next-like prediction — a counterintuitive, well-controlled negative.
4. **Higher-order context is real signal:** SASRec significantly beats a first-order Markov
   chain (p=0.013), so the transformer's gain over "just use the last item" is genuine.
5. **Transformers are markedly less popularity-biased:** SASRec catalog coverage 0.094 vs
   ItemKNN 0.038 (2.5×) and higher novelty (4.16 vs 2.70) at a modest NDCG cost — a real
   diversity/serendipity edge even where they trail on relevance.

## What didn't work (and why)

- **Denser history context** (Exp 2) — noisier positives dilute the like-trajectory.
- **Popularity debiasing** (Exp 3) — held-out likes skew popular, so debiasing demotes
  correct items (replicates Day-6).
- **Beating CF outright on the full list** — data scarcity (547 users) caps the transformer;
  the honest expectation is that more interaction data (or fine-tuning on co-watch logs)
  closes it. Logged as a Day-9/future-sprint candidate, not overclaimed.

## Sample outputs saved

- `results/phase5_sequential.csv` — 7-system, 2-protocol leaderboard
- `results/phase5_debias.csv` — debiasing sweep
- `results/phase5_metrics.json` — headline + significance + model params
- `results/samples/phase5_explanations.json` — grounded "because you liked X" cards
  (from the transformer's own learned item embeddings — no hallucinated titles)
- `results/samples/phase5_seq_reco.json` — per-user next-item hit inspection
- `results/figures/phase5_nextitem_ndcg.png`, `phase5_fulllist_ndcg.png`

Example explanation (BERT4Rec, user 1): recommends *Aliens* → "Because you liked **Alien**
and **Indiana Jones and the Last Crusade**"; *The Untouchables* → "Because you liked **A Few
Good Men** and **Face/Off**." Grounded in learned item similarity, every cited title real
and in-catalog.

---

## Next day (Day 8 — Phase 6)

Frontier comparison + ablation: Claude Opus 4.6 / GPT-5.4 zero-shot ("recommend N movies for
a user who liked X, Y, Z") mapped to the catalog → NDCG@10 / precision vs the specialized
stack, plus **hallucination rate** (% off-catalog titles), latency, and cost. Full ablation:
semantic retrieval → +better embeddings → +CF → +rerank → +sequential → +diversity. **[POST · PHASE-WRAP]**

## Code changes

- New `src/recsys/sequential.py` (SASRec/BERT4Rec/Markov). Key correctness fix: left-padding
  produced NaN via `key_padding_mask`'s `-inf` (all-masked rows → NaN softmax → NaN grads);
  replaced with a **finite** `-1e9` additive mask so masked rows softmax to harmless uniform.
  Validated the architecture learns order (100% next-in-top-5 on a deterministic chain).
- New `src/eval/build_sequential_eval.py`, `src/recsys/seq_compare.py`.
- New `tests/test_day7_sequential.py` (5 tests). **Full suite: 22 passed** (Day-5 10 + Day-6 7 + Day-7 5).
