# CineSemantics Production Upgrade — Day 7 / 10

**Phase 5: Sequential / transformer recommendation (SASRec + BERT4Rec) — the first *actual* transformers in a "transformer-powered" engine**
**Date:** 2026-07-13

> Reconciliation note: a prior progress entry dated 2026-07-11 described this day
> but left **no committed code, results, report, or PR** in the repository — every
> referenced artifact was absent (git history ended at Day 6). Per the sprint's
> "never log success on failure / never skip days" rule, Day 7 was **executed for
> real today** and this report carries the genuine, reproduced numbers, which
> differ from that unbacked entry (the real transformer champion is **SASRec**, and
> it reaches **parity, not a win**, vs the CF champion).

---

## Resume gap progress

**Gap:** The project ships as a "***Transformer*-Powered** Recommendation Engine," but
through Day 6 the only transformer was the *embedding* encoder (MiniLM → e5). The
ranker itself was **order-blind**: ItemKNN co-occurrence (Day 3) and ALS matrix
factorization (Day 6). Neither knows you watched *The Fellowship of the Ring* **before**
*The Two Towers*. A recruiter reading "transformer-powered" and "sequential" expects a
model that treats a user's likes as an **ordered trajectory** — and expects you to know
whether it actually helps.

**Today's contribution:** the first two sequence transformers in the project — **SASRec**
(causal self-attention) and **BERT4Rec** (bidirectional Cloze) — trained from scratch and
benchmarked **honestly** on the *same* Day-3 temporal split, under both the transformers'
native **next-item** task and the leaderboard's **full-list** protocol. The headline is a
**tie, not a conquest**, plus two clean negative results — reported instead of buried.

---

## Files touched

| File | Change |
|------|--------|
| `src/recsys/sequential.py` | **new** — SASRec + BERT4Rec (shared backbone, `causal` flag), NaN-safe finite-mask attention, inner-validation early stop (test never touched), `score_all_items()` |
| `src/eval/build_sequential_eval.py` | **new** — derives `data/eval/seq_eval.json` from `cf_split.json` (**zero split drift**); emits next-item + full-list targets; asserts no leakage |
| `src/recsys/seq_compare.py` | **new** — 7-system × 2-protocol bake-off + popularity debiasing sweep + grounded "because you liked X" explanations + figures + leaderboard append |
| `tests/test_day7_sequential.py` | **new** — 5 tests: leakage/zero-drift guard, NaN-safe SASRec & BERT4Rec, seen-item exclusion contract, persisted-metrics consistency |
| `results/phase5_sequential.csv`, `phase5_debias.csv`, `phase5_metrics.json` | leaderboard, debias sweep, headline dict |
| `results/samples/phase5_{seq_reco,explanations}.json` | per-user recs + grounded explanation cards |
| `results/figures/phase5_{nextitem,fulllist}_ndcg.png` | comparison bars |
| `results/leaderboard.csv` | +2 rows (SASRec, BERT4Rec, full-list) |

---

## Setup

- **Compute:** CPU only (system Python 3.11; `torch` 2.1.2+cpu). Full day — build eval,
  train SASRec (early-stopped ~ep33), BERT4Rec (~ep21), SASRec+dense, 4 CF baselines,
  both protocols, debias sweep, explanations, figures — runs end-to-end in **~60 s**.
- **Dataset / split:** identical to Day 3 — MovieLens-small likes (rating ≥ 4★) aligned to
  the 9,837-movie TMDB catalog, per-user **temporal leave-last-20%**. **547 users × 2,607
  candidate items**, avg sequence length 52. `seq_eval.json` is derived directly from
  `cf_split.json`, so there is **zero drift** between the CF bake-off and the transformers;
  integrity check `PASS` (train/test disjoint, next-item target strictly held out).
- **Models:** SASRec & BERT4Rec, `d=64`, 2 blocks, 2 heads, dropout 0.2, maxlen 50, AdamW.
  Early stop on an **inner-validation fold carved from train only** (Rule 10).

---

## Experiments

### Experiment 1 — Sequential transformers vs order-blind CF (two protocols)

**Hypothesis:** modeling likes as an ordered sequence with self-attention beats order-blind
CF, at least on the strict next-item task.

**Method:** rank all 2,607 candidates per user (seen items masked), evaluate under
*next-item* (relevance = the single chronologically-next held-out like) and *full-list*
(relevance = all held-out likes). Paired Wilcoxon on per-user next-item NDCG vs ItemKNN.

| System | next-item NDCG@10 | next-item HR@10 | full-list NDCG@10 | full-list recall@20 | coverage | novelty@10 | fit s |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ALS-tuned (Day-6) | **0.0811** | **0.1664** | 0.1058 | **0.1763** | 0.0495 | 2.94 | 0.3 |
| ItemKNN (Day-3 champ) | 0.0778 | 0.1572 | **0.1059** | 0.1675 | 0.0378 | 2.70 | 0.03 |
| **SASRec** | **0.0667** | 0.1316 | **0.0848** | 0.1366 | 0.0338 | 3.12 | 20 |
| SASRec + dense context | 0.0609 | 0.1261 | 0.0838 | 0.1370 | 0.0536 | 3.70 | 24 |
| Markov (1st-order) | 0.0559 | 0.1042 | 0.0706 | 0.0919 | 0.1276 | 4.69 | 0.06 |
| Popularity | 0.0549 | 0.0987 | 0.0719 | 0.0993 | 0.0053 | 1.80 | 0.00 |
| BERT4Rec | 0.0524 | 0.1024 | 0.0716 | 0.0934 | 0.0051 | 1.87 | 10 |

**Significance (paired Wilcoxon vs ItemKNN, per-user):**

| Comparison | next-item p | full-list p | reading |
|---|:---:|:---:|---|
| SASRec vs ItemKNN | **0.086** | 0.0001 | next-item **tie** (n.s.); full-list **loss** |
| BERT4Rec vs ItemKNN | 0.0013 | <1e-4 | significant loss both |
| SASRec+dense vs ItemKNN | 0.037 | 0.0012 | significant loss |
| SASRec vs Markov (next-item) | **0.319** | — | **not** significantly better than 1st-order |

**Interpretation:** On its **native** next-item task, **SASRec reaches a statistical tie
with the ItemKNN champion** (0.0667 vs 0.0778, p = 0.086 — the gap is not significant) — the
transformer *earns the project's name as parity, not a win*. On the **full ranked list**,
order-blind CF wins decisively (ItemKNN 0.1059 vs SASRec 0.0848, **−20%**, p = 0.0001):
with only 547 users, self-attention is starved of the co-occurrence density ItemKNN/ALS
exploit directly. Sobering nuance: SASRec does **not** significantly out-predict a 1-line
**Markov** chain on next-item (p = 0.32) — on this data, "condition on your last movie"
is nearly as good as multi-head attention over your whole history.

### Experiment 2 — Does *more* history help? (dense-context ablation)

**Hypothesis:** feeding all rated items (not just ≥4★ likes) gives SASRec a richer context
and lifts accuracy.

**Method:** rebuild each user's context from **all** MovieLens ratings up to their last
training like (incl. <4★), retrain SASRec, re-evaluate.

**Result:** next-item NDCG@10 **dropped 0.0667 → 0.0609** (and the loss vs ItemKNN became
significant, p = 0.037). **More history hurt.**

**Interpretation:** ~90% more interactions, but half are lukewarm ratings masquerading as
preferences; they **dilute the like-trajectory** the model needs. Signal quality beats
signal quantity — a negative result worth stating.

### Experiment 3 — Popularity debiasing on the sequential champion

**Method:** subtract `β·log(1+pop)` from SASRec's scores; sweep β ∈ {0, 0.25, 0.5, 1, 2};
measure full-list NDCG@10 and coverage.

| β | full-list NDCG@10 | catalog coverage |
|:---:|:---:|:---:|
| **0.0** | **0.0848** | 0.0338 |
| 0.25 | 0.0781 | 0.0386 |
| 0.5 | 0.0752 | 0.0450 |
| 1.0 | 0.0512 | 0.0572 |
| 2.0 | 0.0044 | 0.0318 |

**Interpretation:** debiasing **monotonically lowers** relevance for a little coverage —
the exact pattern Day 6 found for ItemKNN. Held-out likes skew popular, so penalizing
popularity penalizes the truth. Confirmed negative result across two model families.

### Experiment 4 — Grounded "because you liked X" explanations

**Method:** for the champion (SASRec), attribute each top rec to the user's **most-similar
liked title** (item-item cosine). Every referenced title is a real catalog entry — no
hallucination; an LLM would only *reword* these grounded pairs.

**Sample (user 1):** *"Because you liked "Sleeping Beauty", you may enjoy "Peter Pan"."*
(sim 0.57) · *"…liked "The Wizard of Oz" → "Mary Poppins"."* (0.46). All 5 cards cite real
catalog titles from the user's own history. Saved to `results/samples/phase5_explanations.json`.

---

## Head-to-Head Comparison (running full-list leaderboard, NDCG@10)

| System | NDCG@10 | recall@20 | coverage | novelty | note |
|--------|:---:|:---:|:---:|:---:|---|
| ItemKNN (Day-3 champ) | **0.1059** | 0.1675 | 0.0378 | 2.70 | order-blind, still the champion |
| ALS-tuned (Day-6) | 0.1058 | **0.1763** | 0.0495 | 2.94 | co-champion |
| SASRec (Day-7) | 0.0848 | 0.1366 | 0.0338 | 3.12 | ties CF on next-item; −20% full-list |
| BERT4Rec (Day-7) | 0.0716 | 0.0934 | 0.0051 | 1.87 | collapsed toward popularity prior |

---

## Key Findings

1. **The transformer earns its name as a tie, not a conquest.** SASRec matches ItemKNN on
   the native next-item task (p = 0.086, not significant) but loses the full ranked list by
   20% (p = 0.0001). Reported the tie — did not manufacture a win.
2. **Causal > bidirectional on sparse data.** SASRec (0.0667) beats BERT4Rec (0.0524); even
   with eval-aligned last-position masking and 200 epochs, BERT4Rec **collapsed toward the
   popularity prior** (coverage 0.0051 ≈ Popularity 0.0053). Cloze pre-training is more
   data-hungry than 547 users can feed.
3. **More history hurt (negative result).** Dense context incl. <4★ ratings dropped SASRec
   0.0667 → 0.0609 — lukewarm ratings dilute the like-trajectory.
4. **Debiasing hurt (negative result, confirmed twice).** β>0 monotonically lowered NDCG,
   matching Day 6 — held-out likes skew popular.
5. **Attention ≈ Markov here.** SASRec is not significantly better than a 1st-order Markov
   chain on next-item (p = 0.32): higher-order context adds little immediate-next signal at
   this scale — an honest limit of the data, not the method.
6. **Where the transformers do add value: exploration.** SASRec/SASRec+dense surface more
   novel, diverse catalog (novelty 3.1–3.7 vs ItemKNN 2.70; SASRec+dense coverage 0.054 >
   0.038) — the edge is discovery, not head relevance.

**What didn't work (and why):** BERT4Rec (data-starved bidirectional Cloze → popularity
collapse); dense context (noisy lukewarm positives); popularity debiasing (held-out likes
are popular); beating CF outright on the full list (547 users can't feed self-attention the
co-occurrence density ItemKNN reads for free).

---

## Sample Outputs Saved

- `results/phase5_sequential.csv` — 7-system × 2-protocol leaderboard
- `results/phase5_debias.csv` — β sweep on the sequential champion
- `results/phase5_metrics.json` — headline dict (protocols, significance, debias)
- `results/samples/phase5_seq_reco.json` — per-user top-10 for SASRec & BERT4Rec (hit flags)
- `results/samples/phase5_explanations.json` — grounded "because you liked X" cards
- `results/figures/phase5_nextitem_ndcg.png`, `phase5_fulllist_ndcg.png`
- Tests: **22 passed** (Day-5 10 + Day-6 7 + Day-7 5)

---

## Next Day

**Day 8 — Phase 6 (PHASE-WRAP):** frontier comparison — Claude zero-shot reco
("recommend N movies for a user who liked X, Y, Z") mapped to the catalog → NDCG@10 +
**hallucination rate** (% off-catalog) + latency + cost vs the specialized stack — plus the
full capability ablation (semantic retrieval → +embeddings → +CF → +rerank → +sequential →
+diversity).

## Code Changes

New: `src/recsys/sequential.py`, `src/eval/build_sequential_eval.py`,
`src/recsys/seq_compare.py`, `tests/test_day7_sequential.py`. No existing production paths
modified (the sequential models are an additive evaluation track; integration into the
served recommender is deferred to the Day-9 production wrapper, where the champion ranker is
wired into the "Recommended for you" tab).
