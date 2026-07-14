# CineSemantics Production Upgrade — Day 10 / 10

**Phase 8: Tests + README + Model Card + demo — PROJECT COMPLETE**
**Date:** 2026-07-14

---

## Resume gap progress

**Gap:** By Day 9 the project was a measured, benchmarked, *served* recommendation stack —
but a stack a recruiter can only trust if it is **locked by tests, explained in a README
that reads like a report, and demoable in an interview**. Two specific credibility risks
remained: (1) the sprint's core corrections (no train/test leakage; the substring
genre-filter bug) had **no regression tests**, so a future refactor could silently
reintroduce them; (2) the repo's README still described the *old* "semantic search + Milvus"
product and made ML claims with **no numbers behind them**.

**Today's contribution — the closing wrap.** (1) **70 new tests** across six files
(`test_retrieval`, `test_recsys`, `test_rerank`, `test_sequential`, `test_eval_metrics`,
`test_api`) — repo now **99 passing / 3 skipped**. The two headline corrections are now
**regression-locked**: `test_recsys.py` asserts the held-out CF split has zero per-user
train/test overlap (Hard Rule 10), and `test_rerank.py` fails if the substring genre filter
ever returns (`"Sci"` must not match `"Science Fiction"`). (2) A **README rewritten as a mini
research report**, leading with the honest "it had zero evaluation" story and carrying the
full IR/recsys leaderboards, ablation, and frontier comparison. (3) **`docs/MODEL_CARD.md`**
for the champion ItemKNN ranker (+ retrieval encoder + sequential models), with data,
metrics, cold-start behavior, and limitations. (4) A **60-second live demo** (`scripts/demo.py`,
runs in ~1s off cached artifacts) + `docs/DEMO.md` narration.

---

## Files touched

| File | Change | Lines |
|------|--------|-------|
| `tests/test_eval_metrics.py` | **new** — 16 tests pinning dcg/ndcg/recall/precision/AP + text helpers | 1–120 |
| `tests/test_retrieval.py` | **new** — 12 tests: faiss HNSW search, metadata filters, similar, title lookup, length guard | 1–140 |
| `tests/test_recsys.py` | **new** — 10 tests: ItemKNN ranking, exclude-seen, cold-start, MMR, save/load, **leakage regression** | 1–130 |
| `tests/test_rerank.py` | **new** — 15 tests: token-set genre match (**substring-bug lock**), passes_filters, reranker | 1–130 |
| `tests/test_sequential.py` | **new** — 7 tests: Markov order/backoff, SASRec/BERT4Rec smoke + determinism | 1–110 |
| `tests/test_api.py` | **new** — 10 tests: /health /search /similar /recommend golden + edge (422/404/400, no hallucination) | 1–120 |
| `README.md` | **rewrite** — mini research report: zero-eval story + 7 result tables + architecture + quickstart | 1–230 |
| `docs/MODEL_CARD.md` | **new** — champion ItemKNN + encoder + sequential models; data/metrics/limits | 1–120 |
| `docs/DEMO.md` | **new** — 60-second narration + run commands | 1–60 |
| `scripts/demo.py` | **new** — runnable 4-act live demo (cached artifacts, ~1s, no Milvus/uvicorn) | 1–90 |

---

## Setup

- **Compute:** CPU only. Full test suite runs in ~94s (the API module loads cached
  e5-base-v2 vectors + faiss HNSW + the ItemKNN artifact via TestClient lifespan; the
  other five files use tiny synthetic fixtures and need no model).
- **Data/artifacts:** existing `data/9000plus.csv`, `data/eval/cf_split.json` +
  `cf_manifest.json`, `results/emb_cache/*` (cached vectors), `models/cf_*` (persisted CF).
- **Frameworks:** pytest, FastAPI TestClient, faiss, torch (sequential smoke).

---

## Experiments (verification runs)

### Experiment 1 — full regression suite

**Hypothesis:** the whole upgraded stack is exercised end-to-end and every sprint
correction is locked.

**Method:** `python -m pytest tests/ -q` on system Python 3.11 (the environment the
prior nine days used).

| # | Test group | Count | Result |
|---|-----------|-------|--------|
| 1 | `test_eval_metrics.py` — IR metric functions | 16 | pass |
| 2 | `test_retrieval.py` — HNSW search + metadata filter | 12 | pass |
| 3 | `test_recsys.py` — CF ranker + **leakage regression** | 10 | pass |
| 4 | `test_rerank.py` — **substring-filter lock** + reranker | 15 | pass |
| 5 | `test_sequential.py` — Markov + SASRec/BERT4Rec smoke | 7 | pass |
| 6 | `test_api.py` — FastAPI golden + edge cases | 10 | pass |
| — | pre-existing (Day 5/6/7/9) | 32 | 29 pass / 3 skip* |
| **Total** | | **102** | **99 pass / 3 skip** |

\*the 3 skips are pre-existing Day-7 tests that need `seq_split.json` (an unbuilt eval
artifact), not failures.

**Interpretation:** the two corrections that define this sprint's credibility are now
regression-locked — a refactor that reintroduces train/test leakage or the substring genre
filter turns the suite red.

### Experiment 2 — live demo (interview reproducibility)

**Hypothesis:** the whole story is reproducible in one command, fast enough to run live.

**Method:** `python scripts/demo.py`.

| Act | Output | Result |
|-----|--------|--------|
| 1 The gap | states no-recommender / no-eval | ok |
| 2 The number | prints baseline leaderboard live | Semantic 0.0295 **loses 7.3× to Popularity 0.2148** |
| 3 The fix | prints CF bake-off | ItemKNN **0.1059 (+47%)** |
| 4 Grounded | 5 live personalized recs | 100% catalog-valid (Jurassic Park, Beauty and the Beast, Toy Story 2, …) |

Total runtime **~1.2s**.

**Interpretation:** the honest narrative — *semantic search lost to popularity; CF closed
the gap; recs are grounded where an LLM hallucinates* — is now a repeatable artifact, not a
claim.

---

## Head-to-Head Comparison (final project leaderboard)

Personalized held-out user/time split unless noted; primary metric NDCG@10.

| System | NDCG@10 | recall@20 | Note |
|--------|--------:|----------:|------|
| **ItemKNN (production champion)** | **0.1059** | 0.1675 | Day-3 CF |
| ALS (Optuna-tuned) | 0.1058 | 0.1763 | Day-6, ties champion |
| PureSVD | 0.0979 | 0.1608 | Day-3 |
| BERT4Rec (transformer) | 0.0933 | 0.1688 | Day-7; next-item ties ItemKNN |
| SASRec (transformer) | 0.0915 | 0.1499 | Day-7 |
| Hybrid (CF+content) | 0.0940 | 0.1546 | Day-3 |
| ALS (untuned) | 0.0861 | 0.1494 | Day-3 |
| Popularity | 0.0719 | 0.0993 | non-personalized baseline |
| Content centroid (e5-base) | 0.0196 | 0.0399 | Day-2 |
| Content centroid (MiniLM, shipped) | 0.0123 | 0.0243 | original "engine" |

Frontier (Day-8 sample): Claude zero-shot NDCG@10 **0.0905** ≈ ItemKNN 0.0889, but **3.9%
off-catalog**, ~5000× slower, ~$0.023/query.

---

## Key findings

1. **The sprint's credibility now lives in the test suite, not the prose.** The single most
   important test in the repo is five lines: per-user `train ∩ test == ∅`. Everything else —
   every NDCG number — is only trustworthy because that holds. Locking it was the right
   closing move.
2. **The substring-filter bug is subtle enough to deserve its own test.** `"Sci" in
   "science fiction"` is `True` in Python; a reviewer skimming a diff could reintroduce it
   without noticing. `test_substring_false_positive_is_gone` makes that impossible to merge.
3. **A good README for an ML project is a results table, not a feature list.** The rewrite
   leads with the number that's *against* the project (semantic search losing to popularity)
   because that's what makes the eventual +47% CF lift believable.
4. **Genuine insight (whole project):** the "transformer-powered recommendation engine" was
   neither transformer-powered as a *ranker* nor a *recommender*. Measured honestly, its
   retrieval lost to popularity 7×; the fix that mattered was the missing collaborative
   filter (~5× ablation lift), and the actual transformers (SASRec/BERT4Rec) only *tie* CF.
   Honest evaluation — not a bigger model — was the entire story.

## What didn't work (and why)

- **A pure-neural direction never paid off.** The cross-encoder reranker (Day 4) and the
  sequential transformers (Day 7) each *underperformed or tied* the cheap CF/metadata
  methods at far higher cost. Reported as-is; no cherry-picking.
- **3 Day-7 tests remain skipped** — they depend on `seq_split.json`, a Day-7 eval artifact
  not regenerated in this autonomous env. Left as skips (honest) rather than deleted.
- **No literal video file** is produced in the autonomous run; `scripts/demo.py` +
  `docs/DEMO.md` are the runnable, recordable substitute (the demo runs live in ~1s).

## Sample outputs saved

- `scripts/demo.py` (runnable live demo) · `docs/DEMO.md` (narration)
- `README.md` (report-style, 7 result tables) · `docs/MODEL_CARD.md`
- Test suite: `tests/test_{retrieval,recsys,rerank,sequential,eval_metrics,api}.py`
- Report: `reports/day10_phase8_report.md`

---

## Phase wrap-up — what was finalized (Phase 8 / PROJECT COMPLETE)

**Final approach.** CineSemantics is now a tested, documented, served, and demoable
recommendation stack: e5-base-v2 semantic retrieval over faiss/Milvus **HNSW** →
token-set metadata filtering → metadata reranker; a **personalized ItemKNN CF layer**
with content cold-start and MMR diversity as the production ranker; SASRec/BERT4Rec
benchmarked as the sequential surface; a FastAPI service (`/search /similar /recommend
/feedback /metrics /telemetry`) with Redis cache and implicit-feedback logging; and a
Streamlit "For You" tab. All of it is measured on a leakage-free held-out split.

**Final metrics.** Production ranker **ItemKNN NDCG@10 = 0.1059 (+47% over popularity)**;
retrieval encoder upgrade **+63%** (MiniLM→e5-base); reranker **+42%** over pure cosine at
1/12 the cross-encoder latency; HNSW **3.9×** faster than IVF at equal recall; frontier LLM
ties on ranking but **hallucinates 3.9%** of titles at ~5000× the latency. Test suite:
**99 passing / 3 skipped (102 total)**.

**What carries forward (to future sprints / the resume).** The evaluation harness and the
leakage-free split protocol are reusable infrastructure. The backlog seeds a natural
follow-up: a **learned two-tower retriever** trained on interactions (vs today's off-the-shelf
encoder), **true cross-modal** text↔poster search, and an **online A/B + bandit** loop on the
implicit-feedback log this sprint started capturing.

**Resume gap progress — CLOSED.** Before: a "recommendation engine" with no recommender and
no evaluation. After: *"I took a semantic-search app that shipped as a recommendation engine,
proved with a held-out MovieLens split that it lost to a popularity baseline, added the
collaborative-filtering and sequential-transformer layers it never had, benchmarked every step
(including against a frontier LLM), and shipped it as a tested FastAPI service."* That is a
recommendation-systems + evaluation-rigor story a hiring manager can interrogate end to end.

---

## Next

**CineSemantics is COMPLETE (Day 10/10).** The sprint moves to **Project C — StockAI**
(Jul 15 – Jul 24). StockAI Day 1: audit + **scaler-leakage fix** (`predictor.py` /
`stock_predictor.py`) + honest persistence & buy-and-hold baselines.

**Post-worthy?** Yes. **Post type:** Project Complete. **Post angle:** "I finished a 10-day
rebuild of a movie 'recommendation engine' and the most valuable thing I added wasn't a
model — it was the evaluation it never had. On day one, honest offline metrics showed the
shipped semantic search *losing to a popularity baseline by 7×*. That one uncomfortable
number set the whole agenda: I added the collaborative-filtering layer the name always
promised (+47%), benchmarked SASRec/BERT4Rec (they only *tie* CF), and pitted it against a
frontier LLM (statistically a tie on ranking — but it hallucinated 3.9% of the titles, ran
5000× slower, and cost 2¢ a call). Shipped as a tested FastAPI service with a leakage-free
eval locked into the test suite. The lesson I keep relearning: for ML, honest measurement
beats a bigger model."
