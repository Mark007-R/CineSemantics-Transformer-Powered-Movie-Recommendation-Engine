# CineSemantics Production Upgrade — Day 08 / 10

**Phase 6 — Frontier-model comparison + capability ablation**
**Date:** 2026-07-12 · Field: Recommendation Systems / Multimodal Retrieval
**Status:** POST · **PHASE-WRAP**

---

## Resume gap progress

**Gap:** the project is named a *"Transformer-Powered Movie Recommendation Engine"* but through Day 1 it was semantic search + a substring genre filter with **zero personalization and zero offline evaluation**. Days 2–7 built a real eval harness, a benchmarked embedding stack, collaborative filtering, and sequential transformers. Two questions remained open for a hiring manager:

1. **Does any of the added machinery actually matter, or is it decoration?** → the ablation isolates the contribution of each capability on one consistent held-out split.
2. **Would a frontier LLM just do this better for free?** → a head-to-head against Claude zero-shot, measuring not just ranking quality but *catalog grounding* — the failure mode an LLM has and a retrieval system cannot.

**Today's contribution:** a self-reproducing capability ablation (`results/ablation.csv`) and an honest frontier comparison (`results/frontier_comparison.csv`) that quantifies exactly where a specialized grounded recommender beats a frontier LLM and where it does not.

---

## Files touched

| File | Lines | Change |
|------|-------|--------|
| `src/eval/ablation.py` | new (≈270) | 6-rung capability ablation on the Day-3 temporal split; reuses cached MiniLM/e5 embeddings + `build_itemknn`; recomputes fast rungs fresh, carries ALS-tuned/BERT4Rec from saved day metrics |
| `src/recsys/frontier_seeds.py` | new (≈115) | deterministic 30-user sample; emits seed titles (LLM prompt), ItemKNN champion recos, held-out ground truth |
| `src/recsys/frontier_compare.py` | new (≈235) | title→catalog grounding (normalized, ±1yr), scores LLM vs ItemKNN vs Popularity, hallucination rate, latency/cost |
| `results/ablation.csv`, `results/frontier_comparison.csv`, `results/frontier_per_user.csv`, `results/phase6_metrics.json` | new | metrics |
| `results/figures/day8_ablation.png`, `results/figures/day8_frontier.png` | new | charts |
| `results/samples/frontier_seeds.json`, `frontier_llm_recos.json`, `frontier_llm_scored.json` | new | sample outputs |
| `results/leaderboard.csv` | +3 rows | Day-8 frontier rows appended to the running leaderboard |

---

## Setup

- **Compute:** CPU. Ablation runs in ~25s (cached embeddings + 0.04s ItemKNN fit). Frontier scoring <5s.
- **Eval:** the Day-3 per-user **temporal leave-last-20%** split (`data/eval/cf_split.json`) — no test interaction leaks into training (Rule 10). Ablation over all **547** eval users; frontier over a deterministic **30-user** sample (seed 42, users with ≥8 train + ≥2 test likes).
- **Data:** public MovieLens ml-latest-small interactions aligned to the 9,837-movie TMDB catalog (`data/9000plus.csv`). No proprietary data (Media Discipline).
- **Frontier model:** **Claude (this session)**, zero-shot, titles-only prompt, **no catalog access** — a genuine frontier-model output. **GPT-5.4 was not run** (no OpenAI key in the autonomous environment); only executed work is reported.

---

## Experiment 1 — Capability ablation (what actually matters?)

**Hypothesis:** most of the Day 2–7 machinery is marginal; the one capability that separates a "recommendation engine" from "semantic search" is the **interaction signal (CF)**.

**Method:** rebuild the recommender one capability at a time; score each rung with identical metric code over the same 547-user split and the same candidate universe (2,607 items).

| Rung | System | Capability added | NDCG@10 | Recall@20 | Coverage | Source |
|-----:|--------|------------------|--------:|----------:|---------:|--------|
| 1 | semantic (MiniLM centroid) | content retrieval (shipped encoder) | 0.0123 | 0.0243 | 0.148 | fresh |
| 2 | +better_embeddings (e5-base-v2) | stronger content encoder | 0.0196 | 0.0399 | 0.195 | fresh |
| 3 | **+CF (ItemKNN)** | **interaction signal / personalization** | **0.1059** | **0.1675** | 0.142 | fresh |
| 4 | +tuning (ALS Optuna) | tuned matrix factorization | 0.1058 | 0.1763 | 0.050 | carried:Day6 |
| 5 | +sequential (BERT4Rec) | sequential transformer | 0.0933 | 0.1688 | 0.064 | carried:Day7 |
| 6 | +diversity (ItemKNN+MMR) | MMR genre re-rank | 0.1017 | 0.1659 | 0.139 | fresh |

**Interpretation.** A better embedding roughly *doubles* content-retrieval NDCG (0.012 → 0.020) — real, but still an order of magnitude below what's needed. **The interaction signal is the whole game: rung 3 is a ~5.4× jump over the best content-only rung.** Everything after CF is a *trade, not a lift* — ALS tuning shifts recall/coverage without raising NDCG; the sequential transformer (the literal "transformer-powered" model) wins the *next-item* task on Day 7 but does not top ItemKNN on the full-list protocol; MMR buys diversity/coverage at a small NDCG cost. This is the honest ordering a resume line should reflect: *"personalization, not embedding choice, is what turned semantic search into a recommender."*

---

## Experiment 2 — Frontier comparison (would an LLM just win?)

**Hypothesis:** a frontier LLM produces fluent recommendations but (a) cannot be trusted to return only real catalog titles and (b) is orders of magnitude slower/costlier — so grounding, latency, and cost, not raw ranking, are where the specialized system wins.

**Method:** for 30 held-out users, give Claude only the user's 10 most-recent liked titles and ask for 12 recommendations (zero-shot, no catalog). Map its free-text titles back to the catalog (normalized title, ±1yr); unmappable titles are **off-catalog**. Score all systems on the identical held-out likes.

| System | Grounded | NDCG@10 | Recall@20 | Prec@10 | Off-catalog % | Latency | Cost/query |
|--------|:--------:|--------:|----------:|--------:|--------------:|--------:|-----------:|
| specialized: **ItemKNN champion** | yes | 0.0889 | 0.1040 | 0.0633 | **0.0%** | **0.40 ms** (measured) | **$0.00** |
| reference: Popularity | yes | 0.0488 | 0.0735 | 0.0200 | 0.0% | 0.01 ms | $0.00 |
| frontier: **Claude zero-shot** | no | 0.0905 | 0.1439 | 0.0600 | **3.9%** | ~2,100 ms (est.) | ~$0.023 (est.) |

**Interpretation — the honest, counterintuitive result.** On this popular-title sample the frontier LLM is **competitive on ranking** (NDCG 0.0905 vs 0.0889; it even leads on recall@20) — MovieLens users rate canonical films the LLM knows well, so raw fluency carries it. But it pays for that with the failure mode a grounded recommender *cannot have*:

- **Grounding:** 3.9% of its picks (14/360) are unservable. Breakdown: **12 are genuine catalog absences** (Amistad, Rushmore, Clerks, Garden State, Do the Right Thing, Time Bandits, …) — films the LLM confidently recommends that simply are not in the 9,837-movie catalog; **2 are title-variant drift** ("Star Wars: Episode IV — A New Hope" vs the catalog's "Star Wars") recoverable with fuzzy matching. The specialized system is 100% servable by construction.
- **Latency:** ~2.1 s vs 0.40 ms → **~5,000× slower**.
- **Cost:** ~$0.023/query vs $0 — at 1 M queries/day that is ~$23k/day the retrieval system does not spend.
- **Personalization ceiling:** the LLM only sees 10 seed titles as text — it has no access to the full interaction history, no cold-start-via-CF, and no diversity/coverage control. Its 3.9% grounding tax would climb sharply on a **long-tail or niche catalog**, exactly where a consumer recommender earns its keep.

**Where the LLM genuinely wins:** cold-start with no history, and natural-language intent ("something funny but not slapstick, under 2h") — kept as the Day-9/Day-10 hybrid framing, not as a replacement for the grounded ranker.

---

## Head-to-Head — running leaderboard (temporal held-out)

| Day | System | NDCG@10 | Note |
|----:|--------|--------:|------|
| 1 | Semantic (MiniLM, substring genre filter) | 0.0295* | *content-retrieval eval; no personalization measurable |
| 3 | ItemKNN (CF champion) | 0.1059 | +personalization — the ~5× jump |
| 6 | ALS Optuna-tuned | 0.1058 | tuned MF |
| 7 | BERT4Rec (sequential transformer) | 0.0933 fl / **wins next-item** | first real transformer recommender |
| **8** | **ItemKNN vs Claude zero-shot** | **0.0889 vs 0.0905** (n=30) | LLM competitive on ranking; **3.9% off-catalog, ~5000× slower, $0.023/q** |

---

## Frontier Model Comparison table (Day 8)

See Experiment 2. One-line summary: **grounded 0% vs 3.9% off-catalog · 0.40ms vs ~2,100ms · $0 vs ~$0.023/query · ranking within noise.** The resume claim is *grounding + latency + cost + personalization headroom*, not "we beat the LLM on NDCG."

---

## Key findings

1. **CF is the only capability that moves the needle** (0.020 → 0.106, ~5×). Embedding upgrades help content retrieval measurably but cannot personalize.
2. **A frontier LLM matches the specialized ranker on popular seeds** but is **unservable for ~1 in 25 recommendations** and ~5,000× slower — the tax the grounded system is built to avoid.
3. **Hallucination is concrete, not hypothetical:** 12 genuinely-absent titles across 30 users, itemized in `frontier_llm_scored.json`.
4. **Honest scoping matters:** GPT-5.4 named in the spec was not runnable here (no key) and is reported as not-run rather than fabricated; LLM latency/cost are labelled estimates while specialized latency is measured.

### What didn't work / caveats
- The 30-user sample makes ItemKNN's NDCG (0.0889) noisier than the full-split 0.1059 — stated explicitly; the *grounding/latency/cost* gaps are structural, not sample-dependent.
- "Off-catalog %" measures **servability** (title must resolve to a catalog item); 2/14 misses are title drift a better resolver would recover — reported transparently rather than hidden.

---

## Sample outputs saved

- `results/ablation.csv`, `results/frontier_comparison.csv`, `results/frontier_per_user.csv`
- `results/phase6_metrics.json`
- `results/figures/day8_ablation.png`, `results/figures/day8_frontier.png`
- `results/samples/frontier_seeds.json`, `frontier_llm_recos.json`, `frontier_llm_scored.json` (per-user seeds, raw LLM recos, itemized off-catalog titles)

---

## Phase wrap-up: What was finalized (PHASE-WRAP — Day 8)

**Final approach.** The upgrade is now fully *measured*: a benchmarked content encoder (e5-base-v2, Day 2) feeds catalog-grounded retrieval; **ItemKNN collaborative filtering is the production ranker** (Day 3/5, +MMR diversity option from Day 6); BERT4Rec is the sequential option that wins next-item (Day 7). Day 8 closes the evaluation story by proving (a) *which* of these capabilities carries the result and (b) that a frontier LLM does not obsolete the grounded system.

**Final metrics (temporal held-out).**
- Content retrieval: MiniLM 0.0123 → e5-base-v2 0.0196 NDCG@10.
- Personalization: **ItemKNN NDCG@10 = 0.1059** (~5× over best content-only; ~1.5× over popularity).
- Frontier: grounded **0% vs 3.9%** off-catalog; **0.40ms vs ~2,100ms**; **$0 vs ~$0.023/query**; ranking within noise on popular seeds.

**What carries forward (Day 9 productionization).** Champion = ItemKNN + content cold-start fallback + MMR diversity toggle. Serve behind FastAPI + Milvus + Redis; add a "Recommended for you" tab with a live offline-metrics panel; log implicit feedback. The frontier finding frames the LLM as a **cold-start / NL-intent hybrid layer on top of** the grounded ranker, never a replacement.

**Resume gap progress.** From "recommendation engine with zero evaluation" → a system with a rigorous offline harness, an isolated per-capability ablation, and an honest frontier benchmark that names exactly where specialization wins (grounding, latency, cost, personalization headroom) and where it doesn't (raw ranking on popular seeds, cold-start, NL intent). This is the differentiated claim (multimodal retrieval + evaluation rigor on a 9.8K catalog), not a MatchMind CF rebuild.

---

## Next day (Day 9 — Phase 7: Production wrapper)

Docker Compose (FastAPI + Milvus + Redis, cache embeddings + hot recs), a "Recommended for you" Streamlit tab backed by the ItemKNN champion with a live offline-metrics panel, and implicit-feedback logging (clicks/adds) for future online eval.

## Code changes
New: `src/eval/ablation.py`, `src/recsys/frontier_seeds.py`, `src/recsys/frontier_compare.py`. No production ranker signatures changed; all additions are evaluation/analysis code building on the Day-3 harness.
