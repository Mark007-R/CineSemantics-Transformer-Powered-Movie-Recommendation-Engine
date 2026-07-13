# CineSemantics Production Upgrade — Day 9 / 10

**Phase 7: Production wrapper — FastAPI + Redis cache + implicit-feedback logging + live offline-metrics panel + a "Recommended for you" Streamlit tab**
**Date:** 2026-07-13

---

## Resume gap progress

**Gap:** Days 1–8 turned a "recommendation engine" with **zero evaluation** into a
measured, benchmarked stack (eval harness → embeddings → CF champion → rerank →
sequential transformers → frontier ablation). But it was all **offline scripts**. A
recruiter's next question about any ML system is: *"is it actually served, cached, and
does it capture the feedback you'd need to improve it?"* Through Day 8 the answer was no —
the CF champion wasn't even reachable through the app (the `/recommend` endpoint had been
silently returning 503 since Day 5 due to a broken model artifact).

**Today's contribution:** a real production wrapper. (1) A **Redis-backed cache** for hot
recommendations with a transparent in-process fallback (runs with *or* without Docker).
(2) An **implicit-feedback log** (impressions / clicks / adds) — the ground truth a future
*online* eval needs, closing the loop the offline harness opened. (3) A **live
offline-metrics panel** sourced from the sprint's own `results/`, surfaced on both
`GET /metrics` and a new Streamlit tab. (4) A **"Recommended for you" tab** backed by the
Day-3 ItemKNN champion — the name's long-promised feature, finally in the UI. (5) **Docker
Compose** extended with `redis` + `api` services alongside Milvus. (6) Per-request
**telemetry**. And a genuine bug fix: **`/recommend` now actually works.**

---

## Files touched

| File | Change |
|------|--------|
| `src/serving/cache.py` | **new** — `RecoCache`: Redis + in-process LRU fallback, TTL, hit/miss stats |
| `src/serving/feedback.py` | **new** — `FeedbackLog`: implicit events → Redis list / JSONL fallback; CTR aggregation |
| `src/serving/offline_metrics.py` | **new** — live panel from `results/` (champion, leaderboard, Day-7/8 headlines) |
| `src/serving/telemetry.py` | **new** — per-endpoint p50/p95 latency |
| `src/serving/verify_serving.py` | **new** — boots the app, measures the cache benefit, writes evidence |
| `api.py` | +`/feedback`, `/metrics`, `/telemetry`; cache on `/recommend`; timing middleware; **self-healing recommender loader** (v0.5.0 → 0.7.0) |
| `src/recsys/recommender.py` | `save()` now persists `item_universe` **and** display metadata in one `cf_meta.json` (the artifact bug that 503'd `/recommend`) |
| `pages/recommend_tab.py` | **new** — "Recommended for you" view (testable pure logic + `st` render) |
| `pages/movieflix.py`, `pages/helpers.py` | add the 6th **"✨ For You"** tab (order-independent `with tab6:` block) |
| `Dockerfile.api`, `requirements-api.txt` | **new** — code-only API image; artifacts bind-mounted |
| `docker-compose.yml` | +`redis` (7-alpine, LRU) + `api` services, health-gated `depends_on` |
| `tests/test_day9_serving.py` | **new** — 10 tests (cache, feedback, telemetry, panel, tab logic, API e2e) |
| `results/phase7_serving.json`, `results/figures/phase7_cache_latency.png` | measured serving evidence |

---

## Setup

- **Compute:** CPU. API boots in-process via FastAPI `TestClient` (loads e5-base catalog
  vectors + faiss HNSW + ItemKNN champion). Redis optional — the cache and feedback log
  fall back to an in-process LRU / JSONL file, so the whole wrapper is verifiable in a
  plain venv with no containers running.
- **Env fix:** the environment had `starlette 1.3.1`, incompatible with `fastapi 0.115.6`
  (which requires `starlette < 0.42`; the newer release removed `Router(on_startup=…)`,
  so **bare `FastAPI()` wouldn't boot**). Pinned to `starlette 0.41.3` and documented the
  constraint in `requirements-api.txt`.

---

## Experiments / verification (measured, not asserted)

### 1 — All production endpoints serve end to end

**Method:** `verify_serving.py` boots the app and exercises every route.

| Endpoint | Result |
|---|---|
| `GET /health` | ok — catalog 9,837, **recommender_loaded = true** (was false/503 before) |
| `POST /recommend` | 200 — e.g. liked *Toy Story + Lion King* → *Toy Story 2, Jurassic Park, Forrest Gump, Aladdin, Beauty and the Beast* |
| `POST /feedback` | 200 on valid events; **422** on invalid event |
| `GET /metrics` | offline panel + cache stats + feedback counts |
| `GET /telemetry` | per-endpoint p50/p95 (measured live) |

### 2 — Cache behavior on hot requests

**Hypothesis:** memoizing repeated liked-set requests cuts recompute.

**Method:** 3 distinct liked-sets, each called once (miss) then 5× (repeat). Measured p50.

| Metric | Value |
|---|---|
| Cache hit rate (15 hits / 18 calls) | **0.833** |
| `/recommend` p50 uncached | 3.92 ms |
| `/recommend` p50 cached | 3.69 ms |
| Cache speedup | **1.1×** |
| `/recommend` telemetry p50 / p95 | 2.95 ms / 4.37 ms |

**Interpretation (honest):** the cache correctly serves 83% of repeat traffic from memory,
but the end-to-end **latency win is only 1.1×** — because the ItemKNN champion is *already*
sub-millisecond and each served response also writes impression events (I/O that both
cached and uncached paths pay). **The cache's value here is offloading compute under
concurrency and future-proofing for expensive rankers** (the Day-7 transformers, or an
LLM reranker), **not** shaving milliseconds off an already-cheap kNN. Reporting the modest
number rather than a staged "100× faster" demo.

### 3 — Implicit feedback captured for future online eval

`FeedbackLog` recorded impressions on every served item plus clicks/adds, aggregating a
running CTR. This is the first time the app **records what users do** — the prerequisite
for any online A/B or bandit re-ranking (logged, but not yet acted upon).

| event | count (cumulative demo log) |
|---|---|
| impression | 210 |
| click | 3 |
| add_favorite / add_watchlist | 1 / 1 |

---

## Head-to-Head (what the wrapper adds vs Day 8)

| Capability | Day 8 (offline) | Day 9 (served) |
|---|---|---|
| CF champion reachable via API | ❌ (503 — broken artifact) | ✅ self-healing loader |
| Repeated-request caching | none | Redis + LRU, 83% hit rate |
| Feedback capture | none | impression/click/add logged |
| Live eval visibility | scripts only | `/metrics` + Streamlit panel |
| Container topology | Milvus only | + Redis + FastAPI (compose) |
| Per-request latency | unmeasured | p50/p95 telemetry |

---

## Key Findings

1. **The champion wasn't actually served.** `models/cf_meta.json` had been written in a
   display-only format with no `item_universe`, so `ItemKNNRecommender.load()` threw and
   `/recommend` had silently 503'd since Day 5. Fixed `save()` to persist both, and added a
   **self-healing loader** that refits from `cf_split.json` if the artifact is legacy — the
   kind of production-hardening a wrapper phase is for.
2. **A cache is only worth its latency if the operation is expensive.** Measured 1.1×, not a
   fabricated 100×. The honest value is 83% compute-offload + headroom for costly rankers.
3. **Serving is where offline eval becomes a loop.** The feedback log turns the Day-1 "zero
   evaluation" story into a system that can *keep* evaluating itself online.
4. **Graceful degradation is a feature.** Cache and feedback both fall back cleanly without
   Redis, so the app runs identically in a laptop venv and a Docker deployment.

**What didn't work / limits:** the cache latency win is marginal for ItemKNN (stated, not
hidden); online metrics (real CTR/NDCG) require live traffic we don't have — only the
*capture* is built; Milvus/Redis containers weren't spun up in the autonomous env, so the
compose topology is validated via `docker compose config` (VALID) rather than a live `up`.

---

## Sample Outputs Saved

- `results/phase7_serving.json` — measured endpoint checks, cache stats, feedback, telemetry
- `results/figures/phase7_cache_latency.png` — uncached vs cached `/recommend` latency
- Tests: **29 passed, 3 skipped** (Day-5 10 + Day-6 7 + Day-7 seq-tests + **Day-9 10**)

---

## Next Day

**Day 10 — Phase 8 (PHASE-WRAP + PROJECT COMPLETE):** 40+ pytest tests across the stack,
README rewrite as a mini research report (lead with the honest "zero evaluation before"
story + full IR/reco metrics tables), `docs/MODEL_CARD.md`, and a 60-second demo video.
**Project-complete post.**

## Code Changes

New serving package `src/serving/` (cache, feedback, offline_metrics, telemetry, verify);
new `pages/recommend_tab.py` + 6th Streamlit tab; `Dockerfile.api` + `requirements-api.txt`;
Redis + API services in `docker-compose.yml`. Modified `api.py` (new ops routes + caching +
self-healing loader) and `src/recsys/recommender.py` (`save()` artifact fix). No Day-1–8
result was altered; the recommender signature is unchanged (backward compatible).
