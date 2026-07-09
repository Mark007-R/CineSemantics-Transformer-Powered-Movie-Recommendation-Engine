# CineSemantics Production Upgrade — Day 5 / 10

**Date:** 2026-07-09 · **Phase:** 3 — Champion integration + production refactor (**PHASE-WRAP**) ·
**Field:** Recommendation Systems / Multimodal Retrieval

## Resume gap progress
**Gap:** Days 1–4 measured everything but changed no production code — the winners
(e5-base-v2 embedding, HNSW index, metadata reranker, ItemKNN CF, poster fusion)
lived only in `src/eval` / `src/rerank` experiment scripts. The app still shipped
the off-the-shelf MiniLM encoder, the IVF_FLAT default, the **substring genre
filter**, and — critically — **no recommendation layer at all**, despite the name.

**Today's contribution:** Phase-3 wrap. Every champion is moved into a production
package (`src/retrieval`, `src/recsys`, `src/rerank`), wired into the shipped
`utils/`, and exposed through a new async FastAPI service (`api.py`). A validation
harness proves the production path **reproduces the Day-2/3/4 offline numbers
exactly** (no silent regression), and the substring→token-set genre fix is
quantified. The "recommendation engine" finally recommends.

## Files touched
**New production modules**
- `src/retrieval/embedder.py` — `ChampionEmbedder` (e5-base-v2, 768-d, "query: "
  prefix, L2-normalised, disk-cached; shared by Flask + API + eval).
- `src/retrieval/index.py` — `MovieIndex`: faiss **HNSW** (M=32, efSearch=64, the
  Day-4 champion), `search()` with proper metadata filtering + over-fetch,
  `similar()` with optional **CLIP poster fusion** (Day-4, a=0.75).
- `src/retrieval/metadata_filter.py` — token-set `genre_match` / `passes_filters`,
  the replacement for the substring filter.
- `src/rerank/metadata_rerank.py` — `MetadataReranker` (cosine + genre-Jaccard +
  popularity prior; the Day-4 champion that beat the cross-encoder).
- `src/recsys/recommender.py` — `ItemKNNRecommender` (Day-3 champion) + **content
  cold-start fallback**; save/load to `models/cf_*`.
- `src/integrate_champions.py` — the build + validation harness (this report's numbers).
- `api.py` — async FastAPI: `/health`, `/search`, `/similar`, `/recommend`, Pydantic v2.

**Modified shipped code (signatures preserved)**
- `utils/config.py` — `TEXT_MODEL_NAME` MiniLM→**e5-base-v2**, dim 384→**768**,
  new `TEXT_MODEL_PREFIX="query: "`; `INDEX_TYPE` IVF_FLAT→**HNSW** + `HNSW_M` /
  `HNSW_EF_CONSTRUCTION` / `SEARCH_EF`; `build_index_params()` / `build_search_params()`.
- `utils/text_embedder.py` — apply the e5 prefix at encode time in `embed_csv`
  (line ~108) and `embed_text` (line ~135); no-op for prefix-less models.
- `utils/milvus_vectordb.py` — **line 354 substring genre filter → `_genre_matches`
  token-set match**; `create_text_collection` / `create_image_collection` /
  `search_similar_movies` now use `config.build_index_params()` /
  `build_search_params()` (HNSW).
- `requirements.txt` — add scikit-learn, scipy, implicit, fastapi, uvicorn, pydantic.

## Setup
- **Compute:** CPU, Python 3.11.9. e5 catalog re-encode 9,837 movies (cache was
  gitignored, rebuilt); HNSW build + all validations < 30 s on the warm cache.
- **Champions integrated:** e5-base-v2 (768-d) · HNSW (M=32, efSearch=64) ·
  metadata reranker (a=0.2, b=0.05) · ItemKNN CF + content cold-start · CLIP
  poster fusion (4,083 cached poster embeddings).
- **Eval sets (unchanged from Days 1–4):** content = 1,072 MovieLens co-rating
  "more like this" queries; CF = per-user temporal split, 547 held-out users ×
  2,607 candidate items.
- **Milvus note:** `pymilvus` is not importable in this local env (a protobuf
  version clash — Milvus is a Docker service, which is why Day-4 characterised the
  ANN frontier via faiss "Docker-free"). The `utils/milvus_vectordb.py` edits are
  the config/filter changes the Docker collection will use; the offline faiss HNSW
  index gives the API the identical behaviour without a running Milvus. The
  Milvus text collection must be **re-ingested at 768-d + HNSW** in Day-7's
  production wrapper (dim changed 384→768).

## Experiment A — does the production path reproduce the leaderboard?
- **Hypothesis:** Moving the champions from experiment scripts into production
  modules should reproduce the exact Day-2/3/4 offline metrics; any drift is a
  bug in integration.
- **Method:** `src/integrate_champions.py` runs the SAME held-out sets through the
  production `MovieIndex` (exact cosine, HNSW ANN, HNSW+rerank) and the production
  `ItemKNNRecommender`, using the identical metric functions.
- **Result** (`results/phase3_integration.csv`):

| Stage (production path) | NDCG@10 | Recall@20 | p95 (ms) | Day-N reference |
|-------------------------|--------:|----------:|---------:|-----------------|
| exact cosine (reference) | 0.0482 | 0.0205 | — | Day-2 e5 **0.0482** ✓ |
| HNSW ANN (production index) | 0.0481 | 0.0205 | 1.7 | Day-4 HNSW ≈ exact ✓ |
| HNSW + metadata rerank | **0.0683** | 0.0271 | — | Day-4 rerank **0.0683** ✓ |
| ItemKNN CF (personalized) | **0.1059** | 0.1675 | — | Day-3 ItemKNN **0.1059** ✓ |

- **Interpretation:** Exact reproduction across the board — the production modules
  are faithful, not re-implemented-and-drifted. HNSW loses only 0.0001 NDCG vs
  exact at ~1.7 ms p95 (Day-4's recall/latency win now lives in the API).

## Experiment B — substring genre filter vs proper token-set filter
- **Hypothesis:** The shipped substring filter (`genre_filter.lower() in
  genre.lower()`) is brittle; a token-set matcher fixes real failure modes.
- **Method:** Replay three query classes over all 9,837 catalog genre fields with
  both filters (`results/phase3_genre_filter.csv`).
- **Result:**

| Scenario | Substring filter | Token-set filter | Verdict |
|----------|------------------|------------------|---------|
| single canonical genre ("Action") | matches | matches | **agree** — substring is fine for the exact dropdown case |
| compound AND ("Action, Adventure") | matches only if the field lists them in that exact order | matches regardless of order | substring **misses 1,877** true AND-matches |
| partial/fuzzy input ("Sci","Rom","a","Anim","Fi") | **13,450 garbage matches** | 0 | token-set correctly rejects non-genres |

- **Interpretation:** The honest finding is nuanced: for a single exact dropdown
  genre the two agree, so the substring filter was *not* silently corrupting the
  common path. Its real defects are (1) it **cannot express multi-genre AND/OR
  intent** at all — the API's `genres:["Action","Comedy"], genre_mode:"all"` query
  (→ Beverly Hills Cop, Extreme Job) is simply impossible with a single substring;
  and (2) it **false-matches partial input** (13,450 spurious hits), so any
  free-text or upstream-query path passing "Sci"/"Rom" gets garbage. The token-set
  filter fixes both and is what `search_similar_movies` and the API now use.

## Experiment C — the recommendation layer + cold-start (the missing "engine")
- **Method:** `ItemKNNRecommender` fit on the MovieLens train interactions,
  persisted to `models/cf_*`, served via `/recommend`. Cold-start = a synthetic
  new user with a single liked seed (falls to the content-centroid path).
- **Result:** personalized NDCG@10 **0.1059** (= Day-3). Cold-start with one seed
  drops to **0.0290** — a real, honest gap that justifies the content fallback.
  API grounding: **100% of recommended titles are real catalog items** by
  construction (the LLM-baseline hallucination angle is Day-8).
- **Live sample** (`/recommend`, liked = Dark Knight / Inception / Interstellar):
  → Dark Knight Rises, Shutter Island, Fight Club, The Departed, Batman Begins,
  LOTR: Return of the King, Guardians of the Galaxy, Up — all catalog-valid,
  method `itemknn`.

## Head-to-Head Comparison (running story)

| Eval | System | NDCG@10 | Note |
|------|--------|--------:|------|
| Day-1 content | Semantic (MiniLM) | 0.0295 | first-ever metric |
| Day-2 content | e5-base | 0.0482 | +63% over MiniLM |
| Day-4 content, reranked | metadata rerank | 0.0683 | +42% over pure |
| Day-3 personalized | ItemKNN | 0.1059 | first personalization |
| **Day-5 production** | **all champions, integrated** | **matches all of the above** | **shipped, not just measured** |

## Key findings
1. **The production path reproduces every offline champion exactly** — e5 0.0482,
   HNSW 0.0481, rerank 0.0683, CF 0.1059. Integration introduced no regression.
2. **The substring genre filter's real sin was expressiveness, not the common
   case.** It agrees with token-set on single exact genres, but cannot represent
   AND/OR intent and returns 13,450 garbage matches on partial input. The fix
   unlocks genuinely useful multi-genre + rating filtered search.
3. **The engine finally recommends, and it's grounded.** `/recommend` returns
   100% catalog-valid, personalized items (NDCG@10 0.1059) an LLM has no history
   to produce — the Day-8 differentiation seed.
4. **Cold-start is a measured gap, not hand-waving.** One-seed users score 0.0290;
   the content fallback is what carries them until interaction history accrues.
5. **A dim change has a production cost.** MiniLM(384)→e5(768) means the Milvus
   collection must be dropped and re-ingested at 768-d/HNSW — flagged for Day-7.

## What didn't work (and why)
- **Milvus in the local env:** `pymilvus` fails to import (protobuf clash). Not a
  Day-5 blocker — Milvus is a Docker service; the offline faiss HNSW index serves
  the API identically and reproducibly. Re-ingestion at 768-d is Day-7 work.
- **First `/search` with a genre filter returned `[]`** — a real integration bug
  caught by the smoke test: `passes_filters` expected lowercase keys but the
  catalog records use capitalised column names, so every candidate was filtered
  out. Fixed by normalising keys in `MovieIndex.search`; re-validated end-to-end.

## Sample outputs saved
- `results/phase3_integration.csv` — production-path validation vs Day-2/3/4.
- `results/phase3_genre_filter.csv` — substring vs token-set, 3 scenarios.
- `results/phase3_metrics.json` — headline dict (champions + all validations).
- `results/samples/phase3_api_samples.json` — search / similar / recommend samples.
- `results/samples/phase3_api_smoke.json` — live `/health`+4-endpoint responses.
- `results/figures/phase3_integration.png` — production reproduces the leaderboard.
- `models/cf_interactions.npz` + `cf_meta.json` — persisted CF champion.

## Phase wrap-up (Phase 3 — Champion integration + production refactor)
- **Final approach:** one production package per capability — `src/retrieval`
  (e5 + HNSW + token-set metadata filter + poster fusion), `src/rerank` (metadata
  reranker), `src/recsys` (ItemKNN + cold-start) — wired into the shipped `utils/`
  and served by an async FastAPI `api.py`. The offline faiss index makes the API
  runnable without Docker/Milvus.
- **Final metrics (all reproduced through production code):** content NDCG@10
  0.0482 (exact) / 0.0481 (HNSW, p95 1.7 ms) / **0.0683** (+ metadata rerank);
  personalized **0.1059**; cold-start 1-seed 0.0290; genre fix removes 13,450
  false-positive matches and unlocks multi-genre AND/OR + rating filters.
- **What carries forward:** Day-6 tunes ALS (Optuna) + cold-start/diversity error
  analysis on top of these production modules; Day-7 trains the SASRec/BERT4Rec
  sequential recommender against the CF champion; Day-8 runs the LLM-vs-specialized
  frontier comparison (hallucination rate) on this exact stack; Day-9 dockerizes
  `api.py` + Milvus (re-ingest at 768-d/HNSW) + Streamlit "Recommended for you".
- **Resume gap progress:** "transformer-powered recommendation engine" now has a
  real, benchmarked, **served** recommendation layer + rigorous offline eval — the
  two things the name always claimed and the audit found missing on Day 1.

## Next day
**Day 6 — Phase 4: Tuning + cold-start/error analysis.** Optuna on ALS (factors /
regularization / iterations, ≥30 trials); error-analyse 30 bad recommendations
(popularity bias / cold-start / genre over-concentration) → dominant failure →
targeted fix (popularity debiasing or MMR diversity rerank); re-evaluate on the
same held-out split.

## Code changes
New: `src/retrieval/{embedder,index,metadata_filter}.py`, `src/recsys/recommender.py`,
`src/rerank/metadata_rerank.py`, `src/integrate_champions.py`, `api.py`, package
`__init__` exports, `src/__init__.py`. Modified: `utils/config.py`,
`utils/text_embedder.py`, `utils/milvus_vectordb.py`, `requirements.txt`.

## Day-5 follow-up (same-day hardening PR)
Two integration-quality gaps found on self-review and fixed in a follow-up PR:
1. **Principled `/search` rerank.** The first cut faked a genre-Jaccard anchor from
   the top hit (arbitrary). Replaced with `MetadataReranker.rerank_query`: cosine +
   popularity prior, plus genre-Jaccard against the **requested** genres when the
   caller supplies them, else popularity-only. Verified live (genre-filtered query
   → all-Romance; free-text → popularity-reranked; `rerank:false` honoured).
2. **Integration regression lock.** `tests/test_day5_integration.py` — 10 fast
   pytest tests covering the headline genre-filter fix (incl. the substring
   false-positive/negative it repairs), config champions + param builders,
   numeric filters, the CF recommender (seen-exclusion, catalog-validity,
   cold-start fallback, save/load round-trip), the text-query reranker, and the
   HNSW item-item path on cached embeddings. **10 passed in 3.65s.** This is the
   integration lock, distinct from Day-10's full per-module suite. Also added
   `src/eval/__init__.py` (packaging).
