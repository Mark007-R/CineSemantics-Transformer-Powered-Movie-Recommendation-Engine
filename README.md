# CineSemantics — Transformer Movie Recommendation Engine

Semantic + multimodal retrieval and personalized recommendation over a **9,826-movie
TMDB catalog** (posters included), served through a Streamlit UI and a FastAPI
inference service backed by Milvus / faiss HNSW.

---

## TL;DR — the honest version

This project shipped as a *"Transformer-Powered Movie Recommendation Engine."* When I
sat down to measure it, two things were true that the name hid:

1. **It had no recommender.** The "engine" was off-the-shelf `all-MiniLM-L6-v2`
   semantic search plus a *substring* genre filter
   (`genre_filter.lower() in movie['genre'].lower()`, `milvus_vectordb.py:354`).
   No user model, no personalization, no collaborative filtering.
2. **It had zero evaluation.** Not one precision@k, NDCG, MAP, coverage, or diversity
   number anywhere in the repo.

So the first thing I built was **the missing offline evaluation harness**, on a
held-out user/time split of public MovieLens ratings aligned to the catalog. The very
first honest number was uncomfortable:

> **Off-the-shelf semantic search scored NDCG@10 = 0.0295 on the content-relevance set —
> and *lost to a popularity baseline (0.2148) by ~7×.***

That result set the agenda. Over ten days I benchmarked the embedding stack, added a
real collaborative-filtering layer, a reranker, and the first genuine *transformer
ranker* (SASRec / BERT4Rec), and production-wrapped the whole thing — measuring every
step against a held-out split with no interaction leakage.

**The punchline:** the win didn't come from a fancier transformer. It came from
**adding the recommender the name always promised.** Collaborative filtering is the one
rung that moves the needle (~5×); the sequential transformers *tie* it on next-item
ranking but don't beat it on full-list quality. And a frontier LLM (Claude, zero-shot)
*matches* the specialized ranker on NDCG — while hallucinating catalog titles 3.9% of
the time, running ~5000× slower, and costing ~$0.023/query.

---

## Headline results

### 1 — The gap: no recommender, no evaluation (Day 1 baseline)

Content-relevance set (item→item "more like this", MovieLens co-rating, 1,072 held-out queries):

| System | NDCG@10 | recall@20 | MAP@20 | coverage | diversity |
|---|---|---|---|---|---|
| Popularity (non-personalized) | **0.2148** | 0.1858 | 0.1042 | 0.001 | 0.639 |
| Semantic search (shipped: MiniLM) | 0.0295 | 0.0109 | 0.0072 | 0.534 | 0.678 |
| Random | 0.0034 | 0.0023 | 0.0006 | 0.659 | 0.858 |

> **Insight:** a "recommendation engine" that can't beat *"show everyone the most popular
> movies"* isn't recommending — it's retrieving. The fix is not a bigger encoder; it's a
> user model.

### 2 — Better embeddings help retrieval, but only marginally (Day 2)

Content retrieval, same held-out set:

| Model | dim | NDCG@10 | recall@20 | ms/query |
|---|---|---|---|---|
| **e5-base-v2 (champion)** | 768 | **0.0482** | 0.0205 | 1.10 |
| all-mpnet-base-v2 | 768 | 0.0480 | 0.0218 | 1.01 |
| bge-base-en-v1.5 | 768 | 0.0393 | 0.0166 | 1.52 |
| MiniLM-L6-v2 (shipped) | 384 | 0.0295 | 0.0109 | 0.71 |

e5-base-v2 is **+63%** over the shipped MiniLM — real, but still far below popularity.
Retrieval quality was never the bottleneck.

### 3 — The recommender that was missing (Day 3, personalized held-out split)

Per-user temporal split, ranking over the training item universe:

| System | NDCG@10 | recall@20 | coverage | fit (s) |
|---|---|---|---|---|
| **ItemKNN (champion)** | **0.1059** | 0.1675 | 0.038 | 0.04 |
| PureSVD | 0.0979 | 0.1608 | 0.052 | 0.29 |
| Hybrid (CF + content) | 0.0940 | 0.1546 | 0.065 | 0.02 |
| ALS | 0.0861 | 0.1494 | 0.066 | 0.35 |
| Popularity | 0.0719 | 0.0993 | 0.005 | — |
| Content (MiniLM centroid) | 0.0123 | 0.0243 | 0.039 | 35.1 |

Item-item CF is **+47% over popularity** *on the personalized task* — the first evidence
of genuine personalization in the project.

### 4 — Rerank & multimodal fusion: the cheap fix beats the neural one (Day 4)

| Variant | NDCG@10 | p95 rerank latency |
|---|---|---|
| **metadata_rerank** (cosine + genre-Jaccard + popularity prior) | **0.0683** | 48.8 ms |
| cross-encoder (ms-marco-MiniLM) top-15 | 0.0503 | 604.7 ms |
| semantic only (e5-base) | 0.0482 | 2.7 ms |

The neural cross-encoder bought **+4% for 220× the latency** and was rejected. Poster
(CLIP) + text fusion lifted held-out NDCG@10 **+43%**. ANN sweep: **faiss/Milvus HNSW
matched IVF_FLAT's exact NDCG at ~3.9× lower p95 latency**, so HNSW is the production index.

### 5 — The first real transformer ranker (Day 7)

Next-item prediction (the honest "transformer-powered" task):

| System | next-item HR@10 | next-item NDCG@10 | full-list NDCG@10 |
|---|---|---|---|
| ALS (tuned) | 0.1664 | **0.0811** | 0.1058 |
| **BERT4Rec** (transformer) | 0.1627 | 0.0797 | 0.0933 |
| ItemKNN | 0.1572 | 0.0778 | **0.1059** |
| SASRec (transformer) | 0.1444 | 0.0757 | 0.0915 |
| Markov (1st-order) | 0.1042 | 0.0559 | 0.0718 |
| Popularity | 0.0987 | 0.0549 | 0.0719 |

> **Insight:** BERT4Rec *ties* ItemKNN on next-item ranking (0.0797 vs 0.0778) but does
> **not** beat it on full-list quality. Sequence order helps a little (Markov > Popularity),
> but for this catalog and interaction density, set-based CF is still the champion. The
> transformer earns the project's *name*, not a leaderboard upset — and I report that honestly.

### 6 — Frontier LLM comparison (Day 8): grounding beats fluency

Same held-out users, "recommend N movies for a user who liked X, Y, Z":

| System | grounded | NDCG@10 | off-catalog | latency | $/query |
|---|---|---|---|---|---|
| **ItemKNN (specialized)** | ✅ | 0.0889 | **0%** | **0.4 ms** | **$0** |
| Claude (zero-shot, no catalog) | ❌ | 0.0905 | **3.9%** | ~2100 ms | ~$0.023 |
| Popularity (reference) | ✅ | 0.0488 | 0% | 0.01 ms | $0 |

The LLM's ranking is statistically a tie, but it **hallucinates titles not in the catalog**,
is **~5000× slower**, and **costs money per call**. For a system that must return *real,
clickable, in-stock* items, grounding + latency + cost win.

### 7 — Capability ablation (what each rung buys)

| Rung | Capability added | NDCG@10 |
|---|---|---|
| 1. Semantic (MiniLM centroid) | content retrieval | 0.0123 |
| 2. + better embeddings (e5-base) | stronger encoder | 0.0196 |
| **3. + CF (ItemKNN)** | **personalization** | **0.1059** ⬅ ~5× jump |
| 4. + tuning (ALS Optuna) | tuned MF | 0.1058 |
| 5. + sequential (BERT4Rec) | transformer ranker | 0.0933 |
| 6. + diversity (MMR) | genre de-concentration | 0.1017 |

**One rung — collaborative filtering — accounts for essentially the entire lift.**

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   query / likes →  │  FastAPI service (api.py, port 8000)          │
                    │   /search   e5-base-v2 → faiss HNSW           │
                    │             → metadata filter → rerank        │
                    │   /similar  item-item + CLIP poster fusion    │
                    │   /recommend ItemKNN CF + content cold-start  │
                    │   /feedback /metrics /telemetry               │
                    │   Redis cache · per-request telemetry         │
                    └──────────────────────────────────────────────┘
   Streamlit UI  ───┤  Discover · Search · Visual · For-You (CF) ·  │
   (pages/)         │  live offline-metrics panel                   │
                    └──────────────────────────────────────────────┘
   Milvus (HNSW)  ── docker-compose · same algorithm as offline faiss

   src/
    ├── retrieval/   embedder (e5-base-v2) · index (HNSW) · metadata_filter (token-set)
    ├── recsys/      recommender (ItemKNN + cold-start + MMR) · sequential (SASRec/BERT4Rec)
    ├── rerank/      metadata_rerank (cosine + genre-Jaccard + pop prior) · fusion
    ├── eval/        build_eval · baseline · embedding_comparison · ablation
    └── serving/     cache · feedback · telemetry · offline_metrics
```

Model details for the production ranker are in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md);
the original audit that started the sprint is in [`docs/RECSYS_AUDIT.md`](docs/RECSYS_AUDIT.md).

---

## Quickstart

```bash
pip install -r requirements.txt          # Streamlit app deps
pip install -r requirements-api.txt      # FastAPI service deps

# 1. Streamlit UI (Discover / Search / Visual / For-You)
streamlit run pages/movieflix.py

# 2. FastAPI inference service (offline — no Milvus required; uses cached vectors)
uvicorn api:app --port 8000
#   POST /search    {"query": "space adventure with robots", "top_k": 5}
#   POST /similar   {"title": "Toy Story", "top_k": 5, "poster_fusion": true}
#   POST /recommend {"liked_titles": ["Toy Story", "The Lion King"], "top_k": 5}

# 3. Full stack (FastAPI + Milvus + Redis)
docker-compose up -d
```

### Reproduce the evaluation

```bash
python -m src.eval.build_eval            # build held-out relevance + CF split
python -m src.eval.baseline              # Day-1 honest baseline (the "0.0295 loses to popularity" number)
python -m src.recsys.cf_compare          # Day-3 CF bake-off leaderboard
python -m src.eval.ablation              # capability ablation table
```

---

## Tests

```bash
python -m pytest tests/ -q
```

99 tests covering retrieval, the CF recommender, reranking, the sequential
transformers, the metric functions, and the FastAPI service end-to-end. Notable
regression locks:

- **`test_recsys.py`** — the held-out CF split has **zero train/test interaction
  leakage** (per-user disjoint), and `save/load` round-trips the load-critical
  `item_universe`.
- **`test_rerank.py`** — the **substring genre-filter bug cannot return**: `"Sci"` no
  longer matches `"Science Fiction"`, and token-set matching is enforced.
- **`test_api.py`** — golden path + edge cases (empty query → 422, out-of-catalog title →
  404, cold-start unknown likes → 400 with **no hallucinated recommendation**).

---

## Data & reproducibility

- **Catalog:** `data/9000plus.csv` (9,826 TMDB movies) + `posters/` (~9,509 images).
- **Interactions:** public **MovieLens ml-latest-small**, aligned to the catalog by
  (title, year); only public data is used.
- **Splits:** per-user temporal hold-out; the candidate universe is training-only items,
  so no test interaction ever leaks into training (checked in `cf_manifest.json`).
- Cached embeddings live in `results/emb_cache/` so the API and eval share one encode.

## License
MIT — see [LICENSE](LICENSE).
