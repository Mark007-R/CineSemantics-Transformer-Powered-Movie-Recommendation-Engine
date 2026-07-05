# CineSemantics — Recommendation-System Audit (Day 1)

**Date:** 2026-07-05 · **Sprint:** CineSemantics Production Upgrade, Day 1 of 10 ·
**Scope:** honest assessment of what the "Transformer-Powered Movie Recommendation
Engine" actually is today, plus the first offline evaluation the repo has ever had.

---

## TL;DR

1. **It is semantic search, not a recommender.** The system embeds a query and
   returns the nearest catalog movies by cosine similarity. There is **no user
   model, no interaction history, no personalization, and no collaborative
   filtering** anywhere in the codebase.
2. **The "genre filter" is a substring match, not semantic.**
   `search_similar_movies()` post-filters hits with
   `genre_filter.lower() in movie_data['genre'].lower()`
   ([utils/milvus_vectordb.py:354](../utils/milvus_vectordb.py)) — a string
   containment test, not a learned relevance signal.
3. **The embeddings are off-the-shelf.** Text is `all-MiniLM-L6-v2` (384-d),
   posters are CLIP ViT-B/32 (512-d)
   ([utils/config.py:7-11](../utils/config.py)). The two spaces have different
   dimensions and are **not cross-modal comparable** — a text query cannot
   retrieve by poster and vice-versa.
4. **There was ZERO offline evaluation before today.** No precision@k, recall@k,
   NDCG, MAP, coverage, or diversity — anywhere. "Transformer-powered
   recommendation engine" was framing; the reality is pretrained-embedding
   semantic search + a substring filter.

The Day-1 deliverable fixes #4: a reproducible eval harness + honest first
numbers, established **before** any new feature is added (Hard Rule 11).

---

## What the code actually does

| Component | File / line | Reality |
|---|---|---|
| Text embedding | `utils/text_embedder.py:40` `load_model()` | Off-the-shelf `sentence-transformers/all-MiniLM-L6-v2`, 384-d, L2-normalized |
| Poster embedding | `utils/image_embedder.py:22` `load_clip_model()` | Off-the-shelf CLIP `openai/clip-vit-base-patch32`, 512-d |
| Retrieval | `utils/milvus_vectordb.py:249` `search_similar_movies()` | Milvus IVF_FLAT, metric `IP` on normalized vectors (= cosine = exact inner product). Embeds query → ANN search → metadata pre-filters (rating/year/popularity) |
| "Genre filter" | `utils/milvus_vectordb.py:354` | **Substring post-filter** `genre_filter.lower() in movie_data['genre'].lower()` — not semantic, not learned |
| Poster→poster | `utils/milvus_vectordb.py:369` `search_similar_images()` | CLIP poster similarity; separate 512-d space |
| Config | `utils/config.py:7-11` | Model names + dims (text 384 / image 512, not comparable) |
| Movie identity | `pages/helpers.py:104` `get_movie_id()` | `f"{title}_{release_date}"` — fragile composite key |
| Evaluation | — | **Does not exist.** No metrics, no held-out split, no tests. `requirements.txt` even lists an unused `faiss-cpu`. |

### Why "semantic search ≠ recommender"
A recommender ranks items for a **specific user** using their history. This system
has no notion of a user. It answers "what is textually similar to X", which is a
*content-retrieval* question. That is a legitimate "more like this" feature — but
it is not personalization, and until today it was never measured.

---

## The first offline evaluation (built today)

**Ground truth — model-agnostic, so it fairly tests the current retrieval.**
We use **MovieLens `ml-latest-small`** (100k ratings, 610 users) as a behavioral
signal the embeddings never saw:

- Aligned MovieLens titles to the TMDB catalog (`data/9000plus.csv`) by
  `(normalized-title, year)` → **4,228 movies matched** (3,907 exact-year,
  321 title-only fallback).
- Built a **"more like this" relevance set** from co-rating: for a query movie,
  the relevant items are the movies most often **co-liked** (rating ≥ 4) by the
  same users, keeping candidates with ≥ 3 co-likers, top-30 per query.
- **1,072 usable queries** (≥ 8 likers, ≥ 5 relevant items each; avg ≈ 30).

Full builder + parameters: [`src/eval/build_eval.py`](../src/eval/build_eval.py).
This relevance is derived purely from user behavior, independent of any embedding
model, so it does not favor MiniLM or any replacement.

**Retrieval reproduction (no Milvus needed).** The baseline harness
([`src/eval/baseline.py`](../src/eval/baseline.py)) reproduces the live pipeline
exactly — same model, same text string
`"{title}. Genre: {genre}. Released: {year}. {overview}"`
(mirroring `embed_csv`), same normalization, IP == cosine == the exact inner
product that IVF_FLAT approximates — then ranks the whole catalog per query.

### Baseline results (1,072 held-out queries)

| System | NDCG@10 | Recall@20 | MAP@20 | Precision@10 | Catalog coverage | Intra-list diversity@10 |
|---|---|---|---|---|---|---|
| Random | 0.0034 | 0.0023 | 0.0006 | 0.0036 | 0.659 | 0.858 |
| **Popularity (non-personalized)** | **0.2148** | **0.1858** | **0.1042** | **0.2360** | 0.001 | 0.639 |
| **Semantic (current system)** | **0.0295** | 0.0109 | 0.0072 | 0.0237 | 0.534 | 0.678 |

*(numbers: [`results/baseline_metrics.json`](../results/baseline_metrics.json),
[`results/baseline_leaderboard.csv`](../results/baseline_leaderboard.csv))*

### The headline finding
> **The current semantic-search "recommendation engine" (NDCG@10 = 0.0295) is
> beaten ~7× by a one-line "recommend the globally most-popular movies to
> everyone" baseline (NDCG@10 = 0.2148).**

Content similarity is *not* what users co-like. Querying "Batman" returns other
Batman titles (thematically nearest), but the movies Batman-fans actually co-liked
are broader blockbusters — which popularity captures and content embeddings miss.
Semantic search beats random by ~9× (it is a real signal), but it is a **poor
recommender** on its own. This is the honest starting line the rest of the sprint
must beat: **any new ranker has to clear both 0.0295 (semantic) AND 0.2148
(popularity).**

Secondary observations:
- **Popularity coverage = 0.001** — it recommends the same ~10 blockbusters to
  everyone (textbook popularity bias). Semantic coverage = 0.534 (reaches half the
  catalog), so it is more personalizable in principle — it just isn't personalized.
- **Diversity:** semantic (0.678) < random (0.858) — retrieval clusters by
  genre/theme, as expected.

---

## Gaps this sprint will close (with the metric that will prove each)

| Gap | Evidence today | Will be measured by |
|---|---|---|
| No offline evaluation | (fixed Day 1) | NDCG@10 / recall@20 / MAP held-out split |
| Off-the-shelf embeddings | MiniLM-384 baseline = 0.0295 | Day 2 embedding bake-off (mpnet / e5 / bge) |
| Substring genre filter | line 354 | Day 2/5 metadata-aware retrieval |
| No personalization / CF | no user model | Day 3 ALS / kNN / SVD / hybrid on held-out users |
| No real transformer recommender | "transformer-powered" is framing | Day 7 SASRec / BERT4Rec next-item HR@10 / NDCG@10 |
| No cross-modal space | 384 vs 512 dims | Day 4 text+poster fusion |

**Differentiation guard:** CineSemantics's claim is **multimodal semantic
retrieval + rigorous offline evaluation on a 9.8K-movie catalog** — not a
rebuild of MatchMind's dating-CF bullets. The audit's spine is evaluation rigor
and the retrieval/multimodal angle.

---

## Reproduce

```bash
venv/Scripts/python.exe src/eval/build_eval.py   # download + align + build relevance
venv/Scripts/python.exe src/eval/baseline.py     # embed catalog + score baselines
```

Artifacts: `data/eval/` (relevance + alignment + manifest),
`results/baseline_metrics.json`, `results/baseline_leaderboard.csv`,
`results/samples/baseline_more_like_this.json`,
`results/figures/baseline_ndcg.png`.
