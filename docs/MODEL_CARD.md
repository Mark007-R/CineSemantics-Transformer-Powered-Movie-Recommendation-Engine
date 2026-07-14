# Model Card — CineSemantics Recommendation Stack

This card documents the production recommendation stack as of Day 10 of the upgrade
sprint. The stack has three cooperating models; the **personalized ranker (ItemKNN)** is
the champion that the project's name always implied but never had before this work.

---

## 1. Champion ranker — ItemKNN (item-item collaborative filtering)

| | |
|---|---|
| **Task** | Personalized movie recommendation: given a user's liked catalog items, rank the remaining catalog. |
| **Type** | Item-item collaborative filtering — cosine similarity over the binary user×item interaction matrix. |
| **Chosen over** | ALS, PureSVD, Hybrid (CF+content), Content-centroid, Popularity (Day-3 bake-off, `src/recsys/cf_compare.py`). |
| **Code** | `src/recsys/recommender.py` (`ItemKNNRecommender`). |
| **Artifacts** | `models/cf_interactions.npz` (interaction matrix) + `models/cf_meta.json` (item universe, sizes, headline NDCG). |

**Training data.** Public **MovieLens ml-latest-small** ratings, filtered to
likes (rating ≥ 4.0), aligned to the 9,826-movie TMDB catalog by (title, year). A
per-user **temporal** hold-out (most-recent 20% held out) forms train/test; the candidate
item universe is training-only items. Train/test are per-user disjoint — verified in
`data/eval/cf_manifest.json` (`integrity: PASS`) and locked by `tests/test_recsys.py`.

**Performance (held-out per-user split, `results/phase2b_cf.csv`).**

| Metric | Value |
|---|---|
| NDCG@10 | **0.1059** (+47% vs popularity 0.0719) |
| recall@20 | 0.1675 |
| MAP@20 | 0.0507 |
| precision@10 | 0.0793 |
| catalog coverage | 0.038 |
| intra-list diversity@10 | 0.787 |
| fit time (CPU) | 0.04 s |

**Cold start.** Users/items with no interaction neighbours fall back to a **content
centroid** over e5-base-v2 catalog embeddings; if no embeddings are available, to
training popularity. This is the genuine cold-start capability a history-free LLM lacks.

**Diversity control (Day-6).** Optional **MMR** reranking (genre-Jaccard redundancy,
λ=0.7) trades **−0.3pp NDCG@10 for +6.4pp intra-list diversity** to counter the dominant
failure mode found in error analysis (genre over-concentration).

**Known limitations.**
- Coverage is low (0.038): CF concentrates on well-connected head items. MMR and the
  content fallback partially mitigate this.
- Cannot rank items with zero training interactions (true cold items) — handled by the
  content fallback, not by CF itself.
- Set-based: ignores sequence order. The sequential transformers (below) model order but
  do not beat ItemKNN on full-list quality for this catalog.

---

## 2. Retrieval encoder — `intfloat/e5-base-v2`

| | |
|---|---|
| **Task** | Semantic search + item-item "more like this". |
| **Type** | 768-dim sentence-transformer bi-encoder, L2-normalized (IP = cosine), `"query: "` prefix. |
| **Chosen over** | all-MiniLM-L6-v2 (shipped), all-mpnet-base-v2, bge-base/small (Day-2, `results/phase2a_embeddings.csv`). |
| **Content NDCG@10** | 0.0482 (+63% vs shipped MiniLM 0.0295). |
| **Index** | faiss / Milvus **HNSW** (M=32, efConstruction=200, efSearch=64) — matches IVF_FLAT's exact NDCG at ~3.9× lower p95 latency. |
| **Code** | `src/retrieval/embedder.py`, `src/retrieval/index.py`. |

## 3. Sequential transformers — SASRec / BERT4Rec

| | |
|---|---|
| **Task** | Next-item prediction (the honest "transformer-powered" ranker). |
| **Type** | Self-attention sequence models (SASRec: causal; BERT4Rec: masked/cloze), CPU-trained. |
| **Result** | BERT4Rec ties ItemKNN on next-item NDCG@10 (0.0797 vs 0.0778); does **not** beat it on full-list NDCG@10. |
| **Status** | Benchmarked and documented; **not** the production ranker (ItemKNN wins full-list quality at ~1000× lower fit cost). Retained for the sequential/next-item surface. |
| **Code** | `src/recsys/sequential.py`, `results/phase5_sequential.csv`. |

---

## Reranker (production `/search`)

`src/rerank/metadata_rerank.py` — score = cosine + 0.2·genre-Jaccard + 0.05·popularity-prior.
Beat a neural cross-encoder by +42% NDCG at ~12× lower latency (Day-4); the cross-encoder
was rejected (+4% for 220× latency).

## Ethical & practical notes

- **Data:** public MovieLens + public TMDB metadata/posters only. No private user data.
- **Popularity prior** can amplify head bias; the MMR diversity control and coverage
  reporting exist to keep this visible and adjustable.
- **Grounding:** all recommendations are real catalog rows (0% off-catalog by construction),
  unlike the zero-shot LLM baseline (3.9% hallucinated titles).
- **Intended use:** content discovery / "more like this" / personalized suggestions on a
  fixed catalog. **Not** intended for high-stakes or fairness-sensitive ranking without
  additional exposure-fairness work (listed in the backlog).

_Last updated: Day 10 (2026-07-14)._
