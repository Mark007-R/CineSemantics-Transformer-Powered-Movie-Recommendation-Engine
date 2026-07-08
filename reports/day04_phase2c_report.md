# CineSemantics Production Upgrade — Day 4 / 10

**Date:** 2026-07-08 · **Phase:** 2c — Reranking + multimodal fusion + ANN tuning ·
**Field:** Recommendation Systems / Multimodal Retrieval

## Resume gap progress
**Gap:** CineSemantics ships two off-the-shelf encoders (text MiniLM/e5 + CLIP
posters) and a Milvus store, but (a) never *reranks* retrieved candidates, (b)
never actually *uses* the poster/CLIP signal in ranking — it only does
poster→poster lookup in a separate tab — and (c) picked its Milvus index type
(IVF_FLAT) with **zero measured recall/latency justification**.
**Today's contribution:** Three head-to-heads on the Day-2 champion backbone
(e5-base-v2), all on the same 1,072-query held-out co-rating eval:
1. **ANN sweep** — Flat vs IVF_FLAT vs HNSW on recall@10-vs-exact, single-query
   p95 latency, and end-task NDCG@10 → **HNSW dominates the frontier**
   (matches exact quality at 3.9× lower latency; the current IVF_FLAT default is
   Pareto-dominated).
2. **Cross-encoder rerank** — the "obvious" neural upgrade → **a near-no-op that
   costs 220× latency**; a cheap metadata reranker beats it outright.
3. **Multimodal fusion** — finally fold the CLIP poster signal into the text
   ranking → **+43% NDCG@10** on a held-out split. The multimodal capability the
   project already paid for, but never ranked with.

## Files touched
- `src/rerank/rerank_fusion.py` (new, ~430 lines) — the three Phase-2c
  experiments; imports the **exact** Day-2 catalog-text builder and IR metrics
  from `src/eval/embedding_comparison.py` so every number stays comparable to
  Days 1–3. faiss ANN sweep, cross-encoder rerank pipeline, and on-demand TMDB
  poster download → CLIP fusion, with disk caches for both the e5 backbone and the
  CLIP poster embeddings.
- `src/retrieval/`, `src/rerank/` (new package dirs — the Day-5 Phase-3 homes).
- No production code (`utils/`, `pages/`) modified — the winning index type
  (HNSW) and the fusion score land in Day-5 Phase-3 integration; today measures
  *what* to ship.

## Setup
- **Compute:** CPU, Python 3.11.9. e5-base re-encode of 9,837 movies 7m43s (cache
  was gitignored, rebuilt); ANN sweep ~2 min; cross-encoder 70k pairs ~35 min;
  poster download 4,083 imgs 116 s + CLIP encode 139 s.
  - **Note (speedup opportunity):** the machine has an **NVIDIA RTX 4050**, but the
    installed PyTorch is the **CPU-only** wheel (`2.1.2+cpu`, `cuda.is_available()
    = False`). The cross-encoder ran ~7-core CPU; a CUDA torch build would cut it
    to seconds. Flagged for a deliberate post-run env upgrade (also benefits the
    StockAI LSTM leg later in the sprint) — not swapped mid-run to keep the
    reproducible environment stable.
- **Backbone:** Day-2 champion `intfloat/e5-base-v2` (768-dim, L2-normalised),
  scored item-item over the full catalog.
- **Eval:** the Day-1 MovieLens co-rating "more like this" set — 1,072 held-out
  queries, avg ~30 relevant each. Fusion uses a seeded 300-query sample (posters
  pulled from public TMDB `w200` thumbnails; the local `posters/` mirror is absent
  on this machine), tuned on a 150-query dev half and reported on the disjoint
  150-query test half.
- **Libraries:** `faiss-cpu` 1.7.4 (finally *used* — it was a dead dependency in
  `requirements.txt`), `sentence-transformers` CrossEncoder, `transformers` CLIP.

## Experiment A — ANN index sweep (Flat vs IVF_FLAT vs HNSW)
- **Hypothesis:** The production Milvus index type matters; the current IVF_FLAT
  default may not be on the recall/latency frontier.
- **Method:** Build each index over the 768-dim e5 catalog; for every query
  measure recall@10 against an exact brute-force ground truth, single-query p95
  latency, on-disk index size, and the **end-task NDCG@10** on the approx ranking.
  Milvus's IVF_FLAT / HNSW *are* these faiss algorithms, so the frontier transfers;
  characterising it here is Docker-free and fully reproducible.
- **Result** (`results/phase2c_ann_sweep.csv`, selected rows):

| Index | recall@10 vs exact | NDCG@10 | p95 latency (ms) | size (MB) |
|-------|-------------------:|--------:|-----------------:|----------:|
| Flat (exact / Milvus FLAT) | 1.000 | 0.0482 | 3.29 | 30.2 |
| IVF_FLAT nprobe=1 | 0.350 | 0.0463 | **0.06** | 30.7 |
| IVF_FLAT nprobe=16 | 0.893 | 0.0491 | 0.63 | 30.7 |
| IVF_FLAT nprobe=64 | 0.993 | 0.0483 | 2.18 | 30.7 |
| **HNSW efSearch=16** | 0.979 | 0.0481 | **0.29** | 32.9 |
| **HNSW efSearch=64** | **0.998** | **0.0482** | **0.84** | 32.9 |
| HNSW efSearch=256 | 1.000 | 0.0482 | 1.82 | 32.9 |

- **Interpretation:** **HNSW Pareto-dominates IVF_FLAT.** To reach ~0.99 recall,
  IVF_FLAT needs nprobe=64 → p95 2.18 ms (only 1.5× faster than exact); HNSW hits
  0.998 recall at p95 0.84 ms — **3.9× faster than exact, 2.6× faster than IVF at
  equal recall** — for +7% index memory (the graph). At efSearch=16 HNSW is
  **11× faster than exact** and still 0.979 recall. The current IVF_FLAT default
  is a defensible-but-suboptimal choice; Day-7 should ship HNSW.

## Experiment B — cross-encoder rerank (the expected upgrade that isn't)
- **Hypothesis:** Reranking the top-K semantic candidates with a cross-encoder
  will lift NDCG@10 (the standard retrieve-then-rerank recipe).
- **Method:** Rerank the top-15 / top-50 e5 candidates with
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, scoring each (query-movie-text,
  candidate-movie-text) pair; keep the semantic tail for recall fairness. Full
  1,072-query eval + realistic added p95 latency. Compared against pure semantic
  and the cheap Day-2 metadata reranker (cosine + genre-Jaccard + popularity).
- **Result** (`results/phase2c_rerank_fusion.csv`):

| Variant | NDCG@10 | Recall@20 | Prec@10 | p95 latency (ms) |
|---------|--------:|----------:|--------:|-----------------:|
| champion_semantic (e5-base) | 0.0482 | 0.0205 | 0.0410 | **2.7** |
| **metadata_rerank (Day-2 carried)** | **0.0683** | **0.0271** | **0.0539** | 48.8 |
| cross_encoder_rerank top-15 | 0.0503 | 0.0205 | 0.0428 | 604.7 |
| cross_encoder_rerank top-50 | 0.0490 | 0.0219 | 0.0419 | 1892.6 |

- **Interpretation:** The cross-encoder is the day's **negative result**. Top-15
  moves NDCG@10 by **+0.0021 (0.0482→0.0503)** while adding **~600 ms p95** — a
  **220× latency tax for a 4% relative bump**. Reranking a *deeper* pool (top-50)
  is *worse* (0.0490), because the mismatched model reshuffles good candidates.
  The cheap metadata reranker beats it outright (0.0683, +42% over pure) at 1/12th
  the latency. **Why:** ms-marco is trained for *query→passage search relevance*,
  not symmetric *movie↔movie* similarity, and — critically — it sees only text,
  so it cannot capture the **popularity signal** that Day-1 proved dominates this
  co-rating relevance (popularity alone scored NDCG@10 0.2148). The metadata
  reranker's popularity prior is exactly what the neural model lacks.

## Experiment C — multimodal (text + poster) fusion
- **Hypothesis:** Folding the CLIP poster signal into the text ranking improves
  "more like this" — co-watched films often share visual style (era, franchise,
  tone) that the text overview misses.
- **Method:** For the sampled queries, rescore the text candidate pool with
  `score = a·text_cos + (1−a)·CLIP_poster_cos`; tune `a` on the dev half, report on
  the disjoint test half. Text-only and fused are scored on the **identical**
  candidate pool, so the delta is the pure fusion effect.
- **Result** (`results/phase2c_fusion.csv`, 150 test queries):

| Variant | NDCG@10 (test) | ILD@5 (test) | tuned a(text) |
|---------|---------------:|-------------:|--------------:|
| text_only (champion cosine) | 0.0395 | 0.5368 | 1.00 |
| **text + poster fusion** | **0.0565** | 0.5194 | **0.75** |

- **Interpretation:** Fusion lifts NDCG@10 by **+43%** (0.0395→0.0565) with 25%
  weight on the poster signal — a real, held-out gain, tuned honestly on a
  disjoint dev half. Intra-list diversity dips slightly (0.537→0.519) as the
  ranking tightens toward visually-coherent neighbours. Concrete example (*Harry
  Potter: Chamber of Secrets*): text-only returned 3/5 relevant; fusion surfaced
  *Order of the Phoenix* and reordered to **4/5 relevant**. **This is the
  multimodal capability the "transformer-powered" project already had CLIP
  embeddings for but never ranked with** — the differentiation-guard win
  (multimodal retrieval + eval rigor), not a CF rebuild.

## Head-to-Head Comparison (running story)

| Eval | System | NDCG@10 | Note |
|------|--------|--------:|------|
| Day-1 content "more like this" | Semantic (MiniLM) | 0.0295 | first-ever metric |
| Day-2 content "more like this" | e5-base (champion) | 0.0482 | +63% over MiniLM |
| Day-3 personalized | ItemKNN (CF champion) | 0.1059 | first real personalization |
| **Day-4 content, reranked** | **metadata rerank** | **0.0683** | +42% over pure, cheap |
| Day-4 content, reranked | cross-encoder top-15 | 0.0503 | +4%, **220× latency** ✗ |
| **Day-4 content, fused** | **text+poster (test split)** | **0.0565** | **+43% over text-only** |

> Fusion rows are on a 150-query test split with a poster-restricted pool, so their
> absolute NDCG is not comparable to the full-1,072 rerank rows — the valid read is
> **each variant vs its own text-only baseline on the same queries**.

## Key findings
1. **HNSW beats the shipped IVF_FLAT on the recall/latency frontier** — exact
   quality at 3.9× lower latency (or 11× faster at 0.98 recall) for +7% memory.
   The current Milvus index choice was never measured; now it is. Ship HNSW (Day 7).
2. **The end task is remarkably robust to ANN approximation.** Even IVF_FLAT
   nprobe=1 (recall@10 vs exact = 0.35!) keeps NDCG@10 at 0.0463 — because
   co-rating relevance is diffuse, so dropping a few exact neighbours barely dents
   ranking quality. Useful: this recommender can afford aggressive ANN speedups.
3. **The cross-encoder is a trap (the counterintuitive one).** The textbook
   "retrieve-then-rerank with a cross-encoder" *loses* to a 5-line metadata
   reranker here and costs 220× the latency, because the reranker is trained for
   the wrong task (search relevance ≠ item similarity) and is blind to the
   popularity signal that actually drives this relevance.
4. **Poster fusion is the real multimodal win: +43% NDCG@10.** The visual signal
   is complementary to text and cheap (CLIP embeddings already exist). This is the
   first time posters influence the ranked list rather than living in a separate
   image-search tab.
5. **A dead dependency became a result.** `faiss-cpu` sat unused in
   `requirements.txt` (audited Day 1); the ANN sweep is its first real use and it
   directly informs the production index choice.

## What didn't work (and why)
- **Cross-encoder rerank:** +0.0021 NDCG@10 for +600 ms p95 — not worth shipping.
  *Why:* ms-marco cross-encoders model query→passage relevance, not movie↔movie
  similarity, and use text only, so they miss the popularity/co-rating structure.
  A *fine-tuned* reranker on co-rating pairs could help (future-sprint candidate),
  but the off-the-shelf one does not.
- **Deeper cross-encoder pool (top-50 < top-15):** more candidates gave the
  mismatched model more room to mis-rank. Reranking depth only helps when the
  reranker is better than the retriever on the task — here it is not.
- **Local poster mirror absent:** the 124 MB `posters/` folder isn't on this
  machine (known env quirk). Handled by pulling public TMDB `w200` thumbnails on
  demand (4,083/4,084 succeeded) and caching CLIP embeddings — reproducible, no
  proprietary data, per the media-discipline rule.

## Metrics update (primary)
- ANN champion: **HNSW efSearch=64** — recall@10 0.998 vs exact, NDCG@10 0.0482,
  **p95 0.84 ms (3.9× faster than exact)**, 32.9 MB.
- Rerank: metadata rerank NDCG@10 **0.0683** (+42%); cross-encoder rejected
  (+4% for 220× latency).
- Fusion: text+poster NDCG@10 **0.0565** vs text-only 0.0395 = **+43%** (held-out).

## Sample outputs saved
- `results/phase2c_ann_sweep.csv` — 12 index configs × recall/latency/NDCG/size.
- `results/phase2c_rerank_fusion.csv` — 4 rerank variants (full 1,072-query eval).
- `results/phase2c_fusion.csv` — text-only vs text+poster (held-out test split).
- `results/phase2c_metrics.json` — headline dict (append-friendly).
- `results/samples/phase2c_rerank_examples.json` — 8 queries, semantic vs
  cross-encoder vs poster-fused top-5 with hit flags.
- `results/figures/phase2c_ann_recall_latency.png`, `phase2c_rerank_ndcg.png`,
  `phase2c_fusion.png`.

## Next day
**Day 5 — Phase 3: Champion integration + production refactor (PHASE-WRAP).**
Create `src/{retrieval,recsys,rerank,eval}`; swap the champion embedding into
`utils/text_embedder.py`; **replace the substring genre filter in
`search_similar_movies` (line 354) with proper metadata filtering**; add the
missing CF/hybrid recommendation layer; carry today's **HNSW** index choice and
**poster-fusion** score into the retrieval path; stand up a FastAPI `api.py`
(`/search`, `/recommend`, `/similar`). Day-5 report includes a Phase wrap-up.

## Code changes
New: `src/rerank/rerank_fusion.py`, `src/retrieval/` + `src/rerank/` package dirs.
No `utils/`/`pages/` production code changed this session — index + fusion
integration is Day-5 Phase 3.
