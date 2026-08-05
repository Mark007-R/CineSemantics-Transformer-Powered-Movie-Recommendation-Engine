# Semantic-Movie-Recommender

> 🔗 **Live demo:** https://iambatman07-semantic-movie-recommender.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/Semantic-Movie-Recommender) — the Milvus-free stack (`src/serving/space_app.py`)

Semantic + multimodal retrieval and personalized recommendation over a **9,826-movie TMDB catalog** (posters included), served through a Streamlit UI and a FastAPI inference service backed by Milvus / faiss HNSW.

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

Run the evaluation harness:

```bash
python -m src.eval.build_eval            # build held-out relevance + CF split
python -m src.eval.baseline
python -m src.recsys.cf_compare
python -m src.eval.ablation
```

---

## Data

- **Catalog:** `data/9000plus.csv` (9,826 TMDB movies) + `posters/` (~9,509 images).
- **Interactions:** public **MovieLens ml-latest-small**, aligned to the catalog by (title, year); only public data is used.
- **Splits:** per-user temporal hold-out; the candidate universe is training-only items, so no test interaction leaks into training (checked in `cf_manifest.json`).
- Cached embeddings live in `results/emb_cache/` so the API and eval share one encode.

---

## License
MIT — see [LICENSE](LICENSE).
