"""
CineSemantics — production inference API (Day-5 Phase-3).

An async FastAPI service that serves the integrated champions independently of the
Streamlit UI and (optionally) of a running Milvus: it loads the e5-base-v2 catalog
embeddings + faiss HNSW index at startup, so /search, /similar and /recommend work
offline and reproducibly. Endpoints:

  GET  /health      liveness + which artifacts loaded
  POST /search      semantic search + PROPER metadata (genre/rating/year) filtering
                    + optional metadata rerank (Day-4 champion)
  POST /similar     item-item "more like this" + optional CLIP poster fusion (Day-4)
  POST /recommend   personalized CF (Day-3 ItemKNN champion) from liked titles/ids,
                    with content cold-start fallback

Run:  uvicorn api:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.embedder import ChampionEmbedder
from src.retrieval.index import MovieIndex
from src.rerank.metadata_rerank import MetadataReranker
from src.recsys.recommender import ItemKNNRecommender

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    embedder = ChampionEmbedder()
    emb = embedder.encode_catalog(catalog)     # cached e5-base-v2 vectors
    index = MovieIndex(catalog, emb, embedder)
    index.build_faiss()
    n_posters = index.load_poster_embeddings()
    reranker = MetadataReranker(catalog)
    rec = None
    if (MODELS / "cf_meta.json").exists():
        try:
            rec = ItemKNNRecommender.load(MODELS, catalog_embeddings=emb)
        except Exception:
            rec = None
    STATE.update(catalog=catalog, index=index, reranker=reranker, rec=rec,
                 n_posters=n_posters)
    yield
    STATE.clear()


app = FastAPI(title="CineSemantics API", version="0.5.0", lifespan=lifespan)


# ---------------------------------------------------------------- schemas
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="natural-language query")
    top_k: int = Field(10, ge=1, le=100)
    genres: list[str] | None = Field(None, description="filter to these genres")
    genre_mode: str = Field("any", pattern="^(any|all)$")
    min_rating: float | None = Field(None, ge=0, le=10)
    min_year: int | None = None
    max_year: int | None = None
    min_popularity: float | None = None
    rerank: bool = Field(True, description="apply the Day-4 metadata reranker")


class SimilarRequest(BaseModel):
    title: str | None = None
    movie_index: int | None = None
    top_k: int = Field(10, ge=1, le=100)
    poster_fusion: bool = Field(False, description="fuse CLIP poster signal (Day-4)")


class RecommendRequest(BaseModel):
    liked_titles: list[str] | None = None
    liked_indices: list[int] | None = None
    top_k: int = Field(10, ge=1, le=100)


class Movie(BaseModel):
    index: int
    title: str
    genre: str = ""
    release_date: str = ""
    score: float
    method: str | None = None


# ---------------------------------------------------------------- routes
@app.get("/health")
async def health():
    idx = STATE.get("index")
    return {
        "status": "ok" if idx is not None else "loading",
        "catalog_size": 0 if idx is None else len(idx.catalog),
        "embedding_model": "intfloat/e5-base-v2",
        "index": "HNSW",
        "poster_embeddings": STATE.get("n_posters", 0),
        "recommender_loaded": STATE.get("rec") is not None,
    }


@app.post("/search", response_model=list[Movie])
async def search(req: SearchRequest):
    idx: MovieIndex = STATE["index"]
    hits = idx.search(req.query, top_k=req.top_k, genres=req.genres,
                      genre_mode=req.genre_mode, min_rating=req.min_rating,
                      min_year=req.min_year, max_year=req.max_year,
                      min_popularity=req.min_popularity,
                      overfetch=30 if req.rerank else 20)
    if req.rerank and hits:
        # rerank the returned pool by the query's own genre profile is undefined
        # for a free-text query, so rerank uses popularity + cosine only via a
        # pseudo query: reuse cosine order (already ranked) — metadata rerank here
        # applies the popularity prior over the candidate pool.
        rr: MetadataReranker = STATE["reranker"]
        cand = [h["index"] for h in hits]
        cos = [h["score"] for h in hits]
        # genre-Jaccard needs a query genre set; for text search we skip Jaccard by
        # passing the top hit's genre as the anchor (keeps popularity prior active).
        anchor = cand[0]
        ordered = rr.rerank(anchor, cand, cos)
        by_idx = {h["index"]: h for h in hits}
        hits = [by_idx[i] for i in ordered]
    return [Movie(**{**h, "method": "search"}) for h in hits[:req.top_k]]


@app.post("/similar", response_model=list[Movie])
async def similar(req: SimilarRequest):
    idx: MovieIndex = STATE["index"]
    qi = req.movie_index
    if qi is None and req.title:
        qi = idx.title_to_index(req.title)
    if qi is None:
        raise HTTPException(status_code=404, detail="movie not found (title/index)")
    hits = idx.similar(qi, top_k=req.top_k, use_poster_fusion=req.poster_fusion)
    return [Movie(**{**h, "method": "poster_fusion" if req.poster_fusion else "text"})
            for h in hits]


@app.post("/recommend", response_model=list[Movie])
async def recommend(req: RecommendRequest):
    rec: ItemKNNRecommender = STATE.get("rec")
    if rec is None:
        raise HTTPException(status_code=503, detail="recommender not loaded (run integrate_champions.py)")
    idx: MovieIndex = STATE["index"]
    liked = list(req.liked_indices or [])
    if req.liked_titles:
        for t in req.liked_titles:
            i = idx.title_to_index(t)
            if i is not None:
                liked.append(i)
    if not liked:
        raise HTTPException(status_code=400, detail="provide liked_titles or liked_indices")
    recs = rec.recommend(liked, top_k=req.top_k)
    out = []
    for r in recs:
        m = idx._meta[r["index"]]
        out.append(Movie(index=r["index"], title=str(m.get("Title", "")),
                         genre=str(m.get("Genre", "")),
                         release_date=str(m.get("Release_Date", "")),
                         score=r["score"], method=r["method"]))
    return out
