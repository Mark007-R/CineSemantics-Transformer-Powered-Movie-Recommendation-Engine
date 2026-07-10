"""
CineSemantics FastAPI service — the champions, served.

Day 5 stands up an async API in front of the production packages so the
benchmarked winners are reachable over HTTP, separate from the Streamlit app:

  GET  /health                service + artifact status
  POST /search      {query, filters}      champion e5 + HNSW + metadata filters
  POST /similar     {title|index, k}       "more like this" (+ metadata rerank)
  POST /recommend   {user_id | liked, k}   personalized ItemKNN CF (100% catalog-valid)

pymilvus does not import in this environment, so the API serves from the offline
faiss HNSW index (identical results); Milvus stays the Streamlit/Docker backend.
Run:  uvicorn api:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval.embedder import TextEmbedder
from retrieval.index import MovieIndex, HNSW_EF_SEARCH
from retrieval.metadata_filter import passes_filters, genre_tokens
from rerank.metadata_rerank import metadata_rerank, popularity_prior
from recsys.recommender import Recommender

DATA = ROOT / "data"

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    emb = TextEmbedder.load_catalog_matrix()
    STATE["catalog"] = catalog
    STATE["emb"] = emb
    STATE["embedder"] = TextEmbedder()               # lazy; model loads on 1st query
    STATE["index"] = MovieIndex(emb, kind="hnsw", ef_search=HNSW_EF_SEARCH)
    STATE["genres"] = [genre_tokens(g) for g in catalog["Genre"]]
    vc = pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0).values
    STATE["pop_norm"] = popularity_prior(vc)
    STATE["title_to_idx"] = {str(t).lower(): i for i, t in enumerate(catalog["Title"])}
    try:
        STATE["recommender"] = Recommender.from_split(catalog_embeddings=emb)
    except Exception as e:  # eval split not present -> recommend endpoint degrades
        STATE["recommender"] = None
        STATE["recommender_error"] = str(e)
    yield
    STATE.clear()


app = FastAPI(title="CineSemantics API", version="1.0", lifespan=lifespan)


# ---------- schemas ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=100)
    genres: list[str] | None = None
    genre_mode: str = Field("any", pattern="^(any|all)$")
    min_rating: float | None = Field(None, ge=0, le=10)
    max_rating: float | None = Field(None, ge=0, le=10)
    min_year: int | None = None
    max_year: int | None = None
    min_popularity: float | None = Field(None, ge=0)


class SimilarRequest(BaseModel):
    title: str | None = None
    index: int | None = Field(None, ge=0)
    k: int = Field(10, ge=1, le=100)
    rerank: bool = True


class RecommendRequest(BaseModel):
    user_id: int | None = None
    liked_titles: list[str] | None = None
    liked_indices: list[int] | None = None
    k: int = Field(10, ge=1, le=100)


class Movie(BaseModel):
    index: int
    title: str
    genre: str
    release_date: str
    vote_average: float
    score: float | None = None


def _movie(i: int, score: float | None = None) -> Movie:
    r = STATE["catalog"].iloc[i]
    return Movie(index=int(i), title=str(r["Title"]), genre=str(r["Genre"]),
                 release_date=str(r["Release_Date"]),
                 vote_average=float(r["Vote_Average"] or 0), score=score)


# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {"status": "ok",
            "catalog_size": int(len(STATE["catalog"])),
            "embedding": "intfloat/e5-base-v2 (768d)",
            "index": "faiss HNSW (efSearch=%d)" % HNSW_EF_SEARCH,
            "recommender": "ItemKNN" if STATE.get("recommender") else "unavailable"}


@app.post("/search")
async def search(req: SearchRequest):
    catalog = STATE["catalog"]
    qvec = STATE["embedder"].encode_query(req.query)
    # over-fetch, then apply metadata predicates and truncate to k
    idx, scores = STATE["index"].search(qvec, k=req.k * 8)
    out = []
    for i, s in zip(idx, scores):
        rec = catalog.iloc[i].to_dict()
        if passes_filters(rec, genres=req.genres, genre_mode=req.genre_mode,
                          min_rating=req.min_rating, max_rating=req.max_rating,
                          min_year=req.min_year, max_year=req.max_year,
                          min_popularity=req.min_popularity):
            out.append(_movie(i, round(float(s), 4)))
        if len(out) >= req.k:
            break
    return {"query": req.query, "count": len(out), "results": out}


@app.post("/similar")
async def similar(req: SimilarRequest):
    catalog = STATE["catalog"]
    if req.index is not None:
        qi = req.index
        if qi >= len(catalog):
            raise HTTPException(404, f"index {qi} out of range")
    elif req.title:
        qi = STATE["title_to_idx"].get(req.title.strip().lower())
        if qi is None:
            raise HTTPException(404, f"title not in catalog: {req.title!r}")
    else:
        raise HTTPException(422, "provide either 'title' or 'index'")

    idx, scores = STATE["index"].more_like_this(qi, k=200)
    if req.rerank:
        sims = np.full(len(catalog), -1e9, np.float32)
        for i, s in zip(idx, scores):
            sims[i] = s
        ranked = metadata_rerank(idx, sims, STATE["genres"], STATE["pop_norm"],
                                 query_genre=STATE["genres"][qi])[:req.k]
        results = [_movie(i, round(float(sims[i]), 4)) for i in ranked]
    else:
        results = [_movie(i, round(float(s), 4))
                   for i, s in zip(idx[:req.k], scores[:req.k])]
    return {"query_title": str(catalog.iloc[qi]["Title"]),
            "reranked": req.rerank, "results": results}


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    rec = STATE.get("recommender")
    if rec is None:
        raise HTTPException(503, "recommender unavailable: "
                            + STATE.get("recommender_error", "no cf split"))
    catalog = STATE["catalog"]
    try:
        if req.user_id is not None:
            items, scores = rec.recommend(req.user_id, k=req.k)
            mode = f"warm user {req.user_id}"
        else:
            liked = list(req.liked_indices or [])
            if req.liked_titles:
                for t in req.liked_titles:
                    j = STATE["title_to_idx"].get(t.strip().lower())
                    if j is not None:
                        liked.append(j)
            if not liked:
                raise HTTPException(422, "provide user_id, liked_titles, or liked_indices")
            items, scores = rec.recommend_from_likes(liked, k=req.k)
            mode = f"cold-start from {len(liked)} liked items"
    except KeyError as e:
        raise HTTPException(404, str(e))
    results = [_movie(i, round(float(s), 4)) for i, s in zip(items, scores)]
    return {"mode": mode, "catalog_valid": all(0 <= i < len(catalog) for i in items),
            "results": results}
