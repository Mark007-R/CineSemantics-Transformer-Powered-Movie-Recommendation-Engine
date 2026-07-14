"""
Day-10 (Phase-8) — retrieval-layer tests (src/retrieval/*).

Exercises the champion retrieval path built across Days 2/4/5 WITHOUT loading the
e5-base model or Milvus: a tiny synthetic catalog + deterministic unit vectors is
enough to prove faiss HNSW search, PROPER metadata filtering (the fix for the
substring genre filter that shipped at milvus_vectordb.py:354), item-item
'more like this', and title lookup all behave. Edge cases: empty/no-hit filters,
out-of-catalog title, self-exclusion, and the catalog/embedding length guard.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.index import MovieIndex          # noqa: E402


def _toy():
    """5-movie catalog with orthonormal-ish 4-d vectors so neighbours are known."""
    catalog = pd.DataFrame([
        {"Title": "Alpha", "Genre": "Action, Adventure", "Release_Date": "2001-01-01",
         "Vote_Average": 8.0, "Vote_Count": 500, "Popularity": 50.0, "Overview": "a", "Poster_Url": ""},
        {"Title": "Beta", "Genre": "Action", "Release_Date": "1998-05-01",
         "Vote_Average": 6.0, "Vote_Count": 100, "Popularity": 20.0, "Overview": "b", "Poster_Url": ""},
        {"Title": "Gamma", "Genre": "Comedy", "Release_Date": "2015-07-01",
         "Vote_Average": 7.5, "Vote_Count": 900, "Popularity": 80.0, "Overview": "c", "Poster_Url": ""},
        {"Title": "Delta", "Genre": "Romance, Comedy", "Release_Date": "2020-01-01",
         "Vote_Average": 5.0, "Vote_Count": 30, "Popularity": 5.0, "Overview": "d", "Poster_Url": ""},
        {"Title": "Epsilon", "Genre": "Sci-Fi", "Release_Date": "1982-06-25",
         "Vote_Average": 9.0, "Vote_Count": 1200, "Popularity": 99.0, "Overview": "e", "Poster_Url": ""},
    ])
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],   # closest to Alpha
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],   # closest to Gamma
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return catalog, emb


class _StubEmbedder:
    """Returns a fixed query vector so search() needs no real model."""
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def encode_query(self, text):
        return self._vec


def _index(query_vec=(1.0, 0.0, 0.0, 0.0)):
    catalog, emb = _toy()
    idx = MovieIndex(catalog, emb, embedder=_StubEmbedder(query_vec))
    idx.build_faiss()
    return idx


# ---- construction ----------------------------------------------------------
def test_length_mismatch_is_rejected():
    catalog, emb = _toy()
    with pytest.raises(AssertionError):
        MovieIndex(catalog, emb[:3])


def test_title_to_index_case_insensitive():
    idx = _index()
    assert idx.title_to_index("alpha") == 0
    assert idx.title_to_index("  EPSILON ") == 4


def test_title_to_index_out_of_catalog_is_none():
    idx = _index()
    assert idx.title_to_index("Nonexistent Movie 9000") is None


# ---- search ----------------------------------------------------------------
def test_search_returns_nearest_first():
    idx = _index(query_vec=(1.0, 0.0, 0.0, 0.0))
    hits = idx.search("anything", top_k=2)
    assert [h["index"] for h in hits] == [0, 1]     # Alpha then Beta
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_hits_are_catalog_valid():
    idx = _index()
    hits = idx.search("q", top_k=3)
    n = len(idx.catalog)
    assert all(0 <= h["index"] < n for h in hits)
    assert all(h["title"] for h in hits)            # never a hallucinated blank


def test_search_genre_filter_any():
    idx = _index()
    hits = idx.search("q", top_k=5, genres=["Comedy"])
    titles = {h["title"] for h in hits}
    assert titles == {"Gamma", "Delta"}             # only Comedy movies survive


def test_search_genre_filter_all_mode():
    idx = _index()
    hits = idx.search("q", top_k=5, genres=["Romance", "Comedy"], genre_mode="all")
    assert {h["title"] for h in hits} == {"Delta"}  # only Delta has BOTH


def test_search_min_rating_filter():
    idx = _index()
    hits = idx.search("q", top_k=5, min_rating=8.0)
    assert {h["title"] for h in hits} == {"Alpha", "Epsilon"}


def test_search_year_window_filter():
    idx = _index()
    hits = idx.search("q", top_k=5, min_year=2010, max_year=2019)
    assert {h["title"] for h in hits} == {"Gamma"}  # 2015 only


def test_search_impossible_filter_returns_empty():
    idx = _index()
    assert idx.search("q", top_k=5, min_rating=9.9, min_year=2100) == []


# ---- similar ---------------------------------------------------------------
def test_similar_excludes_self_and_ranks_neighbour():
    idx = _index()
    hits = idx.similar(0, top_k=2)          # Alpha
    assert 0 not in [h["index"] for h in hits]
    assert hits[0]["index"] == 1            # Beta is nearest to Alpha


def test_similar_poster_fusion_falls_back_without_posters():
    # no poster embeddings loaded -> fusion path must degrade to pure text, not crash
    idx = _index()
    hits = idx.similar(2, top_k=2, use_poster_fusion=True)
    assert len(hits) == 2
    assert all("score" in h for h in hits)
