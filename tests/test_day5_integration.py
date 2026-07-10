"""
Day-5 production integration tests.

Regression guards for the champions moving from experiment scripts into the
production packages. These do NOT re-run the full eval (that is
src/validate_production.py); they lock the behavioural contracts the API relies
on and the specific defects Day 5 fixed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.metadata_filter import genre_matches, genre_tokens, passes_filters
from retrieval.index import MovieIndex
from rerank.metadata_rerank import metadata_rerank, popularity_prior


# ---- genre filter: the Day-5 fix ----
def test_substring_false_positive_is_gone():
    # old bug: "art" in "Martial Arts" -> True. token-set says False.
    assert genre_matches("Martial Arts, Drama", ["art"], "any") is False
    assert genre_matches("Art House", ["art house"], "any") is True


def test_multi_genre_and_or():
    g = "Action, Comedy, Crime"
    assert genre_matches(g, ["Action", "Comedy"], "all") is True
    assert genre_matches(g, ["Action", "Horror"], "all") is False
    assert genre_matches(g, ["Action", "Horror"], "any") is True


def test_empty_wanted_matches_all():
    assert genre_matches("Drama", [], "all") is True
    assert genre_tokens("Action / Comedy | Drama") == {"action", "comedy", "drama"}


def test_passes_filters_rating_year():
    rec = {"Genre": "Action, Comedy", "Vote_Average": 7.5, "Release_Date": "2015-06-01",
           "Popularity": 40.0}
    assert passes_filters(rec, genres=["Action", "Comedy"], genre_mode="all",
                          min_rating=7.0, min_year=2010, max_year=2020) is True
    assert passes_filters(rec, min_rating=8.0) is False
    assert passes_filters(rec, max_year=2010) is False


# ---- index: exact vs HNSW ----
def test_index_exact_topk_and_exclude():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    flat = MovieIndex(x, kind="flat")
    idx, sc = flat.more_like_this(5, k=10)
    assert 5 not in idx                      # self excluded
    assert len(idx) == 10
    assert sc == sorted(sc, reverse=True)    # descending scores


def test_hnsw_matches_exact_on_top1():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 48)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    flat = MovieIndex(x, kind="flat")
    hnsw = MovieIndex(x, kind="hnsw")
    agree = sum(flat.more_like_this(i, 1)[0][0] == hnsw.more_like_this(i, 1)[0][0]
                for i in range(50))
    assert agree >= 45                       # HNSW ~ exact on nearest neighbour


# ---- reranker ----
def test_popularity_rerank_promotes_popular_within_pool():
    # two candidates with near-equal cosine; higher popularity should win.
    order = [0, 1]
    sims = np.array([0.80, 0.79], np.float32)
    genres = [set(), set()]
    pop = np.array([0.0, 1.0], np.float32)   # item 1 far more popular
    ranked = metadata_rerank(order, sims, genres, pop, a=0.0, b=0.1, pool=2)
    assert ranked[0] == 1


def test_popularity_prior_range():
    p = popularity_prior(np.array([0, 10, 100, 1000.0]))
    assert p.min() == 0.0 and p.max() == pytest.approx(1.0, abs=1e-6)
