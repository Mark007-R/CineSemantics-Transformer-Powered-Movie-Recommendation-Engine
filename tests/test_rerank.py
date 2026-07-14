"""
Day-10 (Phase-8) — metadata-filter + reranker tests
(src/retrieval/metadata_filter.py, src/rerank/metadata_rerank.py).

The headline here is the SUBSTRING-FILTER REGRESSION LOCK. The shipped engine
filtered genres with `genre_filter.lower() in movie_data['genre'].lower()`
(milvus_vectordb.py:354), which was wrong both ways:
  * false positive: "Sci" matched anything containing those letters,
  * false negative: "Romance" did NOT match a movie tagged "Romantic".
The token-set matcher fixes both; these tests fail if anyone reintroduces the
substring behaviour. Plus: the Day-4 champion reranker (cosine + genre-Jaccard +
popularity prior) actually reorders, and its free-text variant is anchor-safe.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.metadata_filter import (       # noqa: E402
    genre_tokens, genre_match, passes_filters,
)
from src.rerank.metadata_rerank import MetadataReranker    # noqa: E402


# ---- tokenisation ----------------------------------------------------------
def test_genre_tokens_splits_and_normalises():
    assert genre_tokens("Action, Adventure/Sci-Fi") == {"action", "adventure", "sci-fi"}


# ---- genre_match: the substring-filter regression --------------------------
def test_substring_false_positive_is_gone():
    # "sci" is a substring of "science fiction" but is NOT a genre token -> no match
    assert genre_match("Science Fiction", "Sci") is False


def test_partial_word_no_longer_matches():
    # old bug: "War" matched inside longer strings; token-set matching rejects it
    assert genre_match("Adventure", "Adv") is False


def test_exact_token_still_matches():
    assert genre_match("Action, Adventure", "Action") is True


def test_genre_match_any_mode():
    assert genre_match("Action, Comedy", ["Drama", "Comedy"], mode="any") is True
    assert genre_match("Action, Comedy", ["Drama", "Horror"], mode="any") is False


def test_genre_match_all_mode():
    assert genre_match("Action, Comedy, Romance", ["Action", "Comedy"], mode="all") is True
    assert genre_match("Action, Comedy", ["Action", "Romance"], mode="all") is False


def test_empty_movie_genre_never_matches():
    assert genre_match("", "Action") is False


def test_empty_request_passes_through():
    # no requested genres -> predicate is vacuously satisfied
    assert genre_match("Action", []) is True


# ---- passes_filters (numeric predicates) -----------------------------------
def test_passes_filters_rating_and_year():
    meta = {"genre": "Drama", "vote_average": 7.4,
            "release_date": "2011-08-01", "popularity": 42.0}
    assert passes_filters(meta, min_rating=7.0, min_year=2010) is True
    assert passes_filters(meta, min_rating=8.0) is False
    assert passes_filters(meta, max_year=2010) is False


def test_passes_filters_missing_numeric_fails_bound():
    meta = {"genre": "Drama", "vote_average": "", "release_date": ""}
    assert passes_filters(meta, min_rating=5.0) is False   # unparusable -> excluded


def test_passes_filters_popularity_floor():
    meta = {"genre": "Drama", "popularity": 3.0}
    assert passes_filters(meta, min_popularity=5.0) is False
    assert passes_filters(meta, min_popularity=1.0) is True


# ---- MetadataReranker (Day-4 champion) -------------------------------------
def _catalog():
    return pd.DataFrame([
        {"Genre": "Action, Adventure", "Vote_Count": 100},
        {"Genre": "Action", "Vote_Count": 5000},           # very popular
        {"Genre": "Comedy", "Vote_Count": 50},
    ])


def test_reranker_promotes_genre_and_popularity():
    rr = MetadataReranker(_catalog())
    # query movie 0 (Action,Adventure); candidates tie on cosine -> genre+pop decide
    order = rr.rerank(query_idx=0, candidate_idx=[2, 1], cosines=[0.5, 0.5])
    # movie 1 shares 'action' AND is far more popular -> ranked above the Comedy
    assert order[0] == 1


def test_rerank_query_without_anchor_uses_popularity():
    rr = MetadataReranker(_catalog())
    order = rr.rerank_query(candidate_idx=[0, 1, 2], cosines=[0.5, 0.5, 0.5])
    # equal cosine, no query genres -> popularity prior wins -> movie 1 first
    assert order[0] == 1


def test_rerank_query_honours_requested_genres():
    rr = MetadataReranker(_catalog())
    order = rr.rerank_query(candidate_idx=[1, 2], cosines=[0.5, 0.5],
                            query_genres=["Comedy"])
    # even though movie 1 is more popular, the Comedy anchor should lift movie 2;
    # assert the Comedy title is not dead last despite its tiny vote count
    assert 2 in order and order.index(2) == 0


def test_rerank_is_stable_length_preserving():
    rr = MetadataReranker(_catalog())
    order = rr.rerank(0, [0, 1, 2], [0.9, 0.1, 0.3])
    assert sorted(order) == [0, 1, 2]      # a permutation, nothing dropped/added
