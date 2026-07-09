"""
Day-5 (Phase-3) integration regression tests.

These lock the behaviour of TODAY's champion integration so a future refactor
can't silently undo it. This is the focused integration lock, NOT the full
per-module suite (that is Day-10: test_retrieval / test_recsys / test_rerank /
test_sequential / test_eval_metrics / test_api).

Fast by design: pure-logic tests need no model; the retrieval test reuses the
cached e5 catalog embeddings (results/emb_cache) and only exercises the faiss
item-item path (no query encoding), so no model download is required. Tests that
need data/eval or the cache are skipped cleanly if those artifacts are absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from src.retrieval.metadata_filter import genre_match, genre_tokens, passes_filters
from src.rerank.metadata_rerank import MetadataReranker
from src.recsys.recommender import ItemKNNRecommender

CACHE_EMB = ROOT / "results" / "emb_cache" / "intfloat__e5-base-v2.npy"
CATALOG = ROOT / "data" / "9000plus.csv"


# ------------------------------------------------ headline fix: genre filter
def test_genre_match_single_and_multi():
    assert genre_match("Action, Comedy", "Action") is True
    assert genre_match("Action, Comedy", "Horror") is False
    assert genre_match("Action, Adventure", ["Action", "Comedy"], mode="all") is False
    assert genre_match("Action, Comedy, Adventure", ["Action", "Comedy"], mode="all") is True
    assert genre_match("Action, Comedy", ["Horror", "Comedy"], mode="any") is True


def test_genre_match_fixes_substring_false_positive():
    # the OLD shipped filter: `genre_filter.lower() in genre.lower()`
    old_substring = "sci" in "science fiction"        # True  -> garbage match
    assert old_substring is True
    assert genre_match("Science Fiction", "Sci") is False   # token-set rejects it
    # and the substring false-NEGATIVE: "Romance" not a substring of "Romantic"
    assert genre_match("Romance", "Romantic") is False       # correctly no match
    assert genre_tokens("Action, Adventure") == {"action", "adventure"}


def test_passes_filters_numeric():
    m = {"genre": "Action", "vote_average": 7.5, "release_date": "2010-05-01",
         "popularity": 120.0}
    assert passes_filters(m, genres="Action", min_rating=7) is True
    assert passes_filters(m, min_rating=8) is False
    assert passes_filters(m, min_year=2011) is False
    assert passes_filters(m, max_year=2010) is True
    assert passes_filters(m, min_popularity=200) is False


# ------------------------------------------------ config champions
def test_config_points_to_champions():
    import config
    assert config.TEXT_MODEL_NAME == "intfloat/e5-base-v2"
    assert config.TEXT_EMBEDDING_DIMENSION == 768
    assert config.TEXT_MODEL_PREFIX == "query: "
    assert config.INDEX_TYPE == "HNSW"
    ip = config.build_index_params()
    assert ip["index_type"] == "HNSW" and "M" in ip["params"] and "efConstruction" in ip["params"]
    sp = config.build_search_params()
    assert "ef" in sp["params"]


# ------------------------------------------------ CF recommender
def _tiny_recommender():
    # 3 users, item universe = catalog indices [10,11,12,13,14]
    universe = [10, 11, 12, 13, 14]
    train = {1: [10, 11], 2: [10, 12], 3: [11, 12]}
    return ItemKNNRecommender().fit(train, universe), train, universe


def test_recommender_excludes_seen_and_is_catalog_valid():
    rec, train, universe = _tiny_recommender()
    out = rec.recommend([10, 11], top_k=3)
    idxs = [o["index"] for o in out]
    assert all(i in universe for i in idxs)          # only real catalog items
    assert 10 not in idxs and 11 not in idxs         # seen excluded
    assert all(o["method"] == "itemknn" for o in out)


def test_recommender_cold_start_fallback():
    rec, train, universe = _tiny_recommender()
    # a liked item OUTSIDE the interaction universe -> no CF neighbours
    out = rec.recommend([9999], top_k=3)
    assert out, "cold-start must still return something"
    # with no embeddings attached it falls back to popularity, still catalog-valid
    assert all(o["index"] in universe for o in out)


def test_recommender_save_load_roundtrip(tmp_path):
    rec, train, universe = _tiny_recommender()
    rec.save(tmp_path)
    loaded = ItemKNNRecommender.load(tmp_path)
    assert loaded.item_universe == universe
    a = [o["index"] for o in rec.recommend([10, 12], top_k=3)]
    b = [o["index"] for o in loaded.recommend([10, 12], top_k=3)]
    assert a == b


# ------------------------------------------------ reranker (text-query path)
def test_rerank_query_popularity_only_when_no_genres():
    import pandas as pd
    catalog = pd.DataFrame({
        "Genre": ["Action", "Comedy", "Drama"],
        "Vote_Count": [10, 1000, 500],
    })
    rr = MetadataReranker(catalog)
    cand = [0, 1, 2]
    cos = [0.90, 0.89, 0.88]   # item 0 slightly ahead on cosine
    # with no query genres, the popularity prior should lift the popular item (1)
    ordered = rr.rerank_query(cand, cos, query_genres=None)
    assert set(ordered) == {0, 1, 2}
    assert ordered[0] == 1     # popularity prior overtakes the tiny cosine gap


def test_rerank_query_uses_requested_genres():
    import pandas as pd
    catalog = pd.DataFrame({
        "Genre": ["Action", "Comedy", "Drama"],
        "Vote_Count": [10, 10, 10],   # equal popularity -> Jaccard decides
    })
    rr = MetadataReranker(catalog)
    ordered = rr.rerank_query([0, 1, 2], [0.80, 0.80, 0.80], query_genres=["Comedy"])
    assert ordered[0] == 1     # the Comedy item wins on genre-Jaccard


# ------------------------------------------------ retrieval index (cached emb)
@pytest.mark.skipif(not (CACHE_EMB.exists() and CATALOG.exists()),
                    reason="cached e5 embeddings / catalog not present")
def test_index_similar_and_search_filter():
    import pandas as pd
    from src.retrieval.index import MovieIndex
    catalog = pd.read_csv(CATALOG).fillna("")
    emb = np.load(CACHE_EMB).astype(np.float32)
    assert emb.shape[0] == len(catalog)
    idx = MovieIndex(catalog, emb)
    idx.build_faiss()
    # item-item similar excludes self, returns k valid indices (no model needed)
    qi = 0
    sim = idx.similar(qi, top_k=5)
    assert len(sim) == 5
    assert all(0 <= h["index"] < len(catalog) for h in sim)
    assert qi not in [h["index"] for h in sim]
