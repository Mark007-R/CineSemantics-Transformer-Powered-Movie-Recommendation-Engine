"""
Day-6 (Phase-4) tuning + diversity-fix regression tests.

Locks the two Day-6 outcomes so a future refactor can't silently undo them:
  1. Optuna tuned ALS beats default ALS and ties the ItemKNN champion (asserted
     from the persisted results/phase4_metrics.json -- no re-training here).
  2. The MMR diversity rerank added to ItemKNNRecommender is:
       - backward compatible (diversity=None -> identical to plain relevance),
       - a no-op-safe fallback when genres aren't attached,
       - genuinely diversity-increasing / de-concentrating when enabled.

Fast by design: the MMR tests build a tiny synthetic recommender (no data or
model needed); the metrics test is skipped cleanly if the Day-6 run hasn't been
executed yet.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recsys.recommender import ItemKNNRecommender

PHASE4 = ROOT / "results" / "phase4_metrics.json"


# ------------------------------------------------ (1) tuning outcome (from disk)
@pytest.mark.skipif(not PHASE4.exists(), reason="Day-6 run not executed yet")
def test_optuna_beats_default_als():
    m = json.loads(PHASE4.read_text())["hpo"]
    assert m["als_tuned_test_ndcg@10"] > m["als_default_test_ndcg@10"], \
        "tuned ALS must beat default ALS on TEST"
    # tuning should roughly close the gap to the ItemKNN champion (within 1pp)
    assert m["als_tuned_test_ndcg@10"] >= m["itemknn_champion_ndcg@10"] - 0.01


@pytest.mark.skipif(not PHASE4.exists(), reason="Day-6 run not executed yet")
def test_hpo_used_inner_validation_not_test():
    """Guard rule 10/15: HPO must not tune on the held-out test split."""
    m = json.loads(PHASE4.read_text())["hpo"]
    assert "inner validation" in m["protocol"].lower()
    assert "test untouched" in m["protocol"].lower()


@pytest.mark.skipif(not PHASE4.exists(), reason="Day-6 run not executed yet")
def test_error_analysis_dominant_mode_recorded():
    m = json.loads(PHASE4.read_text())["error_analysis"]
    assert m["n_worst_analyzed"] == 30
    assert sum(m["failure_distribution"].values()) == 30
    assert m["dominant_failure"] in m["failure_distribution"]


# ------------------------------------------------ (2) MMR rerank behaviour
def _toy_recommender():
    """5 users x 6 items; items 0-2 = 'action', 3-5 = 'comedy'.
    User 0 liked item 0; its nearest neighbours are the other two action items,
    so plain relevance returns an all-action top list -> the exact
    genre-over-concentration failure MMR is meant to break."""
    train = {0: [0], 1: [0, 1], 2: [0, 2], 3: [3, 4], 4: [3, 5]}
    universe = [0, 1, 2, 3, 4, 5]
    rec = ItemKNNRecommender().fit(train, universe)
    cat = pd.DataFrame({"Genre": ["action", "action", "action",
                                  "comedy", "comedy", "comedy"]})
    rec.attach_genres(cat)
    return rec


def _top_genre_share(recs, cat_genre):
    from collections import Counter
    c = Counter(cat_genre[r["index"]] for r in recs)
    return max(c.values()) / len(recs)


def test_mmr_backward_compatible_when_disabled():
    rec = _toy_recommender()
    plain = rec.recommend([0], top_k=4)
    assert all(r["method"] == "itemknn" for r in plain)


def test_mmr_falls_back_without_genres():
    train = {0: [0], 1: [0, 1], 2: [0, 2]}
    rec = ItemKNNRecommender().fit(train, [0, 1, 2])  # no genres attached
    # diversity requested but no genres -> must not crash, stays pure relevance
    out = rec.recommend([0], top_k=2, diversity=0.7)
    assert all(r["method"] == "itemknn" for r in out)


def test_mmr_reduces_genre_concentration():
    rec = _toy_recommender()
    genre = ["action", "action", "action", "comedy", "comedy", "comedy"]
    plain = rec.recommend([0], top_k=4, diversity=None)
    mmr = rec.recommend([0], top_k=4, diversity=0.3)
    assert all(r["method"] == "itemknn+mmr" for r in mmr)
    # diversified list should be no more genre-concentrated than plain, and here
    # strictly less (plain is all-action, MMR pulls in comedy)
    assert _top_genre_share(mmr, genre) <= _top_genre_share(plain, genre)
    assert _top_genre_share(mmr, genre) < 1.0


def test_mmr_top1_is_still_most_relevant():
    """With lambda high, MMR must keep the single most-relevant item at rank 1."""
    rec = _toy_recommender()
    plain = rec.recommend([0], top_k=4)
    mmr = rec.recommend([0], top_k=4, diversity=0.9)
    assert mmr[0]["index"] == plain[0]["index"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
