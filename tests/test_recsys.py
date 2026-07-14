"""
Day-10 (Phase-8) — recommender-layer tests (src/recsys/recommender.py) +
the CF-split leakage regression (Hard Rule 10: no test interaction leaks into
training).

The ItemKNN champion is the personalization layer the "Recommendation Engine"
name always promised but never had before this sprint. These tests pin:
  * item-item CF actually ranks by co-occurrence and excludes already-seen items,
  * the content cold-start fallback fires for brand-new users (the capability an
    LLM with no interaction history cannot provide),
  * MMR diversity reranking (Day-6 fix) stays catalog-valid,
  * save/load round-trips the load-critical `item_universe` (the artifact bug that
    503'd /recommend until the Day-9 self-heal),
  * and — the headline regression — the real Day-3 held-out split has ZERO
    train/test overlap per user.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recsys.recommender import ItemKNNRecommender     # noqa: E402

EVAL = ROOT / "data" / "eval"


def _toy_cf():
    """Two co-occurrence cliques so item-item neighbours are obvious.
    Users 0-2 like {0,1,2}; users 3-5 like {3,4,5}. item_universe = 0..5."""
    train = {0: [0, 1, 2], 1: [0, 1], 2: [1, 2],
             3: [3, 4, 5], 4: [3, 4], 5: [4, 5]}
    return train, [0, 1, 2, 3, 4, 5]


def _fit():
    train, universe = _toy_cf()
    return ItemKNNRecommender().fit(train, universe)


# ---- core CF ---------------------------------------------------------------
def test_recommend_returns_catalog_indices_in_universe():
    rec = _fit()
    out = rec.recommend([0], top_k=3)
    assert out and all(r["index"] in range(6) for r in out)
    assert all(r["method"] == "itemknn" for r in out)


def test_recommend_ranks_cooccurring_items_first():
    rec = _fit()
    out = rec.recommend([0], top_k=2, exclude_seen=True)
    # a user who liked item 0 should be steered to items 1 and 2 (its clique),
    # never to the disjoint clique {3,4,5}
    recced = [r["index"] for r in out]
    assert set(recced) <= {1, 2}


def test_recommend_excludes_seen():
    rec = _fit()
    out = rec.recommend([0, 1, 2], top_k=6, exclude_seen=True)
    recced = [r["index"] for r in out]
    assert all(i not in recced for i in (0, 1, 2))     # never re-recommend seen


def test_top_k_is_respected():
    rec = _fit()
    assert len(rec.recommend([0], top_k=1)) == 1


# ---- cold start ------------------------------------------------------------
def test_content_coldstart_for_unknown_user():
    """A user whose likes are all OUTSIDE the interaction universe -> content path."""
    train, universe = _toy_cf()
    emb = np.eye(10, dtype=np.float32)     # 10 "catalog" items, orthonormal
    rec = ItemKNNRecommender().fit(train, universe).attach_embeddings(emb)
    out = rec.recommend([7], top_k=3)      # item 7 has no CF neighbours
    assert out and all(r["method"] == "content_coldstart" for r in out)
    assert 7 not in [r["index"] for r in out]          # excludes the liked item


def test_coldstart_last_resort_popularity_without_embeddings():
    train, universe = _toy_cf()
    rec = ItemKNNRecommender().fit(train, universe)    # no embeddings attached
    out = rec.recommend([999], top_k=3)                # nothing in universe/emb
    assert out and all(r["index"] in range(6) for r in out)   # falls back to popular


# ---- MMR diversity (Day-6) -------------------------------------------------
def test_mmr_diversity_stays_catalog_valid():
    train, universe = _toy_cf()
    catalog = pd.DataFrame({"Genre": ["Action", "Action", "Comedy",
                                       "Drama", "Drama", "Sci-Fi"]})
    rec = ItemKNNRecommender().fit(train, universe).attach_genres(catalog)
    out = rec.recommend([0], top_k=3, diversity=0.7)
    assert out and all(r["method"] == "itemknn+mmr" for r in out)
    assert all(r["index"] in range(6) for r in out)


# ---- persistence -----------------------------------------------------------
def test_save_load_roundtrip_preserves_item_universe(tmp_path):
    rec = _fit()
    rec.save(tmp_path, display={"day3_ndcg@10": 0.1059})
    meta = json.loads((tmp_path / "cf_meta.json").read_text())
    assert "item_universe" in meta                     # the load-critical field
    assert meta["day3_ndcg@10"] == 0.1059              # display merged, not clobbered
    reloaded = ItemKNNRecommender.load(tmp_path)
    assert reloaded.item_universe == rec.item_universe
    # reloaded model produces the same top rec as the original
    assert (reloaded.recommend([0], top_k=1)[0]["index"]
            == rec.recommend([0], top_k=1)[0]["index"])


# ---- LEAKAGE REGRESSION (Hard Rule 10) -------------------------------------
@pytest.mark.skipif(not (EVAL / "cf_split.json").exists(),
                    reason="cf_split.json not built")
def test_cf_split_has_no_train_test_leakage():
    """The real Day-3 held-out split: every test interaction must be strictly
    disjoint from that user's training interactions."""
    split = json.loads((EVAL / "cf_split.json").read_text())
    train, test = split["train"], split["test"]
    assert set(train) == set(test)                     # same users on both sides
    total_overlap = 0
    for u in train:
        overlap = set(train[u]) & set(test.get(u, []))
        total_overlap += len(overlap)
    assert total_overlap == 0, f"{total_overlap} leaked train/test interactions"


@pytest.mark.skipif(not (EVAL / "cf_manifest.json").exists(),
                    reason="cf_manifest.json not built")
def test_cf_manifest_reports_integrity_pass():
    manifest = json.loads((EVAL / "cf_manifest.json").read_text())
    assert manifest["train_test_overlap"] == 0
    assert manifest["integrity"] == "PASS"
