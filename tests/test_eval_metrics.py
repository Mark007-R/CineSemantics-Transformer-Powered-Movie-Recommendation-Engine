"""
Day-10 (Phase-8) — IR / recsys metric-function tests.

These lock the offline evaluation harness (src/eval/baseline.py) that Day 1 added
to close the project's biggest gap: it shipped as a "recommendation engine" with
ZERO evaluation. Every headline number in the README (NDCG@10, recall@20, MAP)
flows through these four functions, so a silent regression here would quietly
falsify the whole leaderboard. We pin them against hand-computed values.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.baseline import (               # noqa: E402
    dcg, ndcg_at_k, recall_at_k, precision_at_k, ap_at_k,
    genre_set, build_catalog_text,
)


# ---- dcg / ndcg ------------------------------------------------------------
def test_dcg_matches_closed_form():
    # relevance [1,0,1] -> 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5
    assert math.isclose(dcg([1.0, 0.0, 1.0]), 1.0 + 0.5, rel_tol=1e-9)


def test_ndcg_perfect_ranking_is_one():
    ranked = [3, 7, 1]
    relevant = {3, 7, 1}
    assert math.isclose(ndcg_at_k(ranked, relevant, 3), 1.0, rel_tol=1e-9)


def test_ndcg_no_relevant_hits_is_zero():
    assert ndcg_at_k([1, 2, 3], {9, 10}, 3) == 0.0


def test_ndcg_is_bounded_unit_interval():
    ranked = list(range(20))
    relevant = {0, 5, 11, 19}
    v = ndcg_at_k(ranked, relevant, 10)
    assert 0.0 <= v <= 1.0


def test_ndcg_rewards_higher_rank():
    relevant = {5}
    top = ndcg_at_k([5, 0, 1, 2], relevant, 4)
    bottom = ndcg_at_k([0, 1, 2, 5], relevant, 4)
    assert top > bottom          # same hit, earlier position scores higher


def test_ndcg_empty_relevant_set_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0


# ---- recall / precision ----------------------------------------------------
def test_recall_at_k_counts_hits_over_relevant():
    # 2 of 4 relevant recovered in the top-20
    assert math.isclose(recall_at_k(list(range(20)), {0, 1, 50, 51}, 20), 0.5)


def test_recall_empty_relevant_is_zero():
    assert recall_at_k([1, 2], set(), 20) == 0.0


def test_precision_at_k_is_hits_over_k():
    assert math.isclose(precision_at_k([0, 1, 2, 3], {0, 2}, 4), 0.5)


# ---- average precision -----------------------------------------------------
def test_ap_at_k_hand_computed():
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2 = 0.8333...
    ap = ap_at_k([10, 99, 11, 98], {10, 11}, 4)
    assert math.isclose(ap, (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-9)


def test_ap_at_k_no_hits_is_zero():
    assert ap_at_k([1, 2, 3], {9}, 3) == 0.0


def test_ap_rewards_relevant_first():
    good = ap_at_k([1, 0, 0], {1}, 3)
    bad = ap_at_k([0, 0, 1], {1}, 3)
    assert good > bad


# ---- genre / text helpers --------------------------------------------------
def test_genre_set_splits_all_delimiters():
    assert genre_set("Action, Adventure/Sci-Fi|Comedy") == {
        "action", "adventure", "sci-fi", "comedy"}


def test_genre_set_of_empty_is_empty():
    assert genre_set("") == set()


def test_build_catalog_text_uses_year_prefix_and_overview():
    row = {"Title": "Blade Runner", "Genre": "Sci-Fi",
           "Release_Date": "1982-06-25", "Overview": "A blade runner hunts."}
    txt = build_catalog_text(row)
    assert "Blade Runner" in txt
    assert "Genre: Sci-Fi" in txt
    assert "Released: 1982" in txt       # year sliced from the date
    assert "A blade runner hunts." in txt


def test_build_catalog_text_tolerates_missing_fields():
    txt = build_catalog_text({"Title": "Untitled"})
    assert txt == "Untitled"             # no genre/year/overview appended
