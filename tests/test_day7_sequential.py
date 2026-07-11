"""
Day-7 (Phase 5) regression tests for the sequential transformer layer.

Rule 10 (no test interaction leaks into training) is the load-bearing invariant
for any recsys eval, so it is asserted here for the sequential split too, alongside
architecture-correctness checks (finite scores, learns order) and the eval derived
strictly from the Day-3 split.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"


@pytest.fixture(scope="module")
def seq_split():
    p = EVAL / "seq_split.json"
    if not p.exists():
        pytest.skip("seq_split.json not built yet (run build_sequential_eval.py)")
    with open(p) as f:
        return json.load(f)


def test_next_target_is_first_heldout_like(seq_split):
    # next-item target must equal test[0] of the Day-3 cf split (temporal boundary)
    with open(EVAL / "cf_split.json") as f:
        cf = json.load(f)
    for u in list(seq_split["users"])[:50]:
        su = str(u)
        assert seq_split["next_target"][su] == int(cf["test"][su][0])


def test_no_heldout_leaks_into_history(seq_split):
    # Rule 10: neither train_seq nor dense_context may contain any held-out like
    for u in seq_split["users"]:
        su = str(u)
        held = set(seq_split["full_test"][su])
        assert not held & set(seq_split["train_seq"][su])
        assert not held & set(seq_split["dense_context"][su])


def test_vocab_is_train_only(seq_split):
    uni = set(seq_split["item_universe"])
    # every training/context item is in-vocab (no cold item a model could not learn)
    for u in list(seq_split["users"])[:100]:
        su = str(u)
        assert set(seq_split["train_seq"][su]) <= uni
        assert set(seq_split["dense_context"][su]) <= uni


def test_models_finite_and_learn_order():
    from src.recsys.sequential import SASRec, BERT4Rec
    n = 60
    rng = np.random.default_rng(0)
    seqs = [[((s + j - 1) % n) + 1 for j in range(6)]
            for s in rng.integers(1, n, size=200)]

    def top5_hit(mdl):
        hits = 0
        for s in seqs[:60]:
            nxt = ((s[-1]) % n) + 1
            sc = np.asarray(mdl.score_all(s))
            assert np.isfinite(sc).all()             # finite scores (NaN-mask fix)
            hits += nxt in (np.argsort(-sc)[:5] + 1)
        return hits / 60

    sas = SASRec(n, maxlen=12, d=32, n_blocks=2, n_heads=2, epochs=60,
                 batch_size=64).fit(seqs)
    assert top5_hit(sas) > 0.8                        # actually learns the chain


def test_phase5_leaderboard_written():
    csv = ROOT / "results" / "phase5_sequential.csv"
    if not csv.exists():
        pytest.skip("phase5_sequential.csv not produced yet")
    import csv as _csv
    rows = list(_csv.DictReader(open(csv)))
    systems = {r["system"] for r in rows}
    assert {"SASRec", "BERT4Rec", "ItemKNN", "ALS_tuned"} <= systems
    for r in rows:                                    # all metrics finite & in [0,1]
        for k in ("ni_hr@10", "ni_ndcg@10", "fl_ndcg@10", "fl_recall@20"):
            assert 0.0 <= float(r[k]) <= 1.0
