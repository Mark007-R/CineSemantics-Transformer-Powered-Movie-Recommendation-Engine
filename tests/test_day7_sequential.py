"""
Day-7 (Phase-5) sequential-transformer regression tests.

Locks the Day-7 guarantees so a future refactor can't silently break them:
  1. LEAKAGE: the sequential eval derives from cf_split.json with ZERO drift --
     same users, train/test disjoint, and every next-item target is strictly
     held out (Rule 10). Asserted on the real split when present.
  2. NaN-SAFE ATTENTION: both SASRec (causal) and BERT4Rec (bidirectional) produce
     finite score matrices of the right shape on a tiny synthetic dataset -- the
     finite-mask (-1e9, never -inf) design must not emit NaNs on padded batches.
  3. SEEN-ITEM EXCLUSION contract: score_all_items returns raw item logits (it
     does NOT drop seen items); ranking is responsible for masking. This guards
     the eval code's assumption.
  4. RESULT PERSISTENCE: if the Day-7 run has executed, the persisted metrics are
     internally consistent (champion is the best next-item transformer; the
     honest full-list finding that no transformer beats the CF champion holds).

Fast by design: synthetic models are tiny (d=16, 1 block, few epochs); the
split/metrics assertions read persisted JSON and skip cleanly if absent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recsys.sequential import (train_sasrec, train_bert4rec,  # noqa: E402
                                   score_all_items)

EVAL = ROOT / "data" / "eval"
RESULTS = ROOT / "results"


def _tiny_seqs(n_users=30, n_items=25, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.integers(1, n_items + 1, size=rng.integers(4, 12)))
            for _ in range(n_users)], n_items


# 1. leakage / zero-drift on the real split ----------------------------------
def test_seq_eval_no_leakage():
    path = EVAL / "seq_eval.json"
    if not path.exists():
        pytest.skip("seq_eval.json not built yet")
    se = json.load(open(path))
    train = {u: set(v) for u, v in se["train"].items()}
    ni = se["next_item"]
    fl = {u: set(v) for u, v in se["full_list"].items()}
    # train/test disjoint and next-item target strictly held out
    assert all(train[u].isdisjoint(fl[u]) for u in train)
    assert all(int(ni[u]) not in train[u] for u in train)
    assert se["integrity"]["status"] == "PASS"
    # zero drift vs cf_split (same user set)
    cf = json.load(open(EVAL / "cf_split.json"))
    assert set(se["train"]) == set(cf["train"])


# 2. NaN-safe finite-mask attention ------------------------------------------
def test_sasrec_scores_finite():
    seqs, n = _tiny_seqs()
    m, _ = train_sasrec(seqs, n, d=16, blocks=1, heads=2, epochs=5, patience=3,
                        log=lambda *a: None)
    sc = score_all_items(m, seqs)
    assert sc.shape == (len(seqs), n)
    assert np.isfinite(sc).all()


def test_bert4rec_scores_finite():
    seqs, n = _tiny_seqs(seed=1)
    m, _ = train_bert4rec(seqs, n, d=16, blocks=1, heads=2, epochs=5, patience=3,
                          log=lambda *a: None)
    sc = score_all_items(m, seqs, append_mask=True)
    assert sc.shape == (len(seqs), n)
    assert np.isfinite(sc).all()


# 3. seen-item exclusion is the ranker's job, not the scorer's ---------------
def test_scorer_does_not_drop_seen():
    seqs, n = _tiny_seqs(seed=2)
    m, _ = train_sasrec(seqs, n, d=16, blocks=1, heads=2, epochs=3, patience=2,
                        log=lambda *a: None)
    sc = score_all_items(m, seqs)
    # every seen item still has a finite (non-masked) score in the raw matrix
    for i, s in enumerate(seqs):
        for it in s:
            assert np.isfinite(sc[i, it - 1])


# 4. persisted Day-7 results are internally consistent -----------------------
def test_day7_metrics_consistent():
    path = RESULTS / "phase5_metrics.json"
    if not path.exists():
        pytest.skip("Day-7 not run yet")
    d = json.load(open(path))
    champ = d["sequential_champion"]
    assert champ in ("SASRec", "BERT4Rec")
    ni = d["nextitem"]
    # champion is the best-next-item transformer
    other = "BERT4Rec" if champ == "SASRec" else "SASRec"
    assert ni[champ]["ndcg@10"] >= ni[other]["ndcg@10"]
    # honest headline: no transformer beats the CF champion on the full list
    fl = d["fulllist"]
    assert fl["ItemKNN"]["ndcg@10"] >= fl["SASRec"]["ndcg@10"]
    assert fl["ItemKNN"]["ndcg@10"] >= fl["BERT4Rec"]["ndcg@10"]
