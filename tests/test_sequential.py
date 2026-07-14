"""
Day-10 (Phase-8) — sequential-recommender tests (src/recsys/sequential.py).

SASRec and BERT4Rec are the FIRST actual transformers used as rankers in this
"transformer-powered" engine (Day 7). Full-scale training is a Day-7 concern; here
we keep it a fast CPU smoke — tiny vocab, 2 epochs — and assert the contract every
downstream metric relies on: `score_all` returns one finite score per item, aligned
to the item vocabulary, so the Day-1 metric functions can rank these models with
zero changes. MarkovChain gets stronger correctness checks (it's cheap and exact):
order-awareness and popularity backoff.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recsys.sequential import MarkovChain, SASRec, BERT4Rec, set_seed   # noqa: E402


# ---- MarkovChain (order-aware, exact) --------------------------------------
def test_markov_learns_transitions():
    mc = MarkovChain(n_items=4)
    # after item 1 the next item is always item 2 (ids are 1-based)
    mc.fit([[1, 2, 1, 2], [1, 2]])
    scores = mc.score_all([1])
    assert scores.shape == (4,)
    assert int(np.argmax(scores)) == 1        # index 1 == item id 2


def test_markov_backs_off_to_popularity_for_unseen_last_item():
    mc = MarkovChain(n_items=4)
    mc.fit([[1, 2, 2, 2]])                     # item 2 is by far most frequent
    scores = mc.score_all([4])                 # item 4 never starts a transition
    # falls back to popularity distribution -> item 2 (index 1) most likely
    assert int(np.argmax(scores)) == 1


def test_markov_score_all_is_a_distribution_row():
    mc = MarkovChain(n_items=3)
    mc.fit([[1, 2, 3, 1]])
    row = mc.score_all([1])
    assert row.shape == (3,)
    assert np.all(row >= 0)


# ---- SASRec smoke ----------------------------------------------------------
def test_sasrec_score_all_shape_and_finite():
    set_seed(0)
    seqs = [[1, 2, 3, 4], [1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]]
    model = SASRec(n_items=6, maxlen=8, d=16, n_blocks=1, n_heads=2,
                   epochs=2, batch_size=8, seed=0)
    model.fit(seqs)
    scores = model.score_all([1, 2, 3])
    assert scores.shape == (6,)                # one score per real item
    assert np.all(np.isfinite(scores))         # finite masking -> no NaN/inf


def test_sasrec_is_deterministic_under_seed():
    seqs = [[1, 2, 3, 4], [2, 3, 4, 5]]
    a = SASRec(n_items=5, maxlen=6, d=16, n_blocks=1, n_heads=2,
               epochs=2, batch_size=4, seed=7).fit(seqs).score_all([1, 2])
    b = SASRec(n_items=5, maxlen=6, d=16, n_blocks=1, n_heads=2,
               epochs=2, batch_size=4, seed=7).fit(seqs).score_all([1, 2])
    assert np.allclose(a, b)                    # same seed -> identical scores


# ---- BERT4Rec smoke --------------------------------------------------------
def test_bert4rec_score_all_shape_and_finite():
    set_seed(0)
    seqs = [[1, 2, 3, 4], [1, 2, 3], [2, 3, 4, 5], [3, 4, 5, 6]]
    model = BERT4Rec(n_items=6, maxlen=8, d=16, n_blocks=1, n_heads=2,
                     epochs=2, batch_size=8, mask_prob=0.3, seed=0)
    model.fit(seqs)
    scores = model.score_all([2, 3, 4])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_sequential_item_vectors_align_to_vocab():
    model = SASRec(n_items=6, maxlen=8, d=16, n_blocks=1, n_heads=2,
                   epochs=1, batch_size=8, seed=0).fit([[1, 2, 3], [4, 5, 6]])
    vecs = model.item_vectors()
    assert vecs.shape == (6, 16)               # rows = items (pad col dropped)
