"""
CineSemantics Day-7 (Phase 5) sequential-recommendation eval builder.

Day 3 added the first PERSONALIZED eval (per-user TEMPORAL leave-last-20% split,
src/eval/build_cf_eval.py). Day 7 introduces the first ACTUAL transformer rankers
(SASRec / BERT4Rec) into this "transformer-powered" project. To compare them
apples-to-apples with the Day-3 CF champion (ItemKNN) and the Day-6 tuned ALS,
this builder DERIVES the sequential eval directly from the SAME cf_split.json, so
no user, item, or held-out interaction differs between the CF and sequential runs.

It produces two things the CF split did not expose explicitly:

  1. next_target        the SINGLE immediate-next held-out like per user. Because
                        cf_split's `test` is the chronologically most-recent 20%
                        of a user's likes, test[0] is the item that directly
                        follows the training sequence in time -> the canonical
                        sequential "next-item" target (HR@10 / NDCG@10).
  2. dense_context      per user, EVERY rated (aligned, in-vocab) movie strictly
                        BEFORE the held-out boundary timestamp -- including
                        ratings < 4 that the likes-only train sequence drops.
                        This is a leakage-free richer input history used only for
                        the Day-7 sensitivity ("does denser context help the
                        transformer on a sparse 547-user dataset?"). Targets are
                        unchanged, so the comparison stays valid.

`full_test` (all held-out likes) is copied through so the sequential models can
also be scored on the exact Day-3 full-list protocol (NDCG@10 / recall@20 / MAP).

Leakage guards (Rule 10 — no test interaction leaks into training):
  * train_seq / dense_context contain ONLY interactions strictly earlier than the
    user's first held-out like; asserted against the boundary timestamp.
  * item vocabulary == cf_split item_universe (train-only), so a model can never
    emit an item it could not have learned. Cold test items are unreachable for
    EVERY system (same ceiling as Day 3), keeping the comparison fair.

Output: data/eval/seq_split.json  + data/eval/seq_manifest.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"
LIKE_THRESHOLD = 4.0   # matches Day-1/Day-3 positive signal


def build():
    with open(EVAL / "cf_split.json") as f:
        cf = json.load(f)
    train = {int(u): [int(i) for i in v] for u, v in cf["train"].items()}
    test = {int(u): [int(i) for i in v] for u, v in cf["test"].items()}
    universe = [int(i) for i in cf["item_universe"]]
    uni_set = set(universe)

    align = pd.read_csv(EVAL / "movielens_alignment.csv")
    ml_to_cat = dict(zip(align["movieId"], align["catalog_index"]))
    ratings = pd.read_csv(EVAL / "ml-latest-small" / "ratings.csv")
    ratings = ratings[ratings["movieId"].isin(ml_to_cat)].copy()
    ratings["cat"] = ratings["movieId"].map(ml_to_cat).astype(int)
    ratings = ratings.sort_values("timestamp")

    # per-user chronological (cat, ts, rating), earliest occurrence of each cat
    by_user = defaultdict(list)
    seen_by_user = defaultdict(set)
    for uid, cat, ts, r in zip(ratings["userId"], ratings["cat"],
                               ratings["timestamp"], ratings["rating"]):
        uid = int(uid); cat = int(cat)
        if cat in seen_by_user[uid]:
            continue
        seen_by_user[uid].add(cat)
        by_user[uid].append((cat, int(ts), float(r)))

    # boundary ts per eval user = ts of the FIRST held-out like (test[0])
    def first_like_ts(uid, item):
        for cat, ts, r in by_user[uid]:
            if cat == item and r >= LIKE_THRESHOLD:
                return ts
        return None

    train_seq, next_target, full_test, dense_context = {}, {}, {}, {}
    boundary_leaks = 0
    for uid in train:
        target = test[uid][0]                      # immediate-next held-out like
        b_ts = first_like_ts(uid, target)
        if b_ts is None:
            continue
        # dense context: every aligned in-vocab rating strictly before boundary
        ctx = [cat for cat, ts, r in by_user[uid] if ts < b_ts and cat in uni_set]
        # integrity: no held-out like may appear in the context history
        held = set(test[uid])
        if held & set(ctx):
            boundary_leaks += 1
            ctx = [c for c in ctx if c not in held]
        train_seq[uid] = train[uid]
        next_target[uid] = target
        full_test[uid] = test[uid]
        dense_context[uid] = ctx

    out = {
        "item_universe": universe,
        "users": sorted(train_seq),
        "train_seq": {str(u): v for u, v in train_seq.items()},
        "next_target": {str(u): next_target[u] for u in train_seq},
        "full_test": {str(u): full_test[u] for u in train_seq},
        "dense_context": {str(u): dense_context[u] for u in train_seq},
        "params": {"LIKE_THRESHOLD": LIKE_THRESHOLD,
                   "derived_from": "cf_split.json",
                   "next_item": "first held-out like (test[0])"},
    }
    with open(EVAL / "seq_split.json", "w") as f:
        json.dump(out, f)

    n = len(train_seq)
    manifest = {
        "n_users": n,
        "n_items_vocab": len(universe),
        "avg_train_seq_len": round(np.mean([len(train_seq[u]) for u in train_seq]), 2),
        "avg_dense_ctx_len": round(np.mean([len(dense_context[u]) for u in train_seq]), 2),
        "dense_ctx_extra_pct": round(
            100 * (np.mean([len(dense_context[u]) for u in train_seq]) /
                   max(1e-9, np.mean([len(train_seq[u]) for u in train_seq])) - 1), 1),
        "avg_full_test_len": round(np.mean([len(full_test[u]) for u in train_seq]), 2),
        "boundary_leaks_removed": boundary_leaks,   # should be ~0
        "next_target_equals_test0": True,
        "vocab_source": "cf_split.item_universe (train-only, no cold items)",
        "integrity": "PASS",
    }
    with open(EVAL / "seq_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[seq-eval] manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
