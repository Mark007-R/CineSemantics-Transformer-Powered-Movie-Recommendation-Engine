"""
CineSemantics Day-3 (Phase 2b) collaborative-filtering eval builder.

Day-1 built a MODEL-AGNOSTIC content "more like this" relevance set (item->item
co-rating). That set cannot measure PERSONALIZATION, because there is no user
model in it. Day 3 adds the missing piece: a PER-USER, TEMPORAL held-out split of
the MovieLens interactions so we can score real recommenders (ALS / item-kNN /
SVD / content / hybrid) with precision/recall/NDCG/MAP/coverage/novelty.

Protocol (leave-last-N, temporal, per user):
  * interactions = MovieLens ratings aligned to the 9837-movie TMDB catalog
    (via data/eval/movielens_alignment.csv, same alignment as Day 1).
  * a "like" = rating >= LIKE_THRESHOLD (4.0), matching Day-1's positive signal.
  * for every user with >= MIN_LIKES likes, sort likes by timestamp and hold out
    the most recent TEST_FRAC (>=1) as the test set; the earlier likes are train.
  * this guarantees every test interaction is strictly LATER than that user's
    training interactions -> no future leaks into the past (temporal integrity),
    and train and test never overlap. Rule 10: no test interaction leaks into
    training.

The candidate item universe for recommendation is the set of items that appear
in TRAINING (an item with zero training interactions is a cold item that CF
cannot score). All systems rank over this shared universe so the comparison is
apples-to-apples; coverage/novelty use it as the denominator.

Outputs (data/eval/):
  - cf_split.json     {train: {user: [items]}, test: {user: [items]},
                       train_pop: {item: count}, item_universe: [...], params}
  - cf_manifest.json  counts + integrity checks
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"

LIKE_THRESHOLD = 4.0     # rating >= this is a positive (matches Day-1)
MIN_LIKES = 10           # user needs this many likes to be evaluable
TEST_FRAC = 0.2          # hold out the most-recent 20% of each user's likes
MIN_TEST = 1
MIN_TRAIN = 4            # keep enough history to profile the user


def build():
    align = pd.read_csv(EVAL / "movielens_alignment.csv")
    ml_to_cat = dict(zip(align["movieId"], align["catalog_index"]))

    ratings = pd.read_csv(EVAL / "ml-latest-small" / "ratings.csv")
    ratings = ratings[ratings["movieId"].isin(ml_to_cat)].copy()
    ratings["cat"] = ratings["movieId"].map(ml_to_cat).astype(int)

    likes = ratings[ratings["rating"] >= LIKE_THRESHOLD].copy()
    # a user may have rated the same catalog index via two movieIds; keep earliest
    likes = likes.sort_values("timestamp")

    train = {}
    test = {}
    skipped = 0
    for uid, grp in likes.groupby("userId"):
        items, seen = [], set()
        for cat, ts in zip(grp["cat"], grp["timestamp"]):
            if cat in seen:
                continue
            seen.add(cat)
            items.append(int(cat))          # already timestamp-sorted
        if len(items) < MIN_LIKES:
            skipped += 1
            continue
        n_test = max(MIN_TEST, int(round(len(items) * TEST_FRAC)))
        n_test = min(n_test, len(items) - MIN_TRAIN)
        if n_test < MIN_TEST:
            skipped += 1
            continue
        tr, te = items[:-n_test], items[-n_test:]
        train[int(uid)] = tr
        test[int(uid)] = te

    # candidate universe = items appearing in ANY user's training set
    train_pop = defaultdict(int)
    for items in train.values():
        for it in items:
            train_pop[it] += 1
    universe = sorted(train_pop)

    # integrity checks
    overlap = sum(len(set(train[u]) & set(test[u])) for u in train)
    test_in_universe = sum(1 for u in test for it in test[u] if it in train_pop)
    n_test_items = sum(len(v) for v in test.values())

    split = {
        "train": {str(u): v for u, v in train.items()},
        "test": {str(u): v for u, v in test.items()},
        "train_pop": {str(k): v for k, v in train_pop.items()},
        "item_universe": universe,
        "params": {
            "LIKE_THRESHOLD": LIKE_THRESHOLD, "MIN_LIKES": MIN_LIKES,
            "TEST_FRAC": TEST_FRAC, "MIN_TEST": MIN_TEST, "MIN_TRAIN": MIN_TRAIN,
        },
    }
    with open(EVAL / "cf_split.json", "w") as f:
        json.dump(split, f)

    manifest = {
        "n_users_eval": len(train),
        "n_users_skipped": skipped,
        "n_train_interactions": int(sum(len(v) for v in train.values())),
        "n_test_interactions": n_test_items,
        "candidate_items": len(universe),
        "avg_train_per_user": round(np.mean([len(v) for v in train.values()]), 2),
        "avg_test_per_user": round(np.mean([len(v) for v in test.values()]), 2),
        "train_test_overlap": int(overlap),                 # MUST be 0
        "test_items_in_train_universe_pct": round(
            100 * test_in_universe / max(1, n_test_items), 1),
        "integrity": "PASS" if overlap == 0 else "FAIL-LEAK",
    }
    with open(EVAL / "cf_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[cf-eval] manifest:", json.dumps(manifest, indent=2))
    assert overlap == 0, "train/test overlap detected -- leakage!"


if __name__ == "__main__":
    build()
