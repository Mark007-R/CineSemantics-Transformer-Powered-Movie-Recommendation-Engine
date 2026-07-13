"""
CineSemantics Day-7 (Phase 5) sequential-recommendation eval builder.

Day 3 built a per-user TEMPORAL leave-last-20% split (`data/eval/cf_split.json`)
whose train lists are already chronologically ordered (build_cf_eval.py sorts
each user's likes by MovieLens `timestamp` before splitting). A sequential
recommender needs exactly that: an ordered history per user plus a temporally
later held-out set. So Day 7 derives its eval DIRECTLY from `cf_split.json` --
there is ZERO split drift between the CF bake-off (Day 3/6) and the sequential
transformers (Day 7); they are scored on the identical users, items, and
train/test partition. Rule 10 (no test interaction leaks into training) is
inherited unchanged and re-asserted here.

Two protocols are emitted so a transformer's *native* task and the running
full-list leaderboard are both measurable on the same split:

  * next_item  -- relevance = {the user's FIRST held-out item} (the item that
                  immediately follows training in time). This is the standard
                  leave-one-out next-item task self-attention models are built
                  for (SASRec / BERT4Rec).
  * full_list  -- relevance = ALL held-out items. Identical to the Day-3 CF
                  protocol, so Day-7 numbers drop straight onto the leaderboard.

Output: data/eval/seq_eval.json  { train, next_item, full_list, item_universe,
                                    params, integrity } and a printed manifest.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"


def build():
    with open(EVAL / "cf_split.json") as f:
        split = json.load(f)

    train = {int(u): [int(x) for x in v] for u, v in split["train"].items()}
    test = {int(u): [int(x) for x in v] for u, v in split["test"].items()}
    universe = [int(x) for x in split["item_universe"]]

    # next-item target = the chronologically FIRST held-out like (test is stored
    # in timestamp order by build_cf_eval, most-recent-N at the tail, in order).
    next_item = {u: test[u][0] for u in train}
    full_list = {u: test[u] for u in train}

    # integrity: train/test disjoint, every next-item target is strictly held out
    overlap = sum(len(set(train[u]) & set(test[u])) for u in train)
    ni_leak = sum(1 for u in train if next_item[u] in set(train[u]))
    assert overlap == 0, "train/test overlap -- leakage!"
    assert ni_leak == 0, "next-item target found in train -- leakage!"

    lens = np.array([len(v) for v in train.values()])
    out = {
        "train": {str(u): v for u, v in train.items()},
        "next_item": {str(u): v for u, v in next_item.items()},
        "full_list": {str(u): v for u, v in full_list.items()},
        "item_universe": universe,
        "params": {
            "derived_from": "cf_split.json (Day-3 temporal leave-last-20%)",
            "n_users": len(train),
            "n_items": len(universe),
            "avg_seq_len": round(float(lens.mean()), 2),
            "min_seq_len": int(lens.min()),
            "max_seq_len": int(lens.max()),
        },
        "integrity": {
            "train_test_overlap": int(overlap),
            "next_item_in_train": int(ni_leak),
            "status": "PASS" if overlap == 0 and ni_leak == 0 else "FAIL-LEAK",
        },
    }
    with open(EVAL / "seq_eval.json", "w") as f:
        json.dump(out, f)
    print("[seq-eval] manifest:", json.dumps(
        {**out["params"], **out["integrity"]}, indent=2))
    return out


if __name__ == "__main__":
    build()
