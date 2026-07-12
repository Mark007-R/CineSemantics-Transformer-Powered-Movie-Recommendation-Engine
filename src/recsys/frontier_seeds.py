"""
CineSemantics Day-8 (Phase 6) — build the frontier-comparison user sample.

We deterministically sample N eval users from the Day-3 temporal held-out split and
emit, for each: the most-recent K liked movie TITLES (the prompt a zero-shot LLM
sees — titles only, no catalog, no ratings) and the ItemKNN champion's top-10
recommendation TITLES for the same user. Ground-truth held-out titles stay in a
separate scored file so the LLM prompt cannot see them.

This lets us run an honest head-to-head:
  * specialized (ItemKNN, grounded in the catalog) vs
  * a frontier LLM asked "recommend movies for someone who liked X, Y, Z"
and measure NDCG@10 / precision@10 / recall@20 on the identical held-out set,
PLUS the LLM-only failure mode the specialized system cannot have: recommending
titles that are not in the 9,837-movie catalog at all (hallucination rate).

Outputs:
  results/samples/frontier_seeds.json   (seeds + specialized recos + gt, for scoring)
  <scratchpad>/frontier_prompt.txt       (human-readable prompt block, reference)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.ablation import build_itemknn  # reuse the exact champion ranker

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)

N_USERS = 30
K_SEED = 10          # most-recent liked titles shown to the LLM
K_SPEC = 10          # specialized recos captured for the head-to-head
SEED = 42


def title_year(catalog, idx):
    row = catalog.loc[idx]
    rd = str(row.get("Release_Date", ""))
    yr = rd[:4] if len(rd) >= 4 else ""
    return str(row.get("Title", "")).strip(), yr


def main():
    split = json.load(open(EVAL / "cf_split.json"))
    train = {int(u): v for u, v in split["train"].items()}
    test = {int(u): v for u, v in split["test"].items()}
    universe = split["item_universe"]
    catalog = pd.read_csv(DATA / "9000plus.csv")

    # deterministic sample of users with enough history to profile
    rng = np.random.RandomState(SEED)
    eligible = [u for u in sorted(test) if len(train[u]) >= 8 and len(test[u]) >= 2]
    chosen = sorted(rng.choice(eligible, size=min(N_USERS, len(eligible)),
                               replace=False).tolist())

    knn_rank, *_ = build_itemknn(train, universe)

    samples = []
    prompt_lines = []
    for u in chosen:
        # seed = most-recent K liked (train is timestamp-sorted, last = most recent)
        seed_idx = train[u][-K_SEED:]
        seed = [{"title": t, "year": y} for t, y in
                (title_year(catalog, i) for i in seed_idx)]
        # specialized champion recos (catalog-grounded)
        spec = knn_rank(train[u], set(train[u]))[:K_SPEC]
        spec_titles = [{"title": t, "year": y, "cat": int(i)} for i, (t, y) in
                       ((i, title_year(catalog, i)) for i in spec)]
        gt = [{"title": t, "year": y, "cat": int(i)} for i, (t, y) in
              ((i, title_year(catalog, i)) for i in test[u])]
        samples.append({
            "user": int(u),
            "seed": seed,
            "specialized_itemknn": spec_titles,
            "ground_truth": gt,
        })
        titles = ", ".join(f"{s['title']} ({s['year']})" for s in seed)
        prompt_lines.append(f"user {u}: {titles}")

    out = {
        "protocol": "Day-3 temporal leave-last-20% split; frontier sample",
        "n_users": len(samples),
        "k_seed": K_SEED,
        "catalog_size": len(catalog),
        "seed": SEED,
        "samples": samples,
    }
    with open(RESULTS / "samples" / "frontier_seeds.json", "w") as f:
        json.dump(out, f, indent=2)

    scratch = Path(r"C:/Users/antho/AppData/Local/Temp/claude/"
                   r"C--Users-antho-OneDrive-Desktop-Mark-ALL-DATA-SCIENTIST/"
                   r"f4f92c91-3174-4c75-847e-6b93c55d7829/scratchpad")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "frontier_prompt.txt").write_text("\n".join(prompt_lines),
                                                 encoding="utf-8")
    print(f"[frontier-seeds] {len(samples)} users -> "
          f"results/samples/frontier_seeds.json")
    print("\n".join(prompt_lines))


if __name__ == "__main__":
    main()
