"""
CineSemantics Day-8 (Phase 6) — frontier-model comparison + scoring.

Head-to-head on the SAME 30-user held-out sample (frontier_seeds.py):

  * specialized  ItemKNN champion (Day 3) — recommends catalog indices directly,
                 so it is 100% catalog-valid by construction.
  * Popularity   non-personalized reference (the baseline that beat semantic
                 search on Day 1).
  * frontier LLM Claude (this session), zero-shot, titles-only prompt, NO catalog
                 access. Its free-text titles are mapped back to the 9,837-movie
                 catalog by normalized title (+/-1yr) match; titles that map to
                 nothing are OFF-CATALOG (the hallucination the grounded systems
                 cannot produce).

All three are scored with the identical metric code (NDCG@10 / precision@10 /
recall@20) against each user's held-out likes, seed/seen items excluded.

Honest scope note: GPT-5.4 is named in the sprint spec as a second frontier model
but was NOT run — no OpenAI key is configured in this autonomous environment. We
report only what we actually executed (Claude). LLM latency/cost are ESTIMATES
(labelled) from typical Opus-class serving; specialized latency is measured.

Inputs:
  results/samples/frontier_seeds.json
  <scratchpad>/frontier_llm_recos.json   (this session's zero-shot output)
Outputs:
  results/frontier_comparison.csv
  results/frontier_per_user.csv
  results/samples/frontier_llm_scored.json
  results/figures/day8_frontier.png
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
SCRATCH = Path(r"C:/Users/antho/AppData/Local/Temp/claude/"
               r"C--Users-antho-OneDrive-Desktop-Mark-ALL-DATA-SCIENTIST/"
               r"f4f92c91-3174-4c75-847e-6b93c55d7829/scratchpad")

K_NDCG, K_RECALL, K_PREC = 10, 20, 10

# Opus-class serving estimate (per zero-shot reco query): ~230 in + ~260 out tok
LLM_EST_LATENCY_S = 2.1
LLM_EST_COST_USD = (230 / 1e6) * 15 + (260 / 1e6) * 75   # in $15/M, out $75/M


# ---------- metrics (identical to cf_compare / ablation) ----------
def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(ranked, relevant, k):
    rel = [1.0 if i in relevant else 0.0 for i in ranked[:k]]
    idcg = dcg([1.0] * min(len(relevant), k))
    return dcg(rel) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


# ---------- title normalization for LLM -> catalog grounding ----------
_ART = re.compile(r"^(the|a|an)\s+")


def norm_title(t):
    t = str(t).lower().strip()
    t = t.replace("&", "and")
    t = re.sub(r"[‘’“”]", "'", t)
    t = re.sub(r"[^a-z0-9']+", " ", t)
    t = _ART.sub("", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def parse_titleyear(s):
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", s.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return s.strip(), None


def build_catalog_index(catalog):
    idx = {}
    for i, row in catalog.iterrows():
        nt = norm_title(row["Title"])
        rd = str(row.get("Release_Date", ""))
        yr = int(rd[:4]) if rd[:4].isdigit() else None
        idx.setdefault(nt, []).append((int(i), yr))
    return idx


def resolve(title, year, cat_idx):
    """Map an LLM title to a catalog index, or None if off-catalog."""
    cands = cat_idx.get(norm_title(title))
    if not cands:
        return None
    if year is not None:
        # prefer exact year, then +/-1
        for tol in (0, 1):
            for ci, cy in cands:
                if cy is not None and abs(cy - year) <= tol:
                    return ci
    return cands[0][0]


def agg(per_user_ndcg, per_user_rec, per_user_prec):
    return (round(float(np.mean(per_user_ndcg)), 4),
            round(float(np.mean(per_user_rec)), 4),
            round(float(np.mean(per_user_prec)), 4))


def main():
    seeds = json.load(open(RESULTS / "samples" / "frontier_seeds.json"))
    llm = json.load(open(SCRATCH / "frontier_llm_recos.json"))
    catalog = pd.read_csv(DATA / "9000plus.csv")
    cat_idx = build_catalog_index(catalog)

    # popularity reference over the CF universe
    split = json.load(open(EVAL / "cf_split.json"))
    pop = {int(k): v for k, v in split["train_pop"].items()}
    pop_ranked = [c for c, _ in sorted(pop.items(), key=lambda x: -x[1])]

    spec_n, spec_r, spec_p = [], [], []
    pop_n, pop_r, pop_p = [], [], []
    llm_n, llm_r, llm_p = [], [], []
    total_reco = 0
    total_offcat = 0
    per_user_rows = []
    scored_samples = []

    for s in seeds["samples"]:
        u = str(s["user"])
        gt = set(x["cat"] for x in s["ground_truth"])
        seen = set()  # seeds are already excluded by both rankers; gt never in seed
        for x in s["seed"]:
            pass
        # specialized (already catalog idx, seen-excluded)
        spec = [x["cat"] for x in s["specialized_itemknn"]]
        spec_n.append(ndcg_at_k(spec, gt, K_NDCG))
        spec_r.append(recall_at_k(spec, gt, K_RECALL))
        spec_p.append(precision_at_k(spec, gt, K_PREC))
        # popularity (exclude the user's seed items)
        seed_cats = set()  # seed titles -> approx exclude via resolve
        for it in s["seed"]:
            ci = resolve(it["title"], it.get("year") and int(it["year"]), cat_idx)
            if ci is not None:
                seed_cats.add(ci)
        pr = [c for c in pop_ranked if c not in seed_cats][:50]
        pop_n.append(ndcg_at_k(pr, gt, K_NDCG))
        pop_r.append(recall_at_k(pr, gt, K_RECALL))
        pop_p.append(precision_at_k(pr, gt, K_PREC))
        # frontier LLM: resolve titles -> catalog idx (order preserved)
        recs = llm["recos"].get(u, [])
        mapped, offcat_titles = [], []
        for r in recs:
            t, y = parse_titleyear(r)
            ci = resolve(t, y, cat_idx)
            total_reco += 1
            if ci is None:
                total_offcat += 1
                offcat_titles.append(r)
                continue
            if ci in seed_cats or ci in mapped:
                continue
            mapped.append(ci)
        llm_n.append(ndcg_at_k(mapped, gt, K_NDCG))
        llm_r.append(recall_at_k(mapped, gt, K_RECALL))
        llm_p.append(precision_at_k(mapped, gt, K_PREC))

        per_user_rows.append({
            "user": u,
            "spec_ndcg@10": round(spec_n[-1], 4),
            "llm_ndcg@10": round(llm_n[-1], 4),
            "llm_offcatalog": len(offcat_titles),
            "llm_n_valid": len(mapped),
        })
        scored_samples.append({
            "user": u,
            "seed_titles": [f"{x['title']} ({x['year']})" for x in s["seed"]],
            "llm_recos": recs,
            "llm_offcatalog_titles": offcat_titles,
            "specialized_titles": [f"{x['title']} ({x['year']})"
                                   for x in s["specialized_itemknn"]],
            "held_out_titles": [f"{x['title']} ({x['year']})"
                                for x in s["ground_truth"]],
            "spec_ndcg@10": round(spec_n[-1], 4),
            "llm_ndcg@10": round(llm_n[-1], 4),
        })

    n = len(seeds["samples"])
    sn = agg(spec_n, spec_r, spec_p)
    pn = agg(pop_n, pop_r, pop_p)
    ln = agg(llm_n, llm_r, llm_p)
    halluc = round(100 * total_offcat / max(1, total_reco), 1)

    # measured specialized latency (rank a representative user)
    spec_lat_ms = _measure_spec_latency()

    rows = [
        {"system": "specialized: ItemKNN champion (Day 3)", "grounded": "yes",
         "ndcg@10": sn[0], "recall@20": sn[1], "precision@10": sn[2],
         "off_catalog_pct": 0.0, "latency_ms": round(spec_lat_ms, 3),
         "cost_usd_per_query": 0.0, "note": "catalog-grounded, personalized, measured"},
        {"system": "reference: Popularity", "grounded": "yes",
         "ndcg@10": pn[0], "recall@20": pn[1], "precision@10": pn[2],
         "off_catalog_pct": 0.0, "latency_ms": 0.01,
         "cost_usd_per_query": 0.0, "note": "non-personalized reference"},
        {"system": llm["model"], "grounded": "no",
         "ndcg@10": ln[0], "recall@20": ln[1], "precision@10": ln[2],
         "off_catalog_pct": halluc, "latency_ms": round(LLM_EST_LATENCY_S * 1000, 1),
         "cost_usd_per_query": round(LLM_EST_COST_USD, 5),
         "note": "zero-shot, no catalog access; latency/cost ESTIMATED"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "frontier_comparison.csv", index=False)
    pd.DataFrame(per_user_rows).to_csv(RESULTS / "frontier_per_user.csv", index=False)
    with open(RESULTS / "samples" / "frontier_llm_scored.json", "w") as f:
        json.dump({"n_users": n, "hallucination_pct": halluc,
                   "total_reco": total_reco, "total_offcatalog": total_offcat,
                   "samples": scored_samples}, f, indent=2)

    print(df.to_string(index=False))
    print(f"\n[frontier] n_users={n}  LLM off-catalog(hallucination)={halluc}% "
          f"({total_offcat}/{total_reco})")
    print(f"[frontier] specialized measured latency={spec_lat_ms:.3f}ms/query; "
          f"LLM est {LLM_EST_LATENCY_S*1000:.0f}ms & ${LLM_EST_COST_USD:.5f}/query "
          f"-> LLM ~{LLM_EST_LATENCY_S*1000/max(spec_lat_ms,1e-6):.0f}x slower")

    _plot(df)


def _measure_spec_latency():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.ablation import build_itemknn
    split = json.load(open(EVAL / "cf_split.json"))
    train = {int(u): v for u, v in split["train"].items()}
    universe = split["item_universe"]
    knn_rank, *_ = build_itemknn(train, universe)
    users = list(train)[:50]
    t0 = time.time()
    for u in users:
        knn_rank(train[u], set(train[u]))[:10]
    return (time.time() - t0) / len(users) * 1000


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["ItemKNN\n(grounded)", "Popularity", "Claude\nzero-shot"]
    ax1.bar(labels, df["ndcg@10"], color=["#2a7de1", "#b0b0b0", "#e0703c"])
    ax1.set_ylabel("NDCG@10 (n=30 held-out users)")
    ax1.set_title("Ranking quality")
    for i, v in enumerate(df["ndcg@10"]):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax2.bar(labels, df["off_catalog_pct"], color=["#2a7de1", "#b0b0b0", "#e0703c"])
    ax2.set_ylabel("% recommendations OFF-catalog")
    ax2.set_title("Hallucination (grounding failure)")
    for i, v in enumerate(df["off_catalog_pct"]):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS / "figures" / "day8_frontier.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
