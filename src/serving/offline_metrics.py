"""
Live offline-metrics panel (Day-9 Phase 7).

The Day-1 audit's core finding was that this "recommendation engine" shipped with
ZERO evaluation. The production wrapper closes that loop visibly: the served app and
the Streamlit "Recommended for you" tab both display the sprint's honest offline
metrics, sourced directly from the persisted `results/` artifacts (no hardcoded
numbers). If a future run re-evaluates, the panel updates itself.

`panel()` returns a compact, display-ready dict: the champion ranker, the running
NDCG@10 leaderboard, and the headline Day-7/Day-8 findings (sequential parity,
frontier grounding tax) when those results are present.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _load_json(name: str):
    p = RESULTS / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _leaderboard():
    p = RESULTS / "leaderboard.csv"
    if not p.exists():
        return []
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = []
    for r in rows:
        try:
            out.append({"day": int(r["day"]), "system": r["system"],
                        "ndcg@10": float(r["ndcg@10"]),
                        "recall@20": float(r.get("recall@20", "nan")),
                        "coverage": float(r.get("catalog_coverage", "nan"))})
        except Exception:
            continue
    return sorted(out, key=lambda x: -x["ndcg@10"])


def panel() -> dict:
    cf_meta = _load_json_file(ROOT / "models" / "cf_meta.json") or {}
    lb = _leaderboard()
    champion = cf_meta.get("champion", "ItemKNN (item-item cosine)")
    out = {
        "champion_ranker": champion,
        "champion_ndcg@10": cf_meta.get("day3_ndcg@10"),
        "eval_protocol": "per-user temporal leave-last-20% (MovieLens, held-out)",
        "catalog_size": cf_meta.get("n_items"),
        "n_users_eval": cf_meta.get("n_users"),
        "leaderboard_top": lb[:6],
    }
    # Day-7 sequential headline (parity, not a win) if present
    seq = _load_json("phase5_metrics.json")
    if seq:
        sc = seq.get("sequential_champion")
        out["sequential"] = {
            "champion": sc,
            "nextitem_ndcg@10": (seq.get("nextitem", {}).get(sc, {}) or {}).get("ndcg@10"),
            "fulllist_ndcg@10": (seq.get("fulllist", {}).get(sc, {}) or {}).get("ndcg@10"),
            "note": "transformer reaches parity with CF on next-item, not a full-list win",
        }
    # Day-8 frontier headline (grounding tax) if present
    fr = _load_json("phase6_metrics.json")
    if fr:
        out["frontier"] = {k: fr.get(k) for k in
                           ("hallucination_rate", "latency_ms_llm",
                            "cost_per_query_usd", "ndcg_itemknn", "ndcg_llm")
                           if k in fr} or {"note": "frontier comparison present"}
    return out


def _load_json_file(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
