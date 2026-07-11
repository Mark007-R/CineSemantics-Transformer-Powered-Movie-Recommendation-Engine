"""
CineSemantics Day-7 (Phase 5): sequential-transformer bake-off.

The project is named "Transformer-Powered ... Recommendation Engine", yet through
Day 6 the transformer only ever produced text/image EMBEDDINGS for retrieval; the
personalized RANKER was co-occurrence CF (ItemKNN, Day-3 champion) and matrix
factorization (tuned ALS, Day-6). Both are ORDER-BLIND -- they treat a user's
likes as a set. Day 7 adds the first transformers that model likes as an ordered
SEQUENCE and predict the next item: SASRec (causal) and BERT4Rec (Cloze).

Everything is scored on the SAME per-user temporal split as Day 3/6
(src/eval/build_sequential_eval.py, derived from cf_split.json) under TWO
protocols so the comparison is honest and complete:

  * NEXT-ITEM (leave-one-out): rank the single immediate-next held-out like.
    HR@10 (= hit rate) and NDCG@10 -- the canonical sequential-rec metric.
  * FULL-LIST (Day-3 protocol): rank all held-out likes. NDCG@10 / recall@20 /
    MAP@20 / coverage / diversity / novelty -- apples-to-apples with the CF
    leaderboard so we can say plainly whether a transformer beats item-kNN.

Also: (a) a popularity-debiasing sweep on the chosen sequential ranker (Day-6
found debiasing wrecks accuracy because held-out likes skew popular -- we verify
the same honest trade-off at the sequence level and report coverage vs accuracy);
(b) grounded "because you liked X" explanations built from the transformer's own
learned item embeddings (no hallucinated titles).

Outputs:
  results/phase5_sequential.csv        the Day-7 leaderboard (both protocols)
  results/phase5_debias.csv            debiasing sweep on the chosen ranker
  results/phase5_metrics.json          headline dict
  results/samples/phase5_explanations.json
  results/samples/phase5_seq_reco.json
  results/figures/phase5_nextitem_ndcg.png, phase5_fulllist_ndcg.png
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.recsys.cf_compare import (ndcg_at_k, recall_at_k, precision_at_k,
                                    ap_at_k, genre_set)
from src.recsys.sequential import SASRec, BERT4Rec, MarkovChain, set_seed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

K = 10
K_RECALL = 20
SEED = 42
NEG_INF = -1e9

# Day-6 tuned ALS (results/phase4_als_tuning.csv, ALS_optuna row)
ALS_PARAMS = dict(factors=27, regularization=0.04315, iterations=35, alpha=3.75)


def load_split():
    with open(EVAL / "seq_split.json") as f:
        s = json.load(f)
    universe = [int(i) for i in s["item_universe"]]
    users = [int(u) for u in s["users"]]
    train = {int(u): [int(i) for i in s["train_seq"][u]] for u in s["train_seq"]}
    nxt = {int(u): int(s["next_target"][u]) for u in s["next_target"]}
    full = {int(u): set(int(i) for i in s["full_test"][u]) for u in s["full_test"]}
    dense = {int(u): [int(i) for i in s["dense_context"][u]] for u in s["dense_context"]}
    return universe, users, train, nxt, full, dense


def evaluate(score_of_user, users, train, nxt, full, universe, col, catalog,
             genres, self_info, n_catalog, need_lists=True):
    """score_of_user(u) -> np.array over universe columns (already real-valued).
    Returns metric dict + per-user next-item and full-list NDCG lists + coverage."""
    n_items = len(universe)
    acc = {k: [] for k in ("ni_hr", "ni_ndcg", "fl_ndcg", "fl_recall", "fl_map",
                           "fl_prec", "ild", "nov")}
    cov = set()
    per_user_fl = []
    for u in users:
        row = np.array(score_of_user(u), dtype=np.float64).copy()
        seen = [col[it] for it in train[u] if it in col]
        row[seen] = NEG_INF
        order = np.argsort(-row)
        ranked_cat = [universe[c] for c in order[:max(K_RECALL, K)]]
        # next-item (single target)
        tgt = {nxt[u]}
        acc["ni_hr"].append(1.0 if nxt[u] in set(ranked_cat[:K]) else 0.0)
        acc["ni_ndcg"].append(ndcg_at_k(ranked_cat, tgt, K))
        # full-list
        rel = full[u]
        fl = ndcg_at_k(ranked_cat, rel, K)
        acc["fl_ndcg"].append(fl); per_user_fl.append(fl)
        acc["fl_recall"].append(recall_at_k(ranked_cat, rel, K_RECALL))
        acc["fl_map"].append(ap_at_k(ranked_cat, rel, K_RECALL))
        acc["fl_prec"].append(precision_at_k(ranked_cat, rel, K))
        topl_cols = order[:K]
        topl = [universe[c] for c in topl_cols]
        cov.update(topl)
        acc["nov"].append(float(np.mean(self_info[topl_cols])))
        ds = []
        for a in range(len(topl)):
            for b in range(a + 1, len(topl)):
                ga, gb = genres[topl[a]], genres[topl[b]]
                un = ga | gb
                ds.append(1 - (len(ga & gb) / len(un) if un else 0.0))
        acc["ild"].append(float(np.mean(ds)) if ds else 0.0)
    out = {
        "ni_hr@10": round(float(np.mean(acc["ni_hr"])), 4),
        "ni_ndcg@10": round(float(np.mean(acc["ni_ndcg"])), 4),
        "fl_ndcg@10": round(float(np.mean(acc["fl_ndcg"])), 4),
        "fl_recall@20": round(float(np.mean(acc["fl_recall"])), 4),
        "fl_map@20": round(float(np.mean(acc["fl_map"])), 4),
        "fl_precision@10": round(float(np.mean(acc["fl_prec"])), 4),
        "catalog_coverage": round(len(cov) / n_catalog, 4),
        "intra_list_diversity@10": round(float(np.mean(acc["ild"])), 4),
        "novelty@10": round(float(np.mean(acc["nov"])), 3),
    }
    return out, per_user_fl, acc["ni_ndcg"]


def main():
    set_seed(SEED)
    universe, users, train, nxt, full, dense = load_split()
    n_items = len(universe)
    col = {cat: j for j, cat in enumerate(universe)}          # catalog idx -> col
    item2id = {cat: j + 1 for j, cat in enumerate(universe)}  # -> model id 1..N
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n_catalog = len(catalog)
    genres = {cat: genre_set(catalog.loc[cat, "Genre"]) for cat in universe}
    users = sorted(users)
    urow = {u: i for i, u in enumerate(users)}
    print(f"[data] {len(users)} users x {n_items} items (vocab)")

    # interaction matrix (for CF systems), popularity/novelty
    rows, cols = [], []
    for u in users:
        for it in train[u]:
            rows.append(urow[u]); cols.append(col[it])
    R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                   shape=(len(users), n_items))
    train_pop = np.asarray(R.sum(axis=0)).ravel()
    pop_prob = train_pop / len(users)
    self_info = -np.log2(np.clip(pop_prob, 1e-9, None))

    # id-sequences for sequential models (train order = chronological)
    seq_ids = {u: [item2id[c] for c in train[u]] for u in users}
    dense_ids = {u: [item2id[c] for c in dense[u] if c in item2id] for u in users}

    # ---- CF score matrices (reuse Day-3/6 recipes) ----
    scores = {}
    timing = {}

    t = time.time()
    scores["Popularity"] = np.tile(train_pop, (len(users), 1)).astype(np.float32)
    timing["Popularity"] = time.time() - t

    t = time.time()
    S = cosine_similarity(R.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    scores["ItemKNN"] = (R @ S).astype(np.float32)
    timing["ItemKNN"] = time.time() - t

    t = time.time()
    from implicit.als import AlternatingLeastSquares
    als = AlternatingLeastSquares(
        factors=ALS_PARAMS["factors"], regularization=ALS_PARAMS["regularization"],
        iterations=ALS_PARAMS["iterations"], random_state=SEED, use_gpu=False)
    als.fit((R * ALS_PARAMS["alpha"]).tocsr(), show_progress=False)
    scores["ALS_tuned"] = (als.user_factors @ als.item_factors.T).astype(np.float32)
    timing["ALS_tuned"] = time.time() - t

    # Markov (order-aware baseline)
    t = time.time()
    mk = MarkovChain(n_items).fit([seq_ids[u] for u in users])
    timing["Markov"] = time.time() - t

    # ---- train transformers ----
    t = time.time()
    sas = SASRec(n_items, maxlen=50, d=64, n_blocks=2, n_heads=2, dropout=0.2,
                 epochs=120, batch_size=128, seed=SEED)
    sas.fit([seq_ids[u] for u in users], verbose=True)
    timing["SASRec"] = time.time() - t
    print(f"[train] SASRec {timing['SASRec']:.1f}s")

    t = time.time()
    bert = BERT4Rec(n_items, maxlen=50, d=64, n_blocks=2, n_heads=2, dropout=0.2,
                    epochs=160, batch_size=128, mask_prob=0.2, seed=SEED)
    bert.fit([seq_ids[u] for u in users], verbose=True)
    timing["BERT4Rec"] = time.time() - t
    print(f"[train] BERT4Rec {timing['BERT4Rec']:.1f}s")

    # SASRec trained on DENSE context (sensitivity: does denser history help?)
    t = time.time()
    sas_dense = SASRec(n_items, maxlen=50, d=64, n_blocks=2, n_heads=2,
                       dropout=0.2, epochs=120, batch_size=128, seed=SEED)
    sas_dense.fit([dense_ids[u] for u in users if len(dense_ids[u]) >= 2],
                  verbose=False)
    timing["SASRec_dense"] = time.time() - t
    print(f"[train] SASRec_dense {timing['SASRec_dense']:.1f}s")

    # per-user scorers
    def cf_scorer(name):
        M = scores[name]
        return lambda u: M[urow[u]]
    scorers = {
        "Popularity": cf_scorer("Popularity"),
        "Markov": lambda u: mk.score_all(seq_ids[u]),
        "ItemKNN": cf_scorer("ItemKNN"),
        "ALS_tuned": cf_scorer("ALS_tuned"),
        "SASRec": lambda u: sas.score_all(seq_ids[u]),
        "BERT4Rec": lambda u: bert.score_all(seq_ids[u]),
        "SASRec_dense": lambda u: sas_dense.score_all(dense_ids[u] or seq_ids[u]),
    }

    order = ["Popularity", "Markov", "ItemKNN", "ALS_tuned",
             "SASRec", "BERT4Rec", "SASRec_dense"]
    rows_out = []
    per_user_fl = {}
    per_user_ni = {}
    for name in order:
        res, fl, ni = evaluate(scorers[name], users, train, nxt, full, universe,
                               col, catalog, genres, self_info, n_catalog)
        res = {"system": name, **res, "fit_seconds": round(timing[name], 2)}
        rows_out.append(res)
        per_user_fl[name] = fl
        per_user_ni[name] = ni
        print(f"  {name:14s} NI-HR@10={res['ni_hr@10']:.4f} "
              f"NI-NDCG@10={res['ni_ndcg@10']:.4f} "
              f"FL-NDCG@10={res['fl_ndcg@10']:.4f} FL-recall@20={res['fl_recall@20']:.4f}")

    lb = pd.DataFrame(rows_out)
    lb.to_csv(RESULTS / "phase5_sequential.csv", index=False)
    print("\n" + lb.to_string(index=False))

    # ---- choose the sequential champion by next-item NDCG@10 ----
    seq_only = [r for r in rows_out if r["system"] in
                ("SASRec", "BERT4Rec", "SASRec_dense")]
    seq_champ = max(seq_only, key=lambda r: r["ni_ndcg@10"])["system"]
    champ_scorer = scorers[seq_champ]
    champ_seq = seq_ids if seq_champ != "SASRec_dense" else \
        {u: (dense_ids[u] or seq_ids[u]) for u in users}
    print(f"[choose] sequential champion (next-item NDCG@10) = {seq_champ}")

    # ---- popularity-debiasing sweep on the sequential champion ----
    debias_rows = []
    log_pop = np.log1p(train_pop)
    for beta in (0.0, 0.5, 1.0, 2.0):
        def dscorer(u, b=beta):
            raw = np.array(champ_scorer(u), dtype=np.float64)
            return raw - b * log_pop
        res, _, _ = evaluate(dscorer, users, train, nxt, full, universe, col,
                             catalog, genres, self_info, n_catalog)
        debias_rows.append({"beta": beta, "ranker": seq_champ,
                            "fl_ndcg@10": res["fl_ndcg@10"],
                            "ni_ndcg@10": res["ni_ndcg@10"],
                            "catalog_coverage": res["catalog_coverage"],
                            "novelty@10": res["novelty@10"]})
        print(f"  debias beta={beta}: FL-NDCG@10={res['fl_ndcg@10']:.4f} "
              f"cov={res['catalog_coverage']:.4f} nov={res['novelty@10']:.2f}")
    pd.DataFrame(debias_rows).to_csv(RESULTS / "phase5_debias.csv", index=False)

    # ---- significance: SASRec vs ItemKNN (paired Wilcoxon on per-user FL-NDCG) ----
    sig = {}
    try:
        from scipy.stats import wilcoxon
        for a, b in (("SASRec", "ItemKNN"), ("BERT4Rec", "ItemKNN"),
                     ("SASRec_dense", "SASRec"), ("SASRec", "Markov")):
            if any(x != y for x, y in zip(per_user_fl[a], per_user_fl[b])):
                stat, p = wilcoxon(per_user_fl[a], per_user_fl[b])
                sig[f"{a}_vs_{b}_fl_ndcg_p"] = round(float(p), 5)
        for a, b in (("SASRec", "ItemKNN"), ("BERT4Rec", "ItemKNN")):
            stat, p = wilcoxon(per_user_ni[a], per_user_ni[b])
            sig[f"{a}_vs_{b}_nextitem_ndcg_p"] = round(float(p), 5)
    except Exception as e:
        sig = {"error": str(e)}

    # ---- grounded "because you liked X" explanations (from learned item vecs) ----
    iv = sas.item_vectors()               # (n_items, d), rows aligned to col
    ivn = iv / (np.linalg.norm(iv, axis=1, keepdims=True) + 1e-9)
    def explain(u, rec_col, k_hist=2):
        hist_cols = [col[it] for it in train[u] if it in col]
        if not hist_cols:
            return []
        sims = ivn[hist_cols] @ ivn[rec_col]
        top = np.argsort(-sims)[:k_hist]
        return [str(catalog.loc[universe[hist_cols[i]], "Title"]) for i in top]

    explanations = []
    for u in users[:5]:
        raw = np.array(champ_scorer(u), dtype=np.float64).copy()
        raw[[col[it] for it in train[u] if it in col]] = NEG_INF
        top_cols = np.argsort(-raw)[:3]
        recs = []
        for c in top_cols:
            because = explain(u, c)
            recs.append({
                "title": str(catalog.loc[universe[c], "Title"]),
                "genre": str(catalog.loc[universe[c], "Genre"]),
                "hit": bool(universe[c] in full[u]),
                "because_you_liked": because,
                "explanation": ("Because you liked " + " and ".join(because)
                                if because else "Popular with similar viewers"),
            })
        explanations.append({
            "user": u, "ranker": seq_champ,
            "recent_likes": [str(catalog.loc[c, "Title"]) for c in train[u][-4:]],
            "recommendations": recs,
        })
    with open(RESULTS / "samples" / "phase5_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    # sample recos (next-item hit inspection) for a few users, per seq model
    seq_samples = {}
    for name in ("SASRec", "BERT4Rec"):
        srows = []
        for u in users[:4]:
            raw = np.array(scorers[name](u), dtype=np.float64).copy()
            raw[[col[it] for it in train[u] if it in col]] = NEG_INF
            top = np.argsort(-raw)[:10]
            srows.append({
                "user": u,
                "history_tail": [str(catalog.loc[c, "Title"]) for c in train[u][-5:]],
                "next_target": str(catalog.loc[nxt[u], "Title"]),
                "next_hit@10": bool(nxt[u] in [universe[c] for c in top]),
                "top10": [{"title": str(catalog.loc[universe[c], "Title"]),
                           "in_full_test": bool(universe[c] in full[u])} for c in top],
            })
        seq_samples[name] = srows
    with open(RESULTS / "samples" / "phase5_seq_reco.json", "w") as f:
        json.dump(seq_samples, f, indent=2)

    # ---- headline metrics ----
    headline = {
        "day": 7, "project": "CineSemantics", "phase": "Phase 5 - sequential transformers",
        "date": "2026-07-11",
        "eval": {
            "protocol_next_item": "leave-one-out on first held-out like (HR@10/NDCG@10)",
            "protocol_full_list": "Day-3 temporal leave-last-20% (NDCG@10/recall@20/MAP)",
            "n_users": len(users), "n_items_vocab": n_items,
        },
        "leaderboard": {r["system"]: {k: r[k] for k in r if k != "system"}
                        for r in rows_out},
        "sequential_champion": seq_champ,
        "debias_sweep": debias_rows,
        "significance": sig,
        "day3_reference": {"ItemKNN_fl_ndcg@10": 0.1059},
        "model_params": {"SASRec": sas.p, "BERT4Rec": bert.p, "ALS": ALS_PARAMS},
    }
    with open(RESULTS / "phase5_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = lb.copy()
        seqset = {"SASRec", "BERT4Rec", "SASRec_dense"}
        for metric, fname, title in (
            ("ni_ndcg@10", "phase5_nextitem_ndcg.png", "Next-item NDCG@10 (leave-one-out)"),
            ("fl_ndcg@10", "phase5_fulllist_ndcg.png", "Full-list NDCG@10 (Day-3 protocol)")):
            dd = d.sort_values(metric)
            fig, ax = plt.subplots(figsize=(7.6, 4.4))
            colors = ["#8e44ad" if s in seqset else "#9bb7d4" for s in dd["system"]]
            bars = ax.barh(dd["system"], dd[metric], color=colors)
            if metric == "fl_ndcg@10":
                ax.axvline(0.1059, color="#c0392b", ls="--", lw=1)
                ax.text(0.1059, -0.4, "Day-3 ItemKNN 0.106", color="#c0392b", fontsize=8)
            ax.set_xlabel(metric)
            ax.set_title(f"CineSemantics Day-7: {title}\n{len(users)} users, "
                         "purple = transformer")
            for b, v in zip(bars, dd[metric]):
                ax.text(v + max(dd[metric]) * 0.01, b.get_y() + b.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(RESULTS / "figures" / fname, dpi=130)
            print(f"[fig] saved {fname}")
    except Exception as e:
        print(f"[fig] skipped: {e}")

    print(f"\n[done] sequential champion = {seq_champ}; wrote phase5_sequential.csv")
    return headline


if __name__ == "__main__":
    main()
