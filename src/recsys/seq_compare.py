"""
CineSemantics Day-7 (Phase 5): sequential-transformer bake-off.

Question the day answers: the repo is a "*Transformer*-Powered Recommendation
Engine," but the ranker was always order-blind (ItemKNN / ALS). Does adding a real
sequence transformer -- SASRec (causal) or BERT4Rec (Cloze) -- that models likes
as an ORDERED trajectory actually beat the Day-3 CF champion?

Everything is scored on the SAME Day-3 temporal split (via seq_eval.py, derived
from cf_split.json -> zero drift) under two protocols:
  * next_item : relevance = the single chronologically-next held-out like
                (the transformers' native leave-one-out task).
  * full_list : relevance = ALL held-out likes (the Day-3 CF protocol, so results
                drop straight onto the running leaderboard).

Systems: Popularity, Markov(1st-order), ItemKNN (Day-3 champ), ALS-tuned (Day-6
params), SASRec, BERT4Rec, SASRec+dense-context (ablation: feed ALL ratings, not
just >=4* likes -> does more history help?).

Also: popularity debiasing on the sequential champion (does it help here?), and
GROUNDED "because you liked X" explanation cards (every referenced title is a real
catalog entry attributed via item-item similarity -- no hallucination; LLM phrasing
is a drop-in surface over this grounding).

Outputs: results/phase5_sequential.csv, results/phase5_debias.csv,
results/phase5_metrics.json, results/samples/phase5_{seq_reco,explanations}.json,
results/figures/phase5_{nextitem,fulllist}_ndcg.png
"""
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.recsys.sequential import (train_sasrec, train_bert4rec,   # noqa: E402
                                   score_all_items)

DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

K_NDCG, K_RECALL, K_MAP, K_PREC, K_LIST = 10, 20, 20, 10, 10
SEED = 42
NEG = -1e9

# Day-6 tuned ALS (results/phase4_metrics.json -> hpo.best_params)
ALS_TUNED = dict(factors=27, regularization=0.04315, iterations=35, alpha=3.75138)


# ---------- metrics (identical to cf_compare) ----------
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


def hr_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def ap_at_k(ranked, relevant, k):
    hits, s = 0, 0.0
    for i, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            s += hits / (i + 1)
    denom = min(len(relevant), k)
    return s / denom if denom else 0.0


def genre_set(g):
    return set(x.strip().lower() for x in re.split(r"[,/|]", str(g)) if x.strip())


def main():
    t0 = time.time()
    with open(EVAL / "seq_eval.json") as f:
        se = json.load(f)
    train = {int(u): [int(x) for x in v] for u, v in se["train"].items()}
    next_item = {int(u): int(v) for u, v in se["next_item"].items()}
    full_list = {int(u): set(int(x) for x in v) for u, v in se["full_list"].items()}
    universe = [int(x) for x in se["item_universe"]]          # catalog indices
    users = sorted(train)
    n_users, n_items = len(users), len(universe)
    col = {cat: j for j, cat in enumerate(universe)}          # catalog -> column
    seqcol = {cat: j + 1 for j, cat in enumerate(universe)}   # catalog -> seq id (1..n)
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    n_catalog = len(catalog)
    genres = {cat: genre_set(catalog.loc[cat, "Genre"]) for cat in universe}
    print(f"[data] {n_users} users x {n_items} items; "
          f"avg seq {np.mean([len(train[u]) for u in users]):.1f}")

    # ---- interaction matrix (train, binary) for the CF baselines ----
    rows, cols = [], []
    for i, u in enumerate(users):
        for it in train[u]:
            rows.append(i); cols.append(col[it])
    R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                   shape=(n_users, n_items))
    train_pop = np.asarray(R.sum(axis=0)).ravel()
    pop_prob = train_pop / n_users
    self_info = -np.log2(np.clip(pop_prob, 1e-9, None))
    seen_cols = {u: [col[it] for it in train[u]] for u in users}

    scores, timing = {}, {}

    # 1. Popularity
    t = time.time()
    scores["Popularity"] = np.tile(train_pop, (n_users, 1)).astype(np.float32)
    timing["Popularity"] = time.time() - t

    # 2. Markov (1st-order transition from consecutive train likes)
    t = time.time()
    T = np.zeros((n_items, n_items), np.float32)
    for u in users:
        s = [col[it] for it in train[u]]
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1.0
    T += 1e-6                                                 # smoothing
    T /= T.sum(axis=1, keepdims=True)
    mk = np.zeros((n_users, n_items), np.float32)
    for i, u in enumerate(users):
        mk[i] = T[col[train[u][-1]]]
    scores["Markov"] = mk
    timing["Markov"] = time.time() - t

    # 3. ItemKNN (Day-3 champion)
    t = time.time()
    S = cosine_similarity(R.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    scores["ItemKNN"] = (R @ S).astype(np.float32)
    timing["ItemKNN"] = time.time() - t

    # 4. ALS tuned (Day-6 params)
    t = time.time()
    from implicit.als import AlternatingLeastSquares
    als = AlternatingLeastSquares(
        factors=ALS_TUNED["factors"], regularization=ALS_TUNED["regularization"],
        iterations=ALS_TUNED["iterations"], random_state=SEED, use_gpu=False)
    als.fit((R * ALS_TUNED["alpha"]).tocsr(), show_progress=False)
    scores["ALS-tuned"] = (als.user_factors @ als.item_factors.T).astype(np.float32)
    timing["ALS-tuned"] = time.time() - t

    # ---- sequential models (seq ids 1..n_items) ----
    like_seqs = [[seqcol[it] for it in train[u]] for u in users]

    t = time.time()
    print("[SASRec] training ...")
    sas, sas_inner = train_sasrec(like_seqs, n_items, seed=SEED)
    scores["SASRec"] = score_all_items(sas, like_seqs)
    timing["SASRec"] = time.time() - t

    t = time.time()
    print("[BERT4Rec] training ...")
    b4r, b4r_inner = train_bert4rec(like_seqs, n_items, seed=SEED)
    scores["BERT4Rec"] = score_all_items(b4r, like_seqs, append_mask=True)
    timing["BERT4Rec"] = time.time() - t

    # 7. SASRec + dense context: feed ALL ratings (incl. <4*), not just likes.
    #    Ablation -- does more history help or does lukewarm signal dilute taste?
    t = time.time()
    dense_seqs = _dense_context_seqs(train, universe, seqcol)
    print("[SASRec+dense] training ...")
    sasd, _ = train_sasrec(dense_seqs, n_items, seed=SEED)
    scores["SASRec+dense"] = score_all_items(sasd, dense_seqs)
    timing["SASRec+dense"] = time.time() - t

    order = ["Popularity", "Markov", "ItemKNN", "ALS-tuned",
             "SASRec", "BERT4Rec", "SASRec+dense"]

    # ---- evaluate both protocols ----
    def evaluate(rel_map, k_ndcg=K_NDCG):
        rowsout, per_user_ndcg, coverage = [], {}, {}
        for s in order:
            M = scores[s]
            acc = {m: [] for m in
                   ["ndcg", "hr", "recall", "map", "prec", "ild", "nov"]}
            cov = set()
            for i, u in enumerate(users):
                row = M[i].copy()
                row[seen_cols[u]] = NEG
                ranked_cols = np.argsort(-row)[:max(K_RECALL, K_LIST)]
                ranked = [universe[c] for c in ranked_cols]
                rel = rel_map[u] if isinstance(rel_map[u], set) else {rel_map[u]}
                acc["ndcg"].append(ndcg_at_k(ranked, rel, k_ndcg))
                acc["hr"].append(hr_at_k(ranked, rel, K_LIST))
                acc["recall"].append(recall_at_k(ranked, rel, K_RECALL))
                acc["map"].append(ap_at_k(ranked, rel, K_MAP))
                acc["prec"].append(precision_at_k(ranked, rel, K_PREC))
                topl_cols = ranked_cols[:K_LIST]
                cov.update(universe[c] for c in topl_cols)
                acc["nov"].append(float(np.mean(self_info[topl_cols])))
                topl = [universe[c] for c in topl_cols]
                ds = []
                for a in range(len(topl)):
                    for bb in range(a + 1, len(topl)):
                        ga, gb = genres[topl[a]], genres[topl[bb]]
                        un = ga | gb
                        ds.append(1 - (len(ga & gb) / len(un) if un else 0.0))
                acc["ild"].append(float(np.mean(ds)) if ds else 0.0)
            per_user_ndcg[s] = acc["ndcg"]
            coverage[s] = cov
            rowsout.append({
                "system": s,
                "ndcg@10": round(float(np.mean(acc["ndcg"])), 4),
                "hr@10": round(float(np.mean(acc["hr"])), 4),
                "recall@20": round(float(np.mean(acc["recall"])), 4),
                "map@20": round(float(np.mean(acc["map"])), 4),
                "precision@10": round(float(np.mean(acc["prec"])), 4),
                "catalog_coverage": round(len(cov) / n_catalog, 4),
                "intra_list_diversity@10": round(float(np.mean(acc["ild"])), 4),
                "novelty@10": round(float(np.mean(acc["nov"])), 3),
                "fit_seconds": round(timing[s], 2),
            })
        return rowsout, per_user_ndcg

    ni_rows, ni_ndcg = evaluate(next_item)
    fl_rows, fl_ndcg = evaluate(full_list)
    ni = pd.DataFrame(ni_rows); fl = pd.DataFrame(fl_rows)

    # merged headline table: both protocols side by side
    merged = pd.DataFrame({
        "system": order,
        "nextitem_ndcg@10": [r["ndcg@10"] for r in ni_rows],
        "nextitem_hr@10": [r["hr@10"] for r in ni_rows],
        "fulllist_ndcg@10": [r["ndcg@10"] for r in fl_rows],
        "fulllist_recall@20": [r["recall@20"] for r in fl_rows],
        "catalog_coverage": [r["catalog_coverage"] for r in fl_rows],
        "novelty@10": [r["novelty@10"] for r in fl_rows],
        "fit_seconds": [r["fit_seconds"] for r in fl_rows],
    }).sort_values("nextitem_ndcg@10", ascending=False)
    merged.to_csv(RESULTS / "phase5_sequential.csv", index=False)
    print("\n=== next-item + full-list ===\n" + merged.to_string(index=False))

    # significance: sequential champion (best next-item) vs ItemKNN
    from scipy.stats import wilcoxon
    seq_champ = max(["SASRec", "BERT4Rec"],
                    key=lambda s: np.mean(ni_ndcg[s]))
    sig = {}
    for s in ["SASRec", "BERT4Rec", "SASRec+dense", "Markov"]:
        try:
            _, p_ni = wilcoxon(ni_ndcg[s], ni_ndcg["ItemKNN"])
            _, p_fl = wilcoxon(fl_ndcg[s], fl_ndcg["ItemKNN"])
            sig[s] = {"nextitem_vs_itemknn_p": round(float(p_ni), 4),
                      "fulllist_vs_itemknn_p": round(float(p_fl), 4)}
        except Exception as e:
            sig[s] = {"error": str(e)}
    # SASRec vs Markov -> is higher-order context real signal?
    try:
        _, p_sm = wilcoxon(ni_ndcg["SASRec"], ni_ndcg["Markov"])
        sig["SASRec_vs_Markov_nextitem_p"] = round(float(p_sm), 4)
    except Exception:
        pass

    # ---- popularity debiasing on the sequential champion (full-list) ----
    champ_M = scores[seq_champ]
    logpop = np.log1p(train_pop)
    debias_rows = []
    for beta in [0.0, 0.25, 0.5, 1.0, 2.0]:
        nd = []
        cov = set()
        for i, u in enumerate(users):
            row = champ_M[i].copy() - beta * logpop
            row[seen_cols[u]] = NEG
            ranked_cols = np.argsort(-row)[:K_RECALL]
            ranked = [universe[c] for c in ranked_cols]
            nd.append(ndcg_at_k(ranked, full_list[u], K_NDCG))
            cov.update(universe[c] for c in ranked_cols[:K_LIST])
        debias_rows.append({"system": seq_champ, "beta": beta,
                            "fulllist_ndcg@10": round(float(np.mean(nd)), 4),
                            "catalog_coverage": round(len(cov) / n_catalog, 4)})
    pd.DataFrame(debias_rows).to_csv(RESULTS / "phase5_debias.csv", index=False)

    # ---- grounded "because you liked X" explanations (champion, sample users) ----
    explanations = _grounded_explanations(
        seq_champ, scores[seq_champ], users, train, full_list, universe, col, S,
        catalog, n=6)
    with open(RESULTS / "samples" / "phase5_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    # ---- per-user sample recos for the two transformers ----
    samples = {}
    for s in ["SASRec", "BERT4Rec"]:
        srows = []
        M = scores[s]
        for i, u in enumerate(users[:3]):
            row = M[i].copy(); row[seen_cols[u]] = NEG
            top = np.argsort(-row)[:10]
            srows.append({
                "user": u,
                "liked_train_tail": [catalog.loc[c, "Title"] for c in train[u][-5:]],
                "next_item_target": catalog.loc[next_item[u], "Title"],
                "top10": [{"title": catalog.loc[universe[c], "Title"],
                           "genre": catalog.loc[universe[c], "Genre"],
                           "hit_fulllist": bool(universe[c] in full_list[u]),
                           "hit_nextitem": bool(universe[c] == next_item[u])}
                          for c in top],
            })
        samples[s] = srows
    with open(RESULTS / "samples" / "phase5_seq_reco.json", "w") as f:
        json.dump(samples, f, indent=2)

    headline = {
        "day": 7, "project": "CineSemantics",
        "phase": "Phase 5 - sequential transformers (SASRec / BERT4Rec)",
        "date": "2026-07-13",
        "eval": {
            "protocol": "Day-3 temporal split (seq_eval from cf_split); "
                        "next-item leave-one-out + full-list",
            "n_users": n_users, "candidate_items": n_items,
        },
        "inner_val_ndcg": {"SASRec": round(float(sas_inner), 4),
                           "BERT4Rec": round(float(b4r_inner), 4)},
        "nextitem": {r["system"]: {k: r[k] for k in r if k != "system"}
                     for r in ni_rows},
        "fulllist": {r["system"]: {k: r[k] for k in r if k != "system"}
                     for r in fl_rows},
        "sequential_champion": seq_champ,
        "significance": sig,
        "debiasing": debias_rows,
        "cf_reference": {"ItemKNN_fulllist_ndcg@10": 0.1059,
                         "ALS_tuned_fulllist_ndcg@10": 0.1058},
    }
    with open(RESULTS / "phase5_metrics.json", "w") as f:
        json.dump(headline, f, indent=2)

    # ---- figures ----
    _figures(ni, fl, seq_champ)

    # ---- append to running leaderboard ----
    _append_leaderboard(fl_rows)

    print(f"\n[done] seq champion (next-item) = {seq_champ}; "
          f"sig vs ItemKNN = {sig.get(seq_champ)}; total {time.time()-t0:.1f}s")


def _dense_context_seqs(train, universe, seqcol):
    """Feed ALL of a user's ratings (not just >=4* likes) as context, then predict.
    Reads raw MovieLens; falls back to the like-only sequences if unavailable."""
    try:
        align = pd.read_csv(EVAL / "movielens_alignment.csv")
        ml_to_cat = dict(zip(align["movieId"], align["catalog_index"]))
        ratings = pd.read_csv(EVAL / "ml-latest-small" / "ratings.csv")
        ratings = ratings[ratings["movieId"].isin(ml_to_cat)].copy()
        ratings["cat"] = ratings["movieId"].map(ml_to_cat).astype(int)
        ratings = ratings.sort_values("timestamp")
        uni = set(universe)
        by_user = {}
        for uid, grp in ratings.groupby("userId"):
            seen = set(); seq = []
            for cat in grp["cat"]:
                if cat in seen or cat not in uni:
                    continue
                seen.add(cat); seq.append(int(cat))
            by_user[int(uid)] = seq
        out = []
        for u in sorted(train):
            liked = set(train[u])
            # dense context = all rated items strictly up to the last TRAIN like
            full = by_user.get(u, train[u])
            if train[u][-1] in full:
                cut = full.index(train[u][-1]) + 1
                ctx = full[:cut]
            else:
                ctx = [c for c in full if c in liked] or train[u]
            out.append([seqcol[c] for c in ctx if c in seqcol])
        return out
    except Exception as e:
        print(f"[dense] fallback to like-only seqs: {e}")
        return [[seqcol[it] for it in train[u]] for u in sorted(train)]


def _grounded_explanations(champ, M, users, train, full_list, universe, col, S,
                           catalog, n=6):
    """For sample users, attribute each top rec to the most-similar liked item.
    Every title is a real catalog entry (item-item cosine attribution) -- no
    hallucinated titles; an LLM would only reword these grounded pairs."""
    out = []
    for u in users[:n]:
        i = users.index(u)
        row = M[i].copy()
        row[[col[it] for it in train[u]]] = -1e9
        top = np.argsort(-row)[:5]
        cards = []
        for c in top:
            rec_cat = universe[c]
            sims = [(it, float(S[c, col[it]])) for it in train[u]]
            infl, sim = max(sims, key=lambda x: x[1])
            cards.append({
                "recommended": catalog.loc[rec_cat, "Title"],
                "genre": catalog.loc[rec_cat, "Genre"],
                "because_you_liked": catalog.loc[infl, "Title"],
                "similarity": round(sim, 3),
                "explanation": (f"Because you liked "
                                f"\"{catalog.loc[infl, 'Title']}\", you may enjoy "
                                f"\"{catalog.loc[rec_cat, 'Title']}\"."),
                "in_held_out": bool(rec_cat in full_list[u]),
            })
        out.append({"user": u, "model": champ,
                    "liked_sample": [catalog.loc[c, "Title"] for c in train[u][-5:]],
                    "cards": cards})
    return out


def _figures(ni, fl, champ):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for df, key, title, fname, ref in [
            (ni, "ndcg@10", "next-item NDCG@10 (leave-one-out)",
             "phase5_nextitem_ndcg.png", None),
            (fl, "ndcg@10", "full-list NDCG@10 (Day-3 protocol)",
             "phase5_fulllist_ndcg.png", 0.1059)]:
            d = df.sort_values(key)
            fig, ax = plt.subplots(figsize=(7.6, 4.4))
            colors = ["#3b7dd8" if s in ("SASRec", "BERT4Rec") else "#9bb7d4"
                      for s in d["system"]]
            bars = ax.barh(d["system"], d[key], color=colors)
            if ref:
                ax.axvline(ref, color="#c0392b", ls="--", lw=1)
                ax.text(ref, -0.4, f"ItemKNN {ref}", color="#c0392b", fontsize=8)
            ax.set_xlabel(title)
            ax.set_title(f"CineSemantics Day-7: {title}")
            for b, v in zip(bars, d[key]):
                ax.text(v + max(d[key]) * 0.01, b.get_y() + b.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(RESULTS / "figures" / fname, dpi=130)
            plt.close(fig)
        print("[fig] saved phase5 figures")
    except Exception as e:
        print(f"[fig] skipped: {e}")


def _append_leaderboard(fl_rows):
    lb_path = RESULTS / "leaderboard.csv"
    try:
        add = []
        for s in ["SASRec", "BERT4Rec"]:
            r = next(x for x in fl_rows if x["system"] == s)
            add.append({"day": 7, "system": f"{s} (Day-7)",
                        "ndcg@10": r["ndcg@10"], "recall@20": r["recall@20"],
                        "catalog_coverage": r["catalog_coverage"],
                        "note": "sequential transformer, full-list"})
        new = pd.DataFrame(add)
        if lb_path.exists():
            old = pd.read_csv(lb_path)
            # idempotent: drop any prior Day-7 sequential rows before re-adding
            old = old[~((old["day"] == 7) &
                        (old["system"].isin([r["system"] for r in add])))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(lb_path, index=False)
        print("[leaderboard] appended SASRec + BERT4Rec rows")
    except Exception as e:
        print(f"[leaderboard] skipped: {e}")


if __name__ == "__main__":
    main()
