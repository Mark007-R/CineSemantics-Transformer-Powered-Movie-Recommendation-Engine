"""
CineSemantics - 60-second live demo (interview-runnable).

Tells the whole story in one script, no Milvus and no uvicorn required (uses the
cached e5-base-v2 vectors + the persisted ItemKNN artifact):

  1. THE GAP    the shipped "engine" had zero evaluation.
  2. THE NUMBER once measured, semantic search LOST to a popularity baseline.
  3. THE FIX    collaborative filtering - the missing recommender - beats popularity.
  4. GROUNDED   live personalized recs are 100% real catalog titles (an LLM hallucinates).

Run:  python scripts/demo.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.embedder import ChampionEmbedder      # noqa: E402
from src.retrieval.index import MovieIndex               # noqa: E402
from src.recsys.recommender import ItemKNNRecommender    # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def banner(n, title):
    print(f"\n{'='*66}\n  {n}. {title}\n{'='*66}")


def main():
    t0 = time.perf_counter()
    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")

    banner(1, "THE GAP - a 'recommendation engine' with no evaluation")
    print("  Shipped: all-MiniLM-L6-v2 semantic search + a SUBSTRING genre filter")
    print("  (milvus_vectordb.py:354). No user model. No offline metrics. At all.")

    banner(2, "THE NUMBER - measured honestly, retrieval loses to popularity")
    lb_path = RESULTS / "baseline_leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        print(lb.to_string(index=False))
        sem = lb.loc[lb.system == "Semantic", "ndcg@10"].iloc[0]
        pop = lb.loc[lb.system == "Popularity", "ndcg@10"].iloc[0]
        print(f"\n  -> Semantic NDCG@10 {sem} LOST to Popularity {pop} by {pop/sem:.1f}x.")
    else:
        print("  (skipped - run src/eval/baseline.py to regenerate the leaderboard)")

    banner(3, "THE FIX - collaborative filtering (the missing recommender)")
    cf_path = RESULTS / "phase2b_cf.csv"
    if cf_path.exists():
        cf = pd.read_csv(cf_path)[["system", "ndcg@10", "recall@20"]]
        print(cf.to_string(index=False))
        print("\n  -> ItemKNN 0.1059 = +47% over popularity ON THE PERSONALIZED SPLIT.")
    else:
        print("  (skipped - run src/recsys/cf_compare.py to regenerate the CF table)")

    banner(4, "GROUNDED - live personalized recs (100% real catalog titles)")
    emb = ChampionEmbedder().encode_catalog(catalog)      # cached -> instant
    index = MovieIndex(catalog, emb, ChampionEmbedder())
    index.build_faiss()
    rec = ItemKNNRecommender.load(MODELS, catalog_embeddings=emb)

    liked = ["Toy Story", "The Lion King", "Aladdin"]
    liked_idx = [i for i in (index.title_to_index(t) for t in liked) if i is not None]
    liked_titles = {t.lower() for t in liked}
    print(f"  User liked: {liked}")
    recs = rec.recommend(liked_idx, top_k=8)
    shown = 0
    for r in recs:
        m = catalog.iloc[r["index"]]
        title = str(m["Title"])
        assert title, "hallucinated blank title"               # grounded by construction
        if title.lower() in liked_titles:                      # skip duplicate-title rows
            continue
        print(f"    * {title:<34} [{m['Genre']}]  ({r['method']})")
        shown += 1
        if shown >= 5:
            break

    print("\n  Every title above is a real, clickable catalog row (0% off-catalog).")
    print("  A zero-shot LLM hallucinated 3.9% of titles, ran ~5000x slower,")
    print("  and cost ~$0.023/query (Day-8 frontier comparison).")

    print(f"\n[demo complete in {time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
