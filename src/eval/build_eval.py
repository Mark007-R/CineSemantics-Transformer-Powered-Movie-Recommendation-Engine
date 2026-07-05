"""
CineSemantics Day-1 eval-set builder.

Downloads MovieLens (ml-latest-small), aligns its titles to the TMDB catalog in
data/9000plus.csv by (normalized-title, year), and builds a content-retrieval
"more like this" relevance set from co-rating signal (users who liked movie A
also liked movie B). The relevance set is model-agnostic ground truth, so it is a
fair held-out test of the CURRENT semantic-search retrieval, which has zero
offline evaluation today.

Outputs (all under data/eval/):
  - ml-latest-small/...            raw MovieLens files
  - movielens_alignment.csv        catalog_index <-> movielens movieId matches
  - content_relevance.json         {catalog_index: [relevant_catalog_indices]}
  - eval_manifest.json             counts + build parameters
"""
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
EVAL.mkdir(parents=True, exist_ok=True)

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_DIR = EVAL / "ml-latest-small"

# ground-truth build parameters
LIKE_THRESHOLD = 4.0        # rating >= this counts as a "like"
MIN_CO_LIKES = 3            # a candidate must be co-liked by >= this many users
MAX_RELEVANT = 30           # keep top-N most co-liked per query
MIN_RELEVANT = 5            # a query needs at least this many relevant items to be usable
MIN_QUERY_LIKES = 8         # query movie must have at least this many likers


def download_movielens():
    if (ML_DIR / "ratings.csv").exists():
        print(f"[ml] already present at {ML_DIR}")
        return
    print(f"[ml] downloading {ML_URL}")
    r = requests.get(ML_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(EVAL)
    print(f"[ml] extracted to {ML_DIR}")


def normalize_title(t: str) -> str:
    t = t.lower().strip()
    # MovieLens stores "Title, The (1999)" -> move article to front
    m = re.match(r"^(.*),\s*(the|a|an|la|le|les|el|il)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = re.sub(r"[^a-z0-9]+", "", t)  # strip punctuation/spaces
    return t


def parse_ml_title(raw: str):
    """'Toy Story (1995)' -> ('Toy Story', 1995)"""
    m = re.match(r"^(.*)\((\d{4})\)\s*$", raw.strip())
    if not m:
        return raw.strip(), None
    return m.group(1).strip(), int(m.group(2))


def build():
    download_movielens()

    catalog = pd.read_csv(DATA / "9000plus.csv").fillna("")
    catalog["year"] = catalog["Release_Date"].astype(str).str[:4]
    catalog["norm"] = catalog["Title"].astype(str).map(normalize_title)

    # index catalog by (norm_title, year) -> catalog_index (first wins)
    cat_lookup = {}
    for idx, row in catalog.iterrows():
        key = (row["norm"], row["year"])
        cat_lookup.setdefault(key, idx)
        # also allow year-agnostic fallback
        cat_lookup.setdefault((row["norm"], None), idx)

    movies = pd.read_csv(ML_DIR / "movies.csv")
    ratings = pd.read_csv(ML_DIR / "ratings.csv")

    # align MovieLens movieId -> catalog_index
    ml_to_cat = {}
    exact, fallback = 0, 0
    for _, row in movies.iterrows():
        title, year = parse_ml_title(row["title"])
        norm = normalize_title(title)
        if year is not None and (norm, str(year)) in cat_lookup:
            ml_to_cat[row["movieId"]] = cat_lookup[(norm, str(year))]
            exact += 1
        elif (norm, None) in cat_lookup:
            ml_to_cat[row["movieId"]] = cat_lookup[(norm, None)]
            fallback += 1
    print(f"[align] {len(ml_to_cat)} MovieLens movies matched to catalog "
          f"(exact-year={exact}, title-only-fallback={fallback})")

    align_rows = [{"movieId": mid, "catalog_index": cidx,
                   "catalog_title": catalog.loc[cidx, "Title"]}
                  for mid, cidx in ml_to_cat.items()]
    pd.DataFrame(align_rows).to_csv(EVAL / "movielens_alignment.csv", index=False)

    # keep only ratings on aligned movies, translate to catalog index space
    ratings = ratings[ratings["movieId"].isin(ml_to_cat)].copy()
    ratings["cat"] = ratings["movieId"].map(ml_to_cat)

    # user -> set of liked catalog indices
    user_likes = defaultdict(set)
    for _, r in ratings[ratings["rating"] >= LIKE_THRESHOLD].iterrows():
        user_likes[r["userId"]].add(int(r["cat"]))

    # co-like counts: for each liked pair within a user, increment
    co = defaultdict(lambda: defaultdict(int))
    like_count = defaultdict(int)
    for liked in user_likes.values():
        liked = list(liked)
        for a in liked:
            like_count[a] += 1
        for i in range(len(liked)):
            for j in range(i + 1, len(liked)):
                a, b = liked[i], liked[j]
                co[a][b] += 1
                co[b][a] += 1

    relevance = {}
    for q, neighbors in co.items():
        if like_count[q] < MIN_QUERY_LIKES:
            continue
        ranked = sorted(
            ((c, n) for c, n in neighbors.items() if n >= MIN_CO_LIKES),
            key=lambda x: x[1], reverse=True,
        )[:MAX_RELEVANT]
        rel = [c for c, _ in ranked]
        if len(rel) >= MIN_RELEVANT:
            relevance[int(q)] = rel

    with open(EVAL / "content_relevance.json", "w") as f:
        json.dump({str(k): v for k, v in relevance.items()}, f)

    manifest = {
        "movielens_source": ML_URL,
        "catalog_rows": int(len(catalog)),
        "aligned_movies": len(ml_to_cat),
        "align_exact_year": exact,
        "align_title_fallback": fallback,
        "ratings_used": int(len(ratings)),
        "n_queries": len(relevance),
        "avg_relevant_per_query": round(
            sum(len(v) for v in relevance.values()) / max(1, len(relevance)), 2),
        "params": {
            "LIKE_THRESHOLD": LIKE_THRESHOLD, "MIN_CO_LIKES": MIN_CO_LIKES,
            "MAX_RELEVANT": MAX_RELEVANT, "MIN_RELEVANT": MIN_RELEVANT,
            "MIN_QUERY_LIKES": MIN_QUERY_LIKES,
        },
    }
    with open(EVAL / "eval_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[build] manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
