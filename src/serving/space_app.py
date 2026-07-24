"""
CineSemantics — self-contained Streamlit demo (Hugging Face Space entrypoint).

The full Streamlit app (pages/movieflix.py) is Milvus-backed and needs the
docker-compose stack. This app serves the SAME champion components the FastAPI
service uses — e5-base-v2 embeddings + faiss HNSW + metadata rerank + ItemKNN
CF — entirely from on-disk artifacts, so it runs on a single free CPU with no
vector database. Wiring mirrors api.py's lifespan exactly; nothing here is a
second implementation.

Run:  streamlit run src/serving/space_app.py
Artifacts required (all shipped with the repo / space):
  data/9000plus.csv, results/emb_cache/intfloat__e5-base-v2.npy,
  models/cf_interactions.npz + cf_meta.json (+ data/eval/cf_split.json fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.embedder import ChampionEmbedder          # noqa: E402
from src.retrieval.index import MovieIndex                    # noqa: E402
from src.rerank.metadata_rerank import MetadataReranker       # noqa: E402
from src.recsys.recommender import ItemKNNRecommender         # noqa: E402
from src.serving.offline_metrics import panel as offline_panel  # noqa: E402

st.set_page_config(page_title="CineSemantics", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading catalog, embeddings and index…")
def load_stack():
    """Same wiring as api.py's lifespan — catalog → embedder → HNSW → CF."""
    catalog = pd.read_csv(ROOT / "data" / "9000plus.csv").fillna("")
    embedder = ChampionEmbedder()
    emb = embedder.encode_catalog(catalog)          # served from the .npy cache
    index = MovieIndex(catalog, emb, embedder)
    index.build_faiss()
    reranker = MetadataReranker(catalog)
    try:
        rec = ItemKNNRecommender.load(ROOT / "models", catalog_embeddings=emb)
    except Exception:                                # noqa: BLE001
        rec = None
    genres = sorted({g.strip() for gs in catalog["Genre"].astype(str)
                     for g in gs.split(",") if g.strip()})
    return catalog, index, reranker, rec, genres


def movie_row(meta: dict | pd.Series, score: float, method: str | None = None):
    """One result row: poster thumbnail + title/genre/year + score."""
    c1, c2, c3 = st.columns([1, 6, 2])
    url = str(meta.get("Poster_Url", "") or "")
    with c1:
        if url.startswith("http"):
            st.image(url, width=64)
    with c2:
        year = str(meta.get("Release_Date", ""))[:4]
        st.markdown(f"**{meta.get('Title', '')}** ({year})  \n"
                    f"<span style='color:gray;font-size:0.85em'>{meta.get('Genre', '')}</span>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"`{score:.3f}`" + (f"  \n{method}" if method else ""))


def main() -> None:
    catalog, index, reranker, rec, genres = load_stack()

    st.title("🎬 CineSemantics")
    st.caption(
        f"Semantic search + personalized recommendation over {len(catalog):,} TMDB movies — "
        "e5-base-v2 embeddings, faiss HNSW, metadata rerank, ItemKNN collaborative filtering. "
        "No Milvus required: this Space serves the exact champion stack the offline "
        "evaluation measured. Source: "
        "[Mark007-R/Semantic-Movie-Recommender](https://github.com/Mark007-R/Semantic-Movie-Recommender)"
    )

    tab_search, tab_similar, tab_foryou, tab_metrics = st.tabs(
        ["🔍 Semantic search", "🎞️ More like this", "🤝 For you", "📊 Honest metrics"])

    # ---------------------------------------------------------------- search
    with tab_search:
        q = st.text_input("Describe what you feel like watching",
                          placeholder="a heist that goes sideways, dark humour")
        f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
        sel_genres = f1.multiselect("Genres (proper filter — not a substring match)", genres)
        min_rating = f2.slider("Min rating", 0.0, 10.0, 0.0, 0.5)
        yr = f3.slider("Year range", 1900, 2026, (1900, 2026))
        rerank = f4.toggle("Rerank", value=True,
                           help="Day-4 metadata reranker: cosine + popularity prior + genre Jaccard")
        if q.strip():
            hits = index.search(q, top_k=10, genres=sel_genres or None,
                                min_rating=min_rating or None,
                                min_year=yr[0] if yr[0] > 1900 else None,
                                max_year=yr[1] if yr[1] < 2026 else None,
                                overfetch=30 if rerank else 20)
            if rerank and hits:
                ordered = reranker.rerank_query([h["index"] for h in hits],
                                                [h["score"] for h in hits],
                                                query_genres=sel_genres or None)
                by_idx = {h["index"]: h for h in hits}
                hits = [by_idx[i] for i in ordered]
            if not hits:
                st.info("No catalog match under those filters — relax them and retry.")
            for h in hits[:10]:
                movie_row(catalog.iloc[h["index"]], h["score"])

    # --------------------------------------------------------------- similar
    with tab_similar:
        title = st.selectbox("Pick a movie", catalog["Title"].astype(str).tolist(),
                             index=None, placeholder="Start typing a title…")
        if title:
            qi = index.title_to_index(title)
            if qi is None:
                st.warning("Title not found in the index.")
            else:
                for h in index.similar(qi, top_k=10):
                    movie_row(catalog.iloc[h["index"]], h["score"])

    # ---------------------------------------------------------------- for you
    with tab_foryou:
        st.markdown("Personalized ranking from the **ItemKNN CF champion** "
                    "(Day-3 winner on the held-out MovieLens split), with a "
                    "content cold-start fallback for titles without interactions.")
        liked = st.multiselect("Movies you liked", catalog["Title"].astype(str).tolist())
        if liked and rec is None:
            st.error("CF artifacts not loaded — models/cf_interactions.npz missing.")
        elif liked:
            liked_idx = [i for t in liked if (i := index.title_to_index(t)) is not None]
            for r in rec.recommend(liked_idx, top_k=10):
                movie_row(catalog.iloc[r["index"]], r["score"], r.get("method"))

    # ---------------------------------------------------------------- metrics
    with tab_metrics:
        st.markdown(
            "This project's Day-1 audit finding was that a *'recommendation engine'* "
            "shipped with **zero evaluation**. This panel is the fix, live: the same "
            "offline numbers the API serves at `/metrics`, sourced from the persisted "
            "`results/` artifacts — including the honest ones."
        )
        st.json(offline_panel())


if __name__ == "__main__":
    main()
