"""
"Recommended for you" Streamlit tab (Day-9 Phase 7).

The name always promised a recommender; Days 1-8 built and measured one. Day 9 finally
surfaces it in the UI: this tab turns the user's Favorites + Watchlist into a "liked"
set, runs the **Day-3 ItemKNN champion** (the same ranker served by the API), and shows
personalized picks — with a **live offline-metrics panel** that renders the honest eval
numbers straight from `results/`, and **implicit-feedback logging** (impressions on what
is shown, clicks/adds on what the user acts on) so a future online eval has ground truth.

The data logic (`title_index_map`, `recommend_for_titles`) is Streamlit-free so it can be
unit-tested; `render_recommend_tab` is the thin `st`-using view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recsys.recommender import ItemKNNRecommender          # noqa: E402
from src.serving.feedback import FeedbackLog                    # noqa: E402
from src.serving.offline_metrics import panel as offline_panel  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"

_STATE: dict = {}


def _catalog() -> pd.DataFrame:
    if "catalog" not in _STATE:
        _STATE["catalog"] = pd.read_csv(DATA / "9000plus.csv").fillna("")
    return _STATE["catalog"]


def title_index_map() -> dict:
    """Lowercased title -> catalog index (first occurrence)."""
    if "t2i" not in _STATE:
        cat = _catalog()
        m = {}
        for i, t in zip(cat.index, cat["Title"].astype(str)):
            key = t.strip().lower()
            if key and key not in m:
                m[key] = int(i)
        _STATE["t2i"] = m
    return _STATE["t2i"]


def get_recommender() -> ItemKNNRecommender | None:
    """Load the champion; self-heal from cf_split.json if the artifact is legacy/missing."""
    if "rec" in _STATE:
        return _STATE["rec"]
    rec = None
    try:
        rec = ItemKNNRecommender.load(MODELS)
    except Exception:
        split = DATA / "eval" / "cf_split.json"
        if split.exists():
            import json
            s = json.loads(split.read_text())
            train = {int(u): [int(x) for x in v] for u, v in s["train"].items()}
            universe = [int(x) for x in s["item_universe"]]
            rec = ItemKNNRecommender().fit(train, universe)
            try:
                rec.save(MODELS, display={"day3_ndcg@10": 0.1059})
            except Exception:
                pass
    if rec is not None:
        cat = _catalog()
        try:
            rec.attach_genres(cat)      # enables optional MMR diversity
        except Exception:
            pass
    _STATE["rec"] = rec
    return rec


def recommend_for_titles(liked_titles, top_k: int = 12, diversity: float | None = None):
    """Pure logic: liked titles -> ranked catalog rows (dicts). Empty list if none map."""
    rec = get_recommender()
    if rec is None:
        return []
    t2i = title_index_map()
    liked_idx = [t2i[t.strip().lower()] for t in liked_titles
                 if t.strip().lower() in t2i]
    if not liked_idx:
        return []
    cat = _catalog()
    recs = rec.recommend(liked_idx, top_k=top_k, diversity=diversity)
    out = []
    for rank, r in enumerate(recs):
        row = cat.loc[r["index"]]
        out.append({"rank": rank, "index": r["index"], "title": str(row["Title"]),
                    "genre": str(row.get("Genre", "")),
                    "release_date": str(row.get("Release_Date", "")),
                    "overview": str(row.get("Overview", "")),
                    "poster": str(row.get("Poster_Url", "")),
                    "score": r["score"], "method": r["method"]})
    return out


def _feedback_log():
    if "fb" not in _STATE:
        _STATE["fb"] = FeedbackLog()
    return _STATE["fb"]


def render_recommend_tab(session_state):
    import streamlit as st

    st.markdown("### ✨ Recommended for you")
    st.caption("Personalized picks from the **Day-3 ItemKNN champion** — the same ranker "
               "the API serves. Built from your Favorites + Watchlist.")

    liked = []
    for m in list(getattr(session_state, "favorites", [])) + \
            list(getattr(session_state, "watchlist", [])):
        t = str(m.get("Title", "")).strip()
        if t:
            liked.append(t)
    liked = list(dict.fromkeys(liked))     # de-dup, keep order

    # --- live offline-metrics panel (closes the Day-1 "zero evaluation" gap visibly) ---
    with st.expander("📊 Live offline evaluation (held-out MovieLens split)", expanded=False):
        p = offline_panel()
        c1, c2, c3 = st.columns(3)
        c1.metric("Champion ranker", "ItemKNN")
        c2.metric("NDCG@10", f"{p.get('champion_ndcg@10', 0):.4f}")
        c3.metric("Users evaluated", p.get("n_users_eval", "—"))
        if p.get("leaderboard_top"):
            st.dataframe(pd.DataFrame(p["leaderboard_top"]), hide_index=True,
                         use_container_width=True)
        if p.get("sequential"):
            s = p["sequential"]
            st.caption(f"Day-7 sequential: {s.get('champion')} next-item "
                       f"NDCG@10 {s.get('nextitem_ndcg@10')} — {s.get('note')}")

    if not liked:
        st.info("Add movies to your **Favorites** or **Watchlist** to unlock personalized "
                "recommendations.")
        return

    diversity = st.checkbox("Diversify (MMR, reduces genre over-concentration)", value=False)
    recs = recommend_for_titles(liked, top_k=12,
                                diversity=0.7 if diversity else None)
    if not recs:
        st.warning("None of your saved titles are in the recommender's training catalog "
                   "yet — try adding a few more mainstream titles.")
        return

    fb = _feedback_log()
    for r in recs:                          # impression = we showed it
        fb.log("impression", r["index"], source="ui/for_you", rank=r["rank"],
               score=r["score"])

    st.write(f"Because you liked **{', '.join(liked[:3])}**"
             f"{' and more' if len(liked) > 3 else ''}:")
    cols = st.columns(3)
    for i, r in enumerate(recs):
        with cols[i % 3]:
            if r["poster"] and r["poster"].startswith("http"):
                st.image(r["poster"], use_container_width=True)
            st.markdown(f"**{r['title']}**")
            st.caption(f"{r['genre']} · {str(r['release_date'])[:4]}")
            if st.button("👍 Not interested → 👀 Interested", key=f"fy_click_{r['index']}"):
                fb.log("click", r["index"], source="ui/for_you", rank=r["rank"],
                       score=r["score"])
                st.toast(f"Logged interest in {r['title']}")

    st.divider()
    stats = fb.counts()
    st.caption(f"Implicit feedback logged ({stats['backend']}): "
               f"{stats['by_event']} · CTR {stats.get('impression_ctr')}")
