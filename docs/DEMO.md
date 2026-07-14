# CineSemantics — 60-second demo script

A tight, interview-runnable walkthrough. The live driver is
[`scripts/demo.py`](../scripts/demo.py) — it runs in ~1–2s off cached artifacts (no
Milvus, no uvicorn). Below is the narration + the optional full-stack version.

## The 60-second narration

**[0:00–0:10] The gap.**
> "This project shipped as a *transformer-powered recommendation engine*. But it had no
> recommender — just semantic search plus a substring genre filter — and, more importantly,
> **zero evaluation**. So step one was building the offline eval harness."

**[0:10–0:25] The uncomfortable number.**
> "The first honest measurement, on a held-out MovieLens split: off-the-shelf semantic
> search scored NDCG@10 of 0.03 — and **lost to a plain popularity baseline by 7×**. A
> recommender that can't beat 'show everyone the most popular movies' isn't recommending."
>
> _(run `python scripts/demo.py` — section 2 prints the leaderboard live)_

**[0:25–0:40] The fix that actually moved the needle.**
> "The lift didn't come from a bigger encoder. It came from adding the recommender the
> name always implied. Item-item collaborative filtering hit NDCG@10 0.106 — **+47% over
> popularity** on the personalized split. In the ablation, CF is the *only* rung that
> moves the metric ~5×; even the SASRec/BERT4Rec transformers only *tie* it."

**[0:40–0:55] Grounded, personalized, live.**
> "Here's a live recommendation for a user who liked Toy Story, The Lion King, and Aladdin —
> Jurassic Park, Beauty and the Beast, Toy Story 2. Every title is a **real catalog row**.
> Compare that to a zero-shot LLM: statistically it ties on ranking, but it **hallucinates
> 3.9% of titles**, runs ~5000× slower, and costs ~2¢ a query."

**[0:55–1:00] Close.**
> "Grounding, personalization, latency, and cost — measured honestly the whole way. That's
> the engine the name always promised."

## Run it

```bash
# Fast live demo (cached vectors + persisted ItemKNN artifact) — ~1–2s
python scripts/demo.py

# Full stack (optional): the API + Streamlit For-You tab
docker-compose up -d
uvicorn api:app --port 8000
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"liked_titles":["Toy Story","The Lion King","Aladdin"],"top_k":5}'
streamlit run pages/movieflix.py     # → "For You" tab
```

## What to have on screen
1. `scripts/demo.py` output (the 4-act story).
2. `README.md` headline tables (the ablation + frontier comparison).
3. `pytest tests/ -q` — 99 green (retrieval, CF, rerank, sequential, metrics, API).
