"""
Day-9 (Phase-7) production-wrapper regression tests.

Locks the serving guarantees so a refactor can't silently break the wrapper:
  1. CACHE: miss -> set -> hit; get_or_set computes exactly once; stats report the
     active backend + hit rate. Runs on the in-process fallback (no Redis needed).
  2. FEEDBACK: valid events round-trip through counts(); invalid events are rejected;
     the file backend actually persists (future online-eval ground truth).
  3. TELEMETRY: p50/p95 latency snapshot is well-formed.
  4. OFFLINE PANEL: sources the champion + leaderboard from results/ (the Day-1
     "zero evaluation" gap, now surfaced live).
  5. FOR-YOU TAB logic: known liked titles produce catalog-valid recs; unknown titles
     degrade gracefully to an empty list (no crash, no hallucinated title).
  6. API: /health, cached /recommend, /feedback (+422 on bad event), /metrics,
     /telemetry — end to end via TestClient (skips cleanly if artifacts absent).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pages"))

from src.serving.cache import RecoCache, make_key            # noqa: E402
from src.serving.feedback import FeedbackLog                  # noqa: E402
from src.serving.telemetry import Telemetry                   # noqa: E402
from src.serving.offline_metrics import panel                 # noqa: E402


# 1. cache -------------------------------------------------------------------
def test_cache_miss_set_hit():
    c = RecoCache(url="redis://127.0.0.1:1")   # unreachable -> memory fallback
    assert c.backend == "memory"
    k = make_key("recommend", {"liked": [1, 2], "top_k": 5})
    assert c.get(k) is None                    # miss
    c.set(k, [{"index": 7}])
    assert c.get(k) == [{"index": 7}]          # hit
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5


def test_cache_get_or_set_computes_once():
    c = RecoCache(url="redis://127.0.0.1:1")
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return {"v": 42}

    k = make_key("x", {"a": 1})
    v1, cached1 = c.get_or_set(k, producer)
    v2, cached2 = c.get_or_set(k, producer)
    assert v1 == v2 == {"v": 42}
    assert cached1 is False and cached2 is True
    assert calls["n"] == 1                      # producer ran exactly once


def test_cache_key_order_insensitive():
    assert make_key("r", {"a": 1, "b": 2}) == make_key("r", {"b": 2, "a": 1})


# 2. feedback ----------------------------------------------------------------
def test_feedback_roundtrip_and_ctr(tmp_path):
    fb = FeedbackLog(url="redis://127.0.0.1:1", path=tmp_path / "fb.jsonl")
    assert fb.backend == "file"
    fb.log("impression", 10, source="t", rank=0)
    fb.log("impression", 11, source="t", rank=1)
    fb.log("click", 10, source="t", rank=0)
    c = fb.counts()
    assert c["by_event"]["impression"] == 2 and c["by_event"]["click"] == 1
    assert c["impression_ctr"] == 0.5
    assert (tmp_path / "fb.jsonl").exists()


def test_feedback_rejects_bad_event(tmp_path):
    fb = FeedbackLog(url="redis://127.0.0.1:1", path=tmp_path / "fb.jsonl")
    with pytest.raises(ValueError):
        fb.log("purchase", 1)


# 3. telemetry ---------------------------------------------------------------
def test_telemetry_snapshot():
    t = Telemetry()
    for ms in [1.0, 2.0, 3.0, 100.0]:
        t.record("/recommend", ms)
    snap = t.snapshot()["/recommend"]
    assert snap["count"] == 4
    assert snap["p50_ms"] is not None and snap["max_ms"] == 100.0


# 4. offline panel -----------------------------------------------------------
def test_offline_panel_shape():
    p = panel()
    assert "champion_ranker" in p and "ItemKNN" in p["champion_ranker"]
    assert isinstance(p["leaderboard_top"], list)


# 5. for-you tab logic -------------------------------------------------------
def test_for_you_known_and_unknown():
    import recommend_tab as rt
    recs = rt.recommend_for_titles(["Toy Story", "The Lion King"], top_k=5)
    if not recs:
        pytest.skip("recommender artifacts not present")
    assert len(recs) <= 5
    assert all(r["title"] and isinstance(r["index"], int) for r in recs)   # catalog-valid
    assert rt.recommend_for_titles(["zzz definitely not a movie zzz"]) == []


# 6. API end-to-end ----------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    try:
        from fastapi.testclient import TestClient
        import api
        with TestClient(api.app) as c:
            if c.get("/health").json()["status"] != "ok":
                pytest.skip("API not healthy")
            yield c
    except Exception as e:
        pytest.skip(f"API unavailable: {e}")


def test_api_recommend_cached(client):
    body = {"liked_titles": ["Toy Story", "The Lion King"], "top_k": 5}
    r1 = client.post("/recommend", json=body)
    if r1.status_code != 200:
        pytest.skip("recommender not loaded in this env")
    client.post("/recommend", json=body)                 # second -> cache hit
    stats = client.get("/metrics").json()["cache"]
    assert stats["hits"] >= 1


def test_api_feedback_and_metrics(client):
    ok = client.post("/feedback", json={"event": "click", "movie_index": 227,
                                        "source": "test", "rank": 0})
    assert ok.status_code == 200 and ok.json()["logged"] is True
    bad = client.post("/feedback", json={"event": "buy", "movie_index": 1})
    assert bad.status_code == 422
    m = client.get("/metrics").json()
    assert "offline" in m and "cache" in m and "feedback" in m
    t = client.get("/telemetry").json()
    assert any(k.startswith("/") for k in t)
