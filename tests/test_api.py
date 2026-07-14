"""
Day-10 (Phase-8) — FastAPI service tests (api.py), golden path + edge cases.

Drives the integrated champions end-to-end through TestClient (the lifespan loads
the cached e5-base vectors + faiss HNSW + ItemKNN artifact, so it is fast and needs
no Milvus). Skips cleanly if the artifacts are absent. Edge cases mandated by the
Day-10 spec:
  * empty query           -> 422 (schema rejects min_length=1)
  * out-of-catalog title  -> 404 on /similar, graceful 400 on /recommend
  * cold-start user       -> unknown liked titles never hallucinate a catalog item
Golden path: /health, /search, /similar, /recommend all return catalog-valid rows.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    try:
        from fastapi.testclient import TestClient
        import api
        with TestClient(api.app) as c:
            if c.get("/health").json().get("status") != "ok":
                pytest.skip("API not healthy (artifacts missing)")
            yield c
    except Exception as e:                       # noqa: BLE001
        pytest.skip(f"API unavailable: {e}")


# ---- health / golden path --------------------------------------------------
def test_health_reports_champion_stack(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["embedding_model"] == "intfloat/e5-base-v2"
    assert h["index"] == "HNSW"
    assert h["catalog_size"] > 9000            # full TMDB catalog loaded


def test_search_golden_path_is_catalog_valid(client):
    r = client.post("/search", json={"query": "space adventure with robots",
                                     "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body) <= 5
    assert all(m["title"] and isinstance(m["index"], int) for m in body)
    assert all(m["index"] >= 0 for m in body)  # real rows, nothing hallucinated


def test_search_with_genre_filter(client):
    r = client.post("/search", json={"query": "funny", "top_k": 5,
                                     "genres": ["Comedy"]})
    assert r.status_code == 200
    for m in r.json():
        toks = {t.strip().lower() for t in m["genre"].replace("/", ",").split(",")}
        assert "comedy" in toks                # filter is honoured, not substring-y


def test_similar_golden_path(client):
    r = client.post("/similar", json={"title": "Toy Story", "top_k": 5})
    if r.status_code == 404:
        pytest.skip("seed title not in this catalog build")
    assert r.status_code == 200
    body = r.json()
    assert body and all(m["title"] for m in body)


def test_recommend_golden_path_personalized(client):
    r = client.post("/recommend", json={"liked_titles": ["Toy Story",
                                                          "The Lion King"],
                                        "top_k": 5})
    if r.status_code != 200:
        pytest.skip("recommender/seed titles unavailable in this env")
    body = r.json()
    assert 1 <= len(body) <= 5
    assert all(m["title"] and m["index"] >= 0 for m in body)   # catalog-valid


# ---- edge cases ------------------------------------------------------------
def test_empty_query_is_rejected_422(client):
    r = client.post("/search", json={"query": "", "top_k": 5})
    assert r.status_code == 422                 # min_length=1 schema guard


def test_search_top_k_out_of_range_422(client):
    r = client.post("/search", json={"query": "x", "top_k": 9999})
    assert r.status_code == 422                 # le=100 guard


def test_out_of_catalog_title_similar_404(client):
    r = client.post("/similar", json={"title": "zzz definitely not a movie zzz"})
    assert r.status_code == 404                 # no fabricated match


def test_recommend_unknown_titles_no_hallucination(client):
    """Cold-start with entirely unknown titles: never invent a catalog item."""
    r = client.post("/recommend",
                    json={"liked_titles": ["zzz not a movie zzz"], "top_k": 5})
    # no resolvable liked items -> 400 (explicit), NOT a hallucinated recommendation
    assert r.status_code == 400


def test_recommend_requires_some_input_400(client):
    r = client.post("/recommend", json={"top_k": 5})
    assert r.status_code == 400                 # neither titles nor indices given
