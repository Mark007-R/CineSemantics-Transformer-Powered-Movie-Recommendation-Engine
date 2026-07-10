"""
Day-5 API smoke test — drives every endpoint end to end through the FastAPI
TestClient (real ASGI lifespan, real model load on /search) and saves the
responses to results/samples/phase3_api_smoke.json. Also checks the intended
error paths (422 validation, 404 unknown title/user).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
import api

out = {}
with TestClient(api.app) as c:
    out["health"] = c.get("/health").json()

    r = c.post("/search", json={"query": "space adventure with a robot", "k": 5})
    out["search_basic"] = {"status": r.status_code, "body": r.json()}

    r = c.post("/search", json={"query": "funny heist movie", "k": 5,
                                "genres": ["Action", "Comedy"], "genre_mode": "all",
                                "min_rating": 6.5})
    out["search_genre_and_rating"] = {"status": r.status_code, "body": r.json()}

    r = c.post("/similar", json={"title": "The Dark Knight", "k": 5})
    out["similar_by_title"] = {"status": r.status_code, "body": r.json()}

    r = c.post("/similar", json={"index": 0, "k": 5, "rerank": False})
    out["similar_by_index_norerank"] = {"status": r.status_code, "body": r.json()}

    # personalized recommend for a known user
    rec = api.STATE.get("recommender")
    known_user = rec.users[0] if rec else 1
    r = c.post("/recommend", json={"user_id": int(known_user), "k": 5})
    out["recommend_warm"] = {"status": r.status_code, "body": r.json()}

    r = c.post("/recommend", json={"liked_titles": ["The Dark Knight", "Inception"], "k": 5})
    out["recommend_coldstart"] = {"status": r.status_code, "body": r.json()}

    # error paths
    out["err_similar_404"] = {"status": c.post("/similar",
                              json={"title": "definitely not a real movie xyz"}).status_code}
    out["err_search_422"] = {"status": c.post("/search", json={"query": ""}).status_code}
    out["err_recommend_404"] = {"status": c.post("/recommend",
                                json={"user_id": 999999}).status_code}

(ROOT / "results" / "samples").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results" / "samples" / "phase3_api_smoke.json", "w") as f:
    json.dump(out, f, indent=2)

# console summary
print("health:", out["health"])
for k in ("search_basic", "search_genre_and_rating", "similar_by_title",
          "recommend_warm", "recommend_coldstart"):
    b = out[k]
    n = b["body"].get("count", len(b["body"].get("results", [])))
    print(f"{k}: status={b['status']} results={n}")
print("error paths:", {k: out[k]["status"] for k in
                       ("err_similar_404", "err_search_422", "err_recommend_404")})
assert out["err_similar_404"]["status"] == 404
assert out["err_search_422"]["status"] == 422
assert out["err_recommend_404"]["status"] == 404
assert out["recommend_warm"]["body"]["catalog_valid"] is True
print("\n[smoke] all endpoints + error paths OK")
