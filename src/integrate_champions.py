"""
Day-5 champion integration — persist the production artifacts.

Builds the CF recommender from the held-out split and saves its interaction
matrix + universe to models/ (gitignored, regenerable) so api.py can hot-load the
personalization layer without recomputing. The e5 catalog embeddings and the
faiss HNSW index are cheap to rebuild from the cached embedding matrix at boot,
so only the CF artifact is persisted here.

Run once after the eval split exists:  python src/integrate_champions.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.embedder import TextEmbedder
from recsys.recommender import Recommender

MODELS = ROOT / "models"


def main():
    emb = TextEmbedder.load_catalog_matrix()
    rec = Recommender.from_split(catalog_embeddings=emb)
    path = rec.save()
    meta = json.loads((MODELS / "cf_meta.json").read_text())
    print(f"[integrate] saved CF artifact -> {path}")
    print(f"[integrate] {meta['n_users']} users x {meta['n_items']} items; "
          f"champion={meta['champion']} (Day-3 NDCG@10={meta['day3_ndcg@10']})")
    print("[integrate] e5 embeddings + faiss HNSW rebuild from cache at API boot.")


if __name__ == "__main__":
    main()
