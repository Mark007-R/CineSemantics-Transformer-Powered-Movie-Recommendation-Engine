"""
CineSemantics — champion text embedder (Day-5 Phase-3 integration).

Day-2 bake-off (src/eval/embedding_comparison.py) picked `intfloat/e5-base-v2`
(768-dim) over the shipped `all-MiniLM-L6-v2` (384-dim): content-retrieval
NDCG@10 0.0482 vs 0.0295, a +63% lift on the held-out co-rating "more like this"
set. This module is the single production home for that champion so the Flask app
(utils/text_embedder.py), the offline FastAPI service (api.py) and the eval
harness (src/eval/) all encode text identically.

E5 models are trained with an instruction prefix. For SYMMETRIC movie<->movie and
query<->movie similarity we prefix every text with "query: " — the exact scheme
the Day-2/Day-4 numbers were measured under, so production matches the leaderboard.
All embeddings are L2-normalised, so inner product == cosine.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# champion config (kept in sync with utils/config.py TEXT_MODEL_NAME)
CHAMPION_NAME = "e5-base-v2"
CHAMPION_HF_ID = "intfloat/e5-base-v2"
CHAMPION_PREFIX = "query: "
CHAMPION_DIM = 768

_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "results" / "emb_cache"


def build_catalog_text(row) -> str:
    """Row -> encodable text. Mirrors utils/text_embedder.embed_csv and the Day-1
    baseline EXACTLY so cached catalog embeddings stay valid across the codebase."""
    title = str(row.get("Title", ""))
    overview = str(row.get("Overview", ""))
    genre = str(row.get("Genre", ""))
    rd = str(row.get("Release_Date", ""))
    year = rd[:4] if len(rd) >= 4 else ""
    parts = [title]
    if genre:
        parts.append(f"Genre: {genre}")
    if year:
        parts.append(f"Released: {year}")
    if overview:
        parts.append(overview)
    return ". ".join(parts).strip()


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.clip(norms, 1e-9, None)).astype(np.float32)


class ChampionEmbedder:
    """Lazy wrapper around the champion SentenceTransformer.

    The model is loaded on first use (cold start ~2-3s) so importing this module
    is cheap for tests and the Flask process. Encoding is always normalised.
    """

    def __init__(self, hf_id: str = CHAMPION_HF_ID, prefix: str = CHAMPION_PREFIX):
        self.hf_id = hf_id
        self.prefix = prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.hf_id, device=device)
        return self._model

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        inp = [self.prefix + t for t in texts] if self.prefix else texts
        emb = self.model.encode(inp, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return _l2(np.asarray(emb, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def encode_catalog(self, catalog, use_cache: bool = True) -> np.ndarray:
        """Encode a catalog DataFrame, cached to disk by model id (same path as
        the eval harness) so Days 2/4 and Day 5 share one 8-min encode."""
        safe = self.hf_id.replace("/", "__")
        cache_f = _CACHE / f"{safe}.npy"
        meta_f = _CACHE / f"{safe}.time.json"
        if use_cache and cache_f.exists() and meta_f.exists():
            emb = np.load(cache_f).astype(np.float32)
            if emb.shape[0] == len(catalog):
                return _l2(emb)
        texts = [build_catalog_text(r) for _, r in catalog.iterrows()]
        t0 = time.perf_counter()
        emb = self.encode(texts)
        enc_s = time.perf_counter() - t0
        _CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache_f, emb)
        meta_f.write_text(json.dumps({"encode_seconds": enc_s, "dim": int(emb.shape[1])}))
        return emb


# module-level singleton for the API / Flask process
_default: ChampionEmbedder | None = None


def get_embedder() -> ChampionEmbedder:
    global _default
    if _default is None:
        _default = ChampionEmbedder()
    return _default
