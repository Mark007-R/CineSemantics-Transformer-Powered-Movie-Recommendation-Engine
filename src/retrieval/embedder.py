"""
Production text embedder — the Day-2 champion (e5-base-v2, 768-d).

Day 1-2 benchmarked five encoders on the held-out co-rating eval and e5-base-v2
won (NDCG@10 0.0482 vs the shipped MiniLM 0.0295, +63%). This wraps that champion
for the production retrieval path so the API and the shipped `utils/` use the same
encoder that was actually measured.

e5 uses an asymmetric instruction convention; for the SYMMETRIC "more like this"
similarity task the Day-2 bake-off prefixed BOTH sides with ``"query: "``. We keep
that convention here so the production numbers reproduce the offline champion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "emb_cache"

CHAMPION_HF_ID = "intfloat/e5-base-v2"
CHAMPION_DIM = 768
E5_PREFIX = "query: "


def _needs_prefix(hf_id: str) -> bool:
    return "e5" in hf_id.lower()


class TextEmbedder:
    """Lazy-loading wrapper around the champion sentence-transformer.

    The heavy model is only instantiated when a *new* string has to be encoded
    (e.g. a live query). Reproducing offline numbers uses the disk-cached catalog
    matrix and never loads the model, so validation stays fast and deterministic.
    """

    def __init__(self, hf_id: str = CHAMPION_HF_ID):
        self.hf_id = hf_id
        self.prefix = E5_PREFIX if _needs_prefix(hf_id) else ""
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.hf_id)
        return self._model

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """Encode a list of raw texts to L2-normalized float32 vectors."""
        if isinstance(texts, str):
            texts = [texts]
        model = self._ensure_model()
        inp = [self.prefix + t for t in texts] if self.prefix else list(texts)
        emb = model.encode(inp, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query, returned shape (768,)."""
        return self.encode([text])[0]

    @staticmethod
    def load_catalog_matrix() -> np.ndarray:
        """Return the cached e5 catalog embeddings (n_movies x 768), L2-normalized.

        Produced by src/eval/embedding_comparison.py (Day 2). Gitignored but
        reproducible; the production path reuses it rather than re-encoding 9.8K
        movies on every boot.
        """
        f = CACHE / "intfloat__e5-base-v2.npy"
        if not f.exists():
            raise FileNotFoundError(
                f"Champion catalog embeddings not found at {f}. "
                "Run src/eval/embedding_comparison.py to regenerate the e5 cache."
            )
        return np.load(f).astype(np.float32)
