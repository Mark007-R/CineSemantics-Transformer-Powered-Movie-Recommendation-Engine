"""
CineSemantics — champion metadata reranker (Day-5 Phase-3 integration).

Day-4 head-to-head (src/rerank/rerank_fusion.py) crowned this cheap reranker over
the "obvious" neural cross-encoder:

    variant                         NDCG@10   p95 latency
    champion_semantic (e5-base)     0.0482    2.7 ms
    metadata_rerank (this)          0.0683    48.8 ms   <- +42%
    cross_encoder ms-marco top-15   0.0503    604.7 ms  <- +4% for 220x latency

Score = cosine + a * genre_Jaccard(query, candidate) + b * popularity_prior.
The popularity prior matters because Day-1 proved popularity dominates co-rating
relevance (popularity alone scored NDCG@10 0.2148) — a text-only cross-encoder is
blind to it. Weights a=0.2, b=0.05 were tuned on a Day-2 dev half and validated on
a disjoint test half; they are carried here unchanged.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DEFAULT_A = 0.2   # genre-Jaccard weight
DEFAULT_B = 0.05  # popularity-prior weight


def _genre_set(g) -> set[str]:
    return {t.strip().lower() for t in re.split(r"[,/|]", str(g)) if t.strip()}


class MetadataReranker:
    """Rerank a semantic candidate pool with genre overlap + a popularity prior."""

    def __init__(self, catalog: pd.DataFrame, a: float = DEFAULT_A, b: float = DEFAULT_B):
        self.a = a
        self.b = b
        self.genres = [_genre_set(g) for g in catalog["Genre"]]
        vc = pd.to_numeric(catalog["Vote_Count"], errors="coerce").fillna(0).values.astype(float)
        self.pop_norm = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)

    def rerank(self, query_idx: int, candidate_idx, cosines) -> list[int]:
        """Reorder candidate_idx (with their cosine scores) for query_idx."""
        gq = self.genres[query_idx]
        scored = []
        for i, cos in zip(candidate_idx, cosines):
            gi = self.genres[i]
            u = gq | gi
            jac = (len(gq & gi) / len(u)) if u else 0.0
            scored.append((int(i), float(cos) + self.a * jac + self.b * float(self.pop_norm[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored]

    def rerank_query(self, candidate_idx, cosines, query_genres=None) -> list[int]:
        """Rerank a candidate pool for a FREE-TEXT query (no query movie index).

        Unlike item-item rerank, a text query has no genres of its own, so the
        Day-4 genre-Jaccard anchor is undefined. Two principled cases:
          * caller supplied `query_genres` (e.g. the /search genre filter) -> use
            those as the Jaccard anchor (honours stated intent);
          * otherwise -> drop the Jaccard term and rerank by cosine + the
            popularity prior only (Day-1 showed popularity is the dominant signal;
            a fabricated anchor genre would just bias toward one arbitrary result).
        """
        want: set[str] = set()
        for g in (query_genres or []):
            want |= _genre_set(g)
        scored = []
        for i, cos in zip(candidate_idx, cosines):
            jac = 0.0
            if want:
                gi = self.genres[i]
                u = want | gi
                jac = (len(want & gi) / len(u)) if u else 0.0
            scored.append((int(i), float(cos) + self.a * jac + self.b * float(self.pop_norm[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored]

    def rerank_scores(self, query_idx: int, candidate_idx, cosines):
        gq = self.genres[query_idx]
        out = []
        for i, cos in zip(candidate_idx, cosines):
            gi = self.genres[i]
            u = gq | gi
            jac = (len(gq & gi) / len(u)) if u else 0.0
            out.append((int(i), round(float(cos) + self.a * jac + self.b * float(self.pop_norm[i]), 4)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
