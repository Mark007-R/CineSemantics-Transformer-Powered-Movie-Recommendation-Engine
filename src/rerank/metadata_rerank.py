"""
Production reranker — the Day-4 champion metadata reranker.

Day 4 head-to-head: a neural cross-encoder (ms-marco-MiniLM) added +4% NDCG@10
for a 220x latency tax and *lost* to this 5-line metadata reranker (0.0683 vs
0.0503). The reranker rescores the top-`pool` cosine candidates by

    score = cosine + a * genre_Jaccard(query, cand) + b * pop_prior(cand)

with (a, b) tuned on a dev half of the Day-2 queries. The tuned optimum was
a=0.0, b=0.1 — i.e. on this co-rating eval popularity is the signal that lifts
NDCG (Day-1's central finding), and genre overlap adds nothing on top of the
already genre-aware embedding.
"""
from __future__ import annotations

import numpy as np

# Day-2 tuned weights (dev-tuned, test-reported).
DEFAULT_A_GENRE = 0.0
DEFAULT_B_POP = 0.1
DEFAULT_POOL = 200


def popularity_prior(vote_counts: np.ndarray) -> np.ndarray:
    """Min-max normalised vote_count, the Day-2/4 popularity prior."""
    vc = np.asarray(vote_counts, dtype=float)
    return (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)


def metadata_rerank(order, sims, genres, pop_norm,
                    a: float = DEFAULT_A_GENRE, b: float = DEFAULT_B_POP,
                    query_genre: set | None = None, pool: int = DEFAULT_POOL):
    """Rerank a semantic candidate list by cosine + genre-Jaccard + pop prior.

    Parameters
    ----------
    order : sequence of catalog indices already sorted by descending cosine.
    sims  : full similarity array (indexed by catalog index).
    genres: list of genre token-sets, indexed by catalog index.
    pop_norm : popularity prior array, indexed by catalog index.
    query_genre : the query item's genre token-set (for the Jaccard term).

    Returns the reranked list of catalog indices (pool reranked, semantic tail
    appended for recall fairness).
    """
    pool_ids = list(order[:pool])
    gq = query_genre if query_genre is not None else set()
    rescored = []
    for i in pool_ids:
        gi = genres[i]
        union = gq | gi
        jac = (len(gq & gi) / len(union)) if union else 0.0
        rescored.append((int(i), float(sims[i]) + a * jac + b * float(pop_norm[i])))
    rescored.sort(key=lambda x: x[1], reverse=True)
    ranked = [i for i, _ in rescored]
    seen = set(ranked)
    tail = [int(i) for i in order if int(i) not in seen]
    return ranked + tail
