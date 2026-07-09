"""
Proper metadata-aware filtering — replaces the substring genre filter that shipped
in utils/milvus_vectordb.py:354 (`genre_filter.lower() in movie_data['genre'].lower()`).

Why the substring filter was wrong:
  * "Sci" or "Fi" matched "Science Fiction" but so did any accidental substring;
    "War" matched the *word inside* "Warner"/"Award"-style overview leakage when a
    caller passed a phrase, and "Romance" would NOT match "Romantic" — brittle both
    ways (false positives AND false negatives).
  * A single string could not express "Action AND Comedy" or "Action OR Comedy".
  * It never tokenised the movie's own multi-genre field ("Action, Adventure,
    Sci-Fi") into discrete genres, so partial-word collisions ranked as matches.

The proper version tokenises BOTH sides into genre sets (split on , / |) and does
exact token-set matching with an explicit any/all mode. Callers can still pass a
single genre string (back-compatible) — it just becomes a one-element set.
"""
from __future__ import annotations

import re
from typing import Iterable


def genre_tokens(value) -> set[str]:
    """Split a raw genre field/string into a normalised set of genre tokens."""
    return {t.strip().lower() for t in re.split(r"[,/|]", str(value)) if t.strip()}


def genre_match(movie_genre, requested, mode: str = "any") -> bool:
    """True if the movie's genres satisfy the requested genres.

    movie_genre : raw genre string of the movie ("Action, Adventure")
    requested   : a genre string, or an iterable of genre strings/tokens
    mode        : "any" (at least one requested genre present, default) or
                  "all" (every requested genre present).
    """
    have = genre_tokens(movie_genre)
    if not have:
        return False
    if isinstance(requested, str):
        want = genre_tokens(requested)
    else:
        want: set[str] = set()
        for r in requested:
            want |= genre_tokens(r)
    if not want:
        return True
    return want.issubset(have) if mode == "all" else bool(have & want)


def passes_filters(meta: dict, *, genres=None, genre_mode="any",
                   min_rating=None, max_rating=None,
                   min_year=None, max_year=None, min_popularity=None) -> bool:
    """Apply the full metadata predicate to one movie's metadata dict."""
    if genres and not genre_match(meta.get("genre", ""), genres, genre_mode):
        return False
    va = _num(meta.get("vote_average"))
    if min_rating is not None and (va is None or va < float(min_rating)):
        return False
    if max_rating is not None and (va is None or va > float(max_rating)):
        return False
    year = _year(meta.get("release_date", ""))
    if min_year is not None and (year is None or year < int(min_year)):
        return False
    if max_year is not None and (year is None or year > int(max_year)):
        return False
    if min_popularity is not None:
        pop = _num(meta.get("popularity"))
        if pop is None or pop < float(min_popularity):
            return False
    return True


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _year(rd) -> int | None:
    s = str(rd)
    return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None
