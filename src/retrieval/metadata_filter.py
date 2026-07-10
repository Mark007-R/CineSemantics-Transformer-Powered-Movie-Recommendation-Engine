"""
Token-set metadata filtering — the Day-5 fix for the substring genre filter.

The shipped filter (utils/milvus_vectordb.py:354) was:

    if genre_filter.lower() in movie_data['genre'].lower():

a raw substring test. It has two defects proven on Day 5:
  * false positives — ``"art"`` matches ``"Martial Arts"``, ``"rom"`` matches
    ``"Drama"`` (substring, not token);
  * it cannot express multi-genre intent at all — there is no way to ask for
    "Action AND Comedy", or "Action OR Horror, rated 7+".

This module replaces it with proper token-set matching plus rating / year /
popularity predicates, so the retrieval layer finally supports real metadata
queries. Genres are tokenised on ``, / |`` and compared as sets.
"""
from __future__ import annotations

import re

_SPLIT = re.compile(r"[,/|]")


def genre_tokens(genre_str) -> set:
    """Normalise a genre string to a lowercase token set."""
    return {t.strip().lower() for t in _SPLIT.split(str(genre_str)) if t.strip()}


def genre_matches(movie_genre, wanted, mode: str = "any") -> bool:
    """True if the movie's genre tokens satisfy `wanted` under `mode`.

    mode="any" (OR): at least one wanted genre present.
    mode="all" (AND): every wanted genre present.
    An empty `wanted` matches everything.
    """
    if not wanted:
        return True
    have = genre_tokens(movie_genre)
    want = {g.strip().lower() for g in wanted if str(g).strip()}
    if not want:
        return True
    if mode == "all":
        return want.issubset(have)
    return bool(want & have)


def _year_of(release_date) -> int | None:
    s = str(release_date)
    return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None


def passes_filters(record: dict, genres=None, genre_mode: str = "any",
                   min_rating=None, max_rating=None,
                   min_year=None, max_year=None, min_popularity=None) -> bool:
    """Apply all metadata predicates to a catalog record dict.

    Keys are matched case-insensitively against both capitalised catalog columns
    (``Genre``, ``Vote_Average``, ``Release_Date``, ``Popularity``) and the
    lowercase forms used by the Milvus output schema (``genre``, ``vote_average``
    ...), so it works on both the DataFrame rows and the API payloads.
    """
    def get(*keys, default=None):
        for k in keys:
            if k in record and record[k] not in (None, ""):
                return record[k]
        return default

    if genres:
        if not genre_matches(get("Genre", "genre", default=""), genres, genre_mode):
            return False

    rating = get("Vote_Average", "vote_average")
    if rating is not None and rating != "":
        try:
            r = float(rating)
            if min_rating is not None and r < float(min_rating):
                return False
            if max_rating is not None and r > float(max_rating):
                return False
        except (TypeError, ValueError):
            pass

    if min_year is not None or max_year is not None:
        y = _year_of(get("Release_Date", "release_date", default=""))
        if y is None:
            return False
        if min_year is not None and y < int(min_year):
            return False
        if max_year is not None and y > int(max_year):
            return False

    if min_popularity is not None:
        pop = get("Popularity", "popularity")
        try:
            if pop is None or float(pop) < float(min_popularity):
                return False
        except (TypeError, ValueError):
            return False

    return True
