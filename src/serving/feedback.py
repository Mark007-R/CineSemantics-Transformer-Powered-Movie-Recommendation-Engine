"""
Implicit-feedback log for future online evaluation (Day-9 Phase 7).

The whole sprint has been OFFLINE eval on a held-out MovieLens split. To ever run an
ONLINE evaluation (CTR, add-rate, online NDCG on logged interactions), the served app
must first *record* what users actually do. This module logs implicit events
(impression / click / add-to-watchlist / add-to-favorites / like) with the reco
context (source ranker, rank position, score) so a later A/B or bandit harness has
ground truth to learn from.

Storage: a Redis list when available (fast, shared across API replicas); otherwise a
local JSONL append file, so logging works in a plain venv. `.counts()` aggregates by
event type for the live metrics panel; `.recent()` tails the log.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "results" / "feedback_events.jsonl"
REDIS_KEY = "cine:feedback:events"

VALID_EVENTS = {"impression", "click", "add_watchlist", "add_favorite", "like"}


class FeedbackLog:
    def __init__(self, url: str | None = None, path: Path | None = None,
                 max_redis: int = 100_000):
        self.path = Path(path) if path else DEFAULT_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_redis = max_redis
        self._r = None
        self.backend = "file"
        url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            client = redis.Redis.from_url(url, socket_connect_timeout=0.3,
                                          socket_timeout=0.3, decode_responses=True)
            client.ping()
            self._r = client
            self.backend = "redis"
        except Exception:
            self._r = None

    def log(self, event: str, movie_index: int, *, source: str = "unknown",
            rank: int | None = None, score: float | None = None,
            user: str | None = None, ts: float | None = None) -> dict:
        if event not in VALID_EVENTS:
            raise ValueError(f"unknown event '{event}'; valid: {sorted(VALID_EVENTS)}")
        rec = {
            "ts": ts if ts is not None else time.time(),
            "event": event,
            "movie_index": int(movie_index),
            "source": source,
            "rank": rank,
            "score": score,
            "user": user or "anon",
        }
        line = json.dumps(rec)
        if self._r is not None:
            try:
                self._r.rpush(REDIS_KEY, line)
                self._r.ltrim(REDIS_KEY, -self.max_redis, -1)
                return rec
            except Exception:
                self._r = None
                self.backend = "file"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec

    def _iter(self):
        if self._r is not None:
            try:
                for line in self._r.lrange(REDIS_KEY, 0, -1):
                    yield json.loads(line)
                return
            except Exception:
                self._r = None
                self.backend = "file"
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)

    def counts(self) -> dict:
        c = Counter(e["event"] for e in self._iter())
        total = sum(c.values())
        # a crude but honest online proxy: click-through on impressions
        ctr = round(c.get("click", 0) / c["impression"], 4) if c.get("impression") else None
        return {"backend": self.backend, "total": total,
                "by_event": dict(c), "impression_ctr": ctr}

    def recent(self, n: int = 20) -> list[dict]:
        evs = list(self._iter())
        return evs[-n:]
