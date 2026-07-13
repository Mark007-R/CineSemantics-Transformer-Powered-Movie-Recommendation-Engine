"""
Redis cache for hot recommendations + embeddings (Day-9 Phase 7).

A recommender that recomputes item-item scores on every request wastes both CPU and
p95 latency on the head of the request distribution (a handful of popular liked-set
combinations dominate real traffic). `RecoCache` memoizes serialized responses keyed
by a stable hash of the request, with a TTL.

Design goal: **runs with or without Redis.** If a Redis server is reachable it is
used; otherwise the cache transparently falls back to a bounded in-process LRU dict,
so the API, the Streamlit tab, and the tests all work in a plain venv with no Docker.
Either way `.stats()` reports hits/misses and the active backend, which the /metrics
endpoint and the Streamlit panel surface.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict


def make_key(namespace: str, payload: dict) -> str:
    """Stable cache key from a request payload (order-insensitive)."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return f"cine:{namespace}:{h}"


class RecoCache:
    def __init__(self, url: str | None = None, ttl: int = 900, max_local: int = 512):
        self.ttl = ttl
        self.max_local = max_local
        self._local: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._r = None
        self.backend = "memory"
        url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis  # optional dependency
            client = redis.Redis.from_url(url, socket_connect_timeout=0.3,
                                          socket_timeout=0.3, decode_responses=True)
            client.ping()
            self._r = client
            self.backend = "redis"
        except Exception:
            self._r = None            # graceful fallback to in-process LRU

    # ---- core ----
    def get(self, key: str):
        if self._r is not None:
            try:
                v = self._r.get(key)
                if v is not None:
                    self.hits += 1
                    return json.loads(v)
                self.misses += 1
                return None
            except Exception:
                self._r = None        # demote to memory on transient failure
                self.backend = "memory"
        # memory backend
        item = self._local.get(key)
        if item and item[0] > time.time():
            self._local.move_to_end(key)
            self.hits += 1
            return json.loads(item[1])
        if item:
            del self._local[key]      # expired
        self.misses += 1
        return None

    def set(self, key: str, value) -> None:
        blob = json.dumps(value, default=str)
        if self._r is not None:
            try:
                self._r.setex(key, self.ttl, blob)
                return
            except Exception:
                self._r = None
                self.backend = "memory"
        self._local[key] = (time.time() + self.ttl, blob)
        self._local.move_to_end(key)
        while len(self._local) > self.max_local:
            self._local.popitem(last=False)   # evict LRU

    def get_or_set(self, key: str, producer):
        """Return cached value or compute+store it. `producer` is a 0-arg callable."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        val = producer()
        self.set(key, val)
        return val, False

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "ttl_seconds": self.ttl,
            "local_entries": len(self._local),
        }
