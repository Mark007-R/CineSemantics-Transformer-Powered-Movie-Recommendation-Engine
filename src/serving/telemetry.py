"""
Per-request telemetry (Day-9 Phase 7).

Records per-endpoint latency and call counts in-process so /telemetry (and the ops
panel) can report p50/p95 without an external APM. Deliberately tiny and dependency
-free; a real deployment would export these to Prometheus, but the point here is that
every served request is measured -- the same "measure honestly" discipline the
offline eval established, carried into serving.
"""
from __future__ import annotations

import time
from collections import defaultdict


class Telemetry:
    def __init__(self, keep: int = 2000):
        self.keep = keep
        self._lat: dict[str, list[float]] = defaultdict(list)
        self._count: dict[str, int] = defaultdict(int)

    def record(self, endpoint: str, ms: float) -> None:
        self._count[endpoint] += 1
        buf = self._lat[endpoint]
        buf.append(ms)
        if len(buf) > self.keep:
            del buf[0]

    def snapshot(self) -> dict:
        out = {}
        for ep, lats in self._lat.items():
            s = sorted(lats)
            n = len(s)
            out[ep] = {
                "count": self._count[ep],
                "p50_ms": round(s[int(0.50 * (n - 1))], 2) if n else None,
                "p95_ms": round(s[int(0.95 * (n - 1))], 2) if n else None,
                "max_ms": round(s[-1], 2) if n else None,
            }
        return out


class _Timer:
    def __init__(self, tel: Telemetry, endpoint: str):
        self.tel, self.endpoint = tel, endpoint

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.tel.record(self.endpoint, (time.perf_counter() - self.t) * 1000.0)
