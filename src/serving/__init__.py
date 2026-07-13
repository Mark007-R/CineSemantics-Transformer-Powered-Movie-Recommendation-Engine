"""CineSemantics Day-9 (Phase 7) production-serving layer.

Adds the operational scaffolding a "recommendation engine" needs to actually be
served: a Redis cache for hot recommendations (with an in-process fallback so the
API runs with or without Redis), an implicit-feedback log for future online eval,
a live offline-metrics panel sourced from the sprint's own results, and per-request
telemetry. Everything degrades gracefully when Redis/Docker are absent so it is
testable and runnable in a plain venv.
"""
