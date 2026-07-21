"""Prometheus metrics for the unified LLM pipeline (routing / overflow / health).

Answers the operational question the compact Langfuse trace cannot: *"what share
of traffic is being served by the managed (gpt-5.1) tier right now?"* — the exact
number needed in week 1 and during a GPU-outage failover. Every hook site
(``fallback.emit``/``_record_served``, ``health`` breaker transitions,
``concurrency`` reorders) already fires at the right moment; this module is the
sink they were missing.

Design constraints:
- **Never break boot or a request.** ``prometheus_client`` is an OPTIONAL import;
  when it is absent every ``record_*`` / ``set_*`` call is a silent no-op and
  ``render()`` returns an explanatory comment. This keeps the module safe to ship
  before the dependency lands in every environment.
- **Never raise from a telemetry call.** Each recorder swallows its own errors —
  a metrics bug must never take down a farmer's turn.
- Its own ``CollectorRegistry`` (not the global default) so the ``/metrics``
  endpoint exposes exactly these series and nothing the process imports elsewhere.
"""

from __future__ import annotations

from typing import Optional

try:  # optional dependency — the pipeline runs fine without it (no-op mode)
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    _ENABLED = True
except Exception:  # pragma: no cover - exercised only where the lib is absent
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


def _s(v: object) -> str:
    """Coerce a label value to a bounded string (None -> "none")."""
    if v is None:
        return "none"
    return str(v)


if _ENABLED:
    REGISTRY = CollectorRegistry()

    # Which tier actually served a step (incremented where fallback commits/returns).
    # kind = "oss" | "managed"; provider/model identify the concrete tier.
    _served_total = Counter(
        "llm_served_total",
        "Turns served, by step and the tier that actually served them.",
        ["step", "kind", "provider", "model"],
        registry=REGISTRY,
    )
    # Every classified failure the fallback engine emits (fallbacks AND terminals).
    _fallback_total = Counter(
        "llm_fallback_total",
        "Classified overflow/fallback events by step, reason and outcome.",
        ["step", "reason", "fell_back", "committed"],
        registry=REGISTRY,
    )
    # Per-endpoint circuit-breaker state: 0 closed, 1 half-open, 2 open.
    _breaker_state = Gauge(
        "llm_health_breaker_state",
        "Per-endpoint health breaker state (0=closed, 1=half_open, 2=open).",
        ["endpoint"],
        registry=REGISTRY,
    )
    # Concurrency-gauge deprioritizations (a saturated vLLM tier pushed back).
    _deprioritized_total = Counter(
        "llm_concurrency_deprioritized_total",
        "Times a saturated vLLM tier was deprioritized by the concurrency gauge.",
        ["step"],
        registry=REGISTRY,
    )
    # Last-scraped in-flight (running+waiting) request count per vLLM endpoint.
    _inflight = Gauge(
        "llm_concurrency_inflight",
        "Last-scraped in-flight (running+waiting) requests per vLLM endpoint.",
        ["endpoint"],
        registry=REGISTRY,
    )

_BREAKER_STATE_CODES = {"closed": 0, "half_open": 1, "half-open": 1, "open": 2}


def record_served(step: object, kind: object, provider: object, model: object) -> None:
    """A step was served by ``kind`` (oss/managed) tier ``provider``/``model``.

    Call at the point the walker commits/returns a result (``_record_served`` sites).
    The aggregate ``managed``-share over this counter is the system's core KPI."""
    if not _ENABLED:
        return
    try:
        _served_total.labels(_s(step), _s(kind), _s(provider), _s(model)).inc()
    except Exception:
        pass


def record_fallback(step: object, reason: object, fell_back: object, committed: object) -> None:
    """A classified failure was emitted (mirror of ``fallback.emit``).

    ``fell_back`` True = moved to the next tier; ``committed`` True = failure after
    the stream had already yielded (post-commit, not fallbackable)."""
    if not _ENABLED:
        return
    try:
        _fallback_total.labels(_s(step), _s(reason), _s(bool(fell_back)).lower(), _s(bool(committed)).lower()).inc()
    except Exception:
        pass


def set_breaker_state(endpoint: object, state: object) -> None:
    """Publish a per-endpoint breaker transition (closed/half_open/open)."""
    if not _ENABLED:
        return
    try:
        code = _BREAKER_STATE_CODES.get(_s(state).lower().replace(" ", "_"), 0)
        _breaker_state.labels(_s(endpoint)).set(code)
    except Exception:
        pass


def record_deprioritized(step: object) -> None:
    """A vLLM tier was deprioritized by the concurrency gauge for ``step``."""
    if not _ENABLED:
        return
    try:
        _deprioritized_total.labels(_s(step)).inc()
    except Exception:
        pass


def set_inflight(endpoint: object, value: object) -> None:
    """Publish the last-scraped in-flight request count for a vLLM endpoint."""
    if not _ENABLED:
        return
    try:
        _inflight.labels(_s(endpoint)).set(float(value))
    except Exception:
        pass


def render() -> tuple[bytes, str]:
    """(_body_, _content_type_) for the ``GET /metrics`` endpoint."""
    if not _ENABLED:
        return (
            b"# prometheus_client not installed; llm_core metrics disabled.\n",
            CONTENT_TYPE_LATEST,
        )
    try:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    except Exception:
        return b"# metrics render error\n", CONTENT_TYPE_LATEST


def enabled() -> bool:
    return _ENABLED
