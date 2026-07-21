"""P2: health as a pre-flight chain FILTER — per-endpoint circuit-breaker.

Two composable pieces feed one per-endpoint breaker, and one filter consumes it:

* **Passive breaker** (fed by the fallback failure/success path in
  ``app.services.fallback``): a ``FALLBACKABLE`` classified failure on a tier is a
  ``record_failure(endpoint)``; a clean success is a ``record_success(endpoint)``.
  ``N`` consecutive failures trip the endpoint ``open``; a cooldown lets ONE
  half-open probe through; a real success resets it ``closed``.
* **Active poller** (``app.tasks.health_poller``): periodically GETs the LB
  ``/health`` and reports ``record_healthy_poll`` / ``record_failed_poll``.
  Failback carries **hysteresis** — ``K`` consecutive healthy polls are required
  before an ``open`` endpoint returns to ``closed`` (a single ``/health`` blip
  can't un-trip it, given the H200 crash-and-half-boot history).
* **The filter** ``prune_unhealthy(step, tiers)`` drops the tiers whose endpoint
  is currently ``open`` — so we skip the OSS attempt (and its timeout tax)
  entirely instead of paying it every call during an outage. **Contract: never
  return empty** — if pruning would drop every tier, the input is returned
  unchanged (better to try a suspect tier than have no chain).

The breaker is keyed by **endpoint URL**, so the three independent self-hosted
boxes (agent/OSS, pre-translation, post-translation TranslateGemma) trip and
recover independently.

Gating (the P2 bar — ZERO behaviour change with the flags off):
  * ``record_failure`` / ``record_success``   → no-op unless ``HEALTH_BREAKER_ENABLED``.
  * ``record_healthy_poll`` / ``record_failed_poll`` → no-op unless ``HEALTH_POLLER_ENABLED``.
  * ``prune_unhealthy``                        → returns tiers unchanged unless
    ``HEALTH_BREAKER_ENABLED`` **or** ``HEALTH_POLLER_ENABLED``.

Kept import-clean (stdlib + ``app.config`` + ``config_model``) so the voice repo
can mirror the same public API and the eventual repo-merge stays mechanical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from app.config import settings
from app.llm_core.config_model import Step
from helpers.utils import get_logger

logger = get_logger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"        # healthy; requests flow
    OPEN = "open"            # tripped; endpoint pruned from the chain
    HALF_OPEN = "half_open"  # cooldown elapsed; one probe allowed through


@dataclass(frozen=True)
class BreakerConfig:
    """Trip / cooldown / hysteresis knobs (seconds)."""

    fail_threshold: int = 5          # N consecutive failures -> open
    cooldown_s: float = 30.0         # open -> half_open after this idle window
    healthy_polls_required: int = 3  # K consecutive healthy polls -> closed (hysteresis)


@dataclass
class _EndpointState:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_healthy_polls: int = 0
    opened_at: Optional[float] = None  # monotonic timestamp of the last trip


class HealthRegistry:
    """Per-endpoint breaker state machine (pure mechanics; no flag gating).

    All mutating methods accept an injectable ``now`` (monotonic seconds) so the
    cooldown / hysteresis transitions are testable without real time. The
    module-level convenience functions add the settings gate on top of a single
    process-global instance.
    """

    def __init__(self, config: Optional[BreakerConfig] = None) -> None:
        self._config = config or BreakerConfig()
        self._by_endpoint: dict[str, _EndpointState] = {}

    @property
    def config(self) -> BreakerConfig:
        return self._config

    def _get(self, endpoint: str) -> _EndpointState:
        st = self._by_endpoint.get(endpoint)
        if st is None:
            st = _EndpointState()
            self._by_endpoint[endpoint] = st
        return st

    # ── passive breaker feed ─────────────────────────────────────────────────
    def record_failure(self, endpoint: str, *, now: Optional[float] = None) -> None:
        """A classified infrastructure failure on ``endpoint``.

        Fed by real request failures (fallback) AND failed polls. Any failure
        resets the healthy-poll hysteresis progress. A failure in ``half_open``
        (the probe failed) immediately re-opens the endpoint."""
        if not endpoint:
            return
        now = time.monotonic() if now is None else now
        st = self._get(endpoint)
        st.consecutive_healthy_polls = 0
        st.consecutive_failures += 1
        if st.state is BreakerState.HALF_OPEN:
            st.state = BreakerState.OPEN
            st.opened_at = now
            logger.warning("health: endpoint %s re-opened (half-open probe failed)", endpoint)
            return
        if st.state is BreakerState.CLOSED and st.consecutive_failures >= self._config.fail_threshold:
            st.state = BreakerState.OPEN
            st.opened_at = now
            logger.warning(
                "health: endpoint %s OPEN after %d consecutive failures",
                endpoint, st.consecutive_failures,
            )

    def record_success(self, endpoint: str) -> None:
        """A clean end-to-end request success — the strongest healthy signal, so
        it resets the endpoint ``closed`` immediately (no hysteresis: unlike a
        lightweight ``/health`` poll, a real success proves the whole path)."""
        if not endpoint:
            return
        st = self._get(endpoint)
        was_open = st.state is not BreakerState.CLOSED
        st.consecutive_failures = 0
        st.consecutive_healthy_polls = 0
        st.state = BreakerState.CLOSED
        st.opened_at = None
        if was_open:
            logger.info("health: endpoint %s reset CLOSED on live success", endpoint)

    # ── active poller feed ───────────────────────────────────────────────────
    def record_healthy_poll(self, endpoint: str) -> None:
        """A 200 from the LB ``/health``. Applies hysteresis: only after
        ``healthy_polls_required`` consecutive healthy polls does an ``open`` /
        ``half_open`` endpoint fail back to ``closed``. A healthy poll on an
        already-closed endpoint just clears any partial failure streak."""
        if not endpoint:
            return
        st = self._get(endpoint)
        st.consecutive_healthy_polls += 1
        if st.state is BreakerState.CLOSED:
            st.consecutive_failures = 0
            return
        if st.consecutive_healthy_polls >= self._config.healthy_polls_required:
            logger.info(
                "health: endpoint %s failed back CLOSED after %d healthy polls",
                endpoint, st.consecutive_healthy_polls,
            )
            st.state = BreakerState.CLOSED
            st.consecutive_failures = 0
            st.consecutive_healthy_polls = 0
            st.opened_at = None

    def record_failed_poll(self, endpoint: str, *, now: Optional[float] = None) -> None:
        """A non-200 / unreachable ``/health`` — same evidence as a request
        failure, so it feeds the same trip counter (whole-box death trips it)."""
        self.record_failure(endpoint, now=now)

    # ── read side (the filter consumes this) ─────────────────────────────────
    def is_open(self, endpoint: str, *, now: Optional[float] = None) -> bool:
        """Should ``endpoint`` be pruned right now?

        ``open`` past its cooldown lazily transitions to ``half_open`` and is
        NOT pruned (so a single probe request can re-validate it). ``closed`` and
        ``half_open`` are never pruned; only a still-cooling ``open`` is."""
        if not endpoint:
            return False
        st = self._by_endpoint.get(endpoint)
        if st is None:
            return False
        if st.state is BreakerState.OPEN:
            now = time.monotonic() if now is None else now
            if st.opened_at is not None and (now - st.opened_at) >= self._config.cooldown_s:
                st.state = BreakerState.HALF_OPEN
                logger.info("health: endpoint %s HALF_OPEN (cooldown elapsed, probe allowed)", endpoint)
                return False
            return True
        return False

    def state_of(self, endpoint: str) -> BreakerState:
        st = self._by_endpoint.get(endpoint)
        return st.state if st is not None else BreakerState.CLOSED

    def snapshot(self) -> dict[str, dict]:
        return {
            ep: {
                "state": st.state.value,
                "consecutive_failures": st.consecutive_failures,
                "consecutive_healthy_polls": st.consecutive_healthy_polls,
            }
            for ep, st in self._by_endpoint.items()
        }


def _default_config() -> BreakerConfig:
    return BreakerConfig(
        fail_threshold=settings.health_breaker_fail_threshold,
        cooldown_s=settings.health_breaker_cooldown_ms / 1000.0,
        healthy_polls_required=settings.health_poller_healthy_polls,
    )


# Process-global registry the request path + poller share.
_registry = HealthRegistry(_default_config())


def registry() -> HealthRegistry:
    return _registry


def reset(config: Optional[BreakerConfig] = None) -> HealthRegistry:
    """Replace the global registry (test seam / config reload). ``config=None``
    re-reads the thresholds from settings."""
    global _registry
    _registry = HealthRegistry(config or _default_config())
    return _registry


# ── module-level convenience API (settings-gated; delegates to the global) ────
def record_failure(endpoint: str) -> None:
    if not settings.health_breaker_enabled:
        return
    _registry.record_failure(endpoint)


def record_success(endpoint: str) -> None:
    if not settings.health_breaker_enabled:
        return
    _registry.record_success(endpoint)


def record_healthy_poll(endpoint: str) -> None:
    if not settings.health_poller_enabled:
        return
    _registry.record_healthy_poll(endpoint)


def record_failed_poll(endpoint: str) -> None:
    if not settings.health_poller_enabled:
        return
    _registry.record_failed_poll(endpoint)


def _endpoint_of(tier: Any) -> Optional[str]:
    """Endpoint key for a tier-like object — works for the inert ``Tier``
    (``.endpoint`` is the URL, or ``None`` for OpenAI) and for the materialized
    ``Attempt`` / ``MaterializedTier`` (``.endpoint`` is the URL or ``"managed"``).
    Only real self-hosted URLs ever key a breaker; ``None`` / ``"managed"`` are
    never tracked (we don't poll OpenAI), so they are never pruned here."""
    ep = getattr(tier, "endpoint", None)
    if not ep or ep == "managed":
        return None
    return ep


def prune_unhealthy(step: Optional[Step], tiers: list) -> list:
    """Pre-flight FILTER: drop tiers whose endpoint is currently ``open``.

    Runs BEFORE materialize (on inert ``Tier`` s in the config path) and also on
    the legacy ``Attempt`` chain — both expose ``.endpoint``. **Never returns
    empty**: if every tier would be pruned, the input is returned unchanged
    (degrade-safe). No-op (identity) unless a health flag is on, which is what
    keeps the flags-off path byte-identical.

    NOTE (P3 composition seam): this is the FIRST pre-flight filter. The P3
    concurrency-gauge REORDER runs AFTER this prune and BEFORE materialize —
    ``split.resolve_chain`` calls this, then leaves the reorder hook, then
    materializes. Health prunes known-DOWN tiers; concurrency only DEPRIORITIZES
    saturated (but up) tiers, so composing prune-then-reorder is order-safe."""
    if not (settings.health_breaker_enabled or settings.health_poller_enabled):
        return tiers
    if not tiers:
        return tiers

    # ``is_open`` has a lazy open->half_open side effect, so evaluate it exactly
    # once per tier and reuse the result for both the filter and the trace record.
    open_by_tier = {id(t): _registry.is_open(_endpoint_of(t) or "") for t in tiers}
    kept = [t for t in tiers if not open_by_tier[id(t)]]

    # ── tracing-only (no behaviour change): record which endpoints were pruned
    # and the breaker state consulted per endpoint, onto the current turn's trace.
    from app.llm_core import trace as _trace
    if _trace.current() is not None:
        pruned = [
            ep for t in tiers
            if open_by_tier[id(t)] and (ep := _endpoint_of(t)) is not None
        ]
        breaker_states = {
            ep: _registry.state_of(ep).value
            for t in tiers
            if (ep := _endpoint_of(t)) is not None
        }
        _trace.record_health_prune(step, pruned, breaker_states)

    if not kept:
        # Contract: never return an empty chain. Every tier's endpoint is open —
        # degrade to trying the whole (suspect) chain rather than having none.
        logger.warning(
            "health: prune would empty step=%s chain (all %d endpoints open); "
            "returning chain unchanged (degrade-safe)",
            getattr(step, "value", step), len(tiers),
        )
        return tiers
    if len(kept) != len(tiers):
        logger.info(
            "health: pruned %d/%d unhealthy tier(s) from step=%s chain",
            len(tiers) - len(kept), len(tiers), getattr(step, "value", step),
        )
    return kept
