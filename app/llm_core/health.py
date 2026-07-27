"""P2: health as a pre-flight chain FILTER — per-endpoint circuit-breaker.

Two composable pieces feed one per-endpoint breaker, and one filter consumes it:

* **Passive breaker** (fed by the fallback failure/success path in
  ``app.services.fallback``): a ``FALLBACKABLE`` classified failure on a tier is a
  ``record_failure(endpoint)``; a clean success is a ``record_success(endpoint)``.
  The endpoint trips ``open`` on EITHER of two signals:
    * ``N`` **consecutive** failures (whole-box death — every request errors), OR
    * a rolling-window **failure RATE** above a threshold (a *brownout* — the box
      answers 40–60% of requests, so each success resets the consecutive counter
      and the consecutive trip NEVER fires, yet every failing request still pays
      the full timeout tax). The rolling window (last ``HEALTH_FAIL_RATE_WINDOW``
      outcomes) catches exactly that case: once the window is full and the
      failure share exceeds ``HEALTH_FAIL_RATE_THRESHOLD``, the next failure
      trips ``open``.
  A cooldown lets ONE half-open probe through (a single in-flight probe token —
  the FIRST caller after the cooldown probes; concurrent callers still see the
  endpoint open and are pruned, so the dead box is not re-flooded during the
  probe window). A real success resets it ``closed`` immediately. The probe token
  is freed on ANY terminal outcome of the probed request: the fallback walkers
  call ``release_probe`` in a ``finally`` for every chain tier, so a probe that
  ended without a definitive success / ``BREAKER_EVIDENCE`` failure (a 4xx / caller
  ``TypeError`` -> UNKNOWN, or BAD_OUTPUT) — or a probed tier a concurrency reorder
  left unexecuted — cannot leak the token and pin the endpoint HALF_OPEN forever. A
  ``HEALTH_PROBE_MAX_S`` time-box in ``is_open`` is the backstop that auto-releases
  a lost token, so a recovered box can never be pruned forever.
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
recover independently. Every state transition (closed/half_open/open) is
published to Prometheus via ``app.metrics.set_breaker_state``.

Gating (the P2 bar — ZERO behaviour change with the flags off):
  * ``record_failure`` / ``record_success``   → no-op unless ``HEALTH_BREAKER_ENABLED``.
  * ``record_healthy_poll`` / ``record_failed_poll`` → no-op unless ``HEALTH_POLLER_ENABLED``.
  * ``prune_unhealthy``                        → returns tiers unchanged unless
    ``HEALTH_BREAKER_ENABLED`` **or** ``HEALTH_POLLER_ENABLED``.

Kept import-clean (stdlib + ``app.config`` + ``config_model`` + ``app.metrics``,
itself dependency-optional/no-op) so the voice repo can mirror the same public API
and the eventual repo-merge stays mechanical.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app import metrics
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
    # Rolling-window failure-RATE trip (brownout coverage). The breaker also trips
    # when, over the last ``fail_rate_window`` outcomes, the failure share exceeds
    # ``fail_rate_threshold``. The window must be FULL before the rate can trip, so
    # a short burst never fires it (that is the consecutive-failure trip's job) —
    # this is purely the "40–60% flaky box that never trips consecutively" case.
    fail_rate_window: int = 20       # rolling outcome window size (0 disables the rate trip)
    fail_rate_threshold: float = 0.5  # failure share (>, strict) over a FULL window -> open
    # Half-open probe time-box (backstop). A granted probe token is normally freed by
    # the fallback walker's ``finally`` (``release_probe``) on ANY terminal outcome; if
    # that release is ever missed (a lost token), ``is_open`` auto-releases the token
    # after this many seconds so a recovered endpoint can be re-probed instead of
    # staying HALF_OPEN (and thus pruned) forever.
    probe_max_s: float = 30.0


@dataclass
class _EndpointState:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_healthy_polls: int = 0
    opened_at: Optional[float] = None  # monotonic timestamp of the last trip
    # Rolling window of recent outcomes (True=success, False=failure) for the
    # failure-RATE trip; bounded to the configured window in ``_push_outcome``.
    outcomes: deque = field(default_factory=deque)
    # Single half-open probe token: True while ONE caller holds the in-flight
    # probe after the cooldown elapsed. Concurrent callers see the endpoint as
    # still open (pruned) until the probe resolves (success->closed / failure->open).
    probe_in_flight: bool = False
    # Monotonic timestamp when the current half-open probe token was granted; lets
    # ``is_open`` time-box a lost probe (see BreakerConfig.probe_max_s). None whenever
    # no probe is in flight.
    probe_started_at: Optional[float] = None


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

    def _push_outcome(self, st: _EndpointState, ok: bool) -> None:
        """Append an outcome to the rolling window and trim to the configured
        size. Fed by BOTH failures and live successes so the failure-RATE reflects
        a realistic brownout mix (each success does NOT clear the window — clearing
        it on every success is exactly what let the brownout hide from the old
        consecutive-only trip)."""
        st.outcomes.append(ok)
        window = self._config.fail_rate_window
        while len(st.outcomes) > window:
            st.outcomes.popleft()

    def _rate_tripped(self, st: _EndpointState) -> bool:
        """True when the rolling window is FULL and its failure share strictly
        exceeds the threshold. Requires a full window so a short failure burst
        (already covered by the consecutive trip) never fires this."""
        window = self._config.fail_rate_window
        if window <= 0 or len(st.outcomes) < window:
            return False
        fails = sum(1 for ok in st.outcomes if not ok)
        return (fails / len(st.outcomes)) > self._config.fail_rate_threshold

    def _emit(self, endpoint: str, state: BreakerState) -> None:
        """Publish a breaker transition to Prometheus (no-op if the lib is absent;
        never raises)."""
        try:
            metrics.set_breaker_state(endpoint, state.value)
        except Exception:  # pragma: no cover - telemetry must never break routing
            pass

    # ── passive breaker feed ─────────────────────────────────────────────────
    def record_failure(self, endpoint: str, *, now: Optional[float] = None) -> None:
        """A classified infrastructure failure on ``endpoint``.

        Fed by real request failures (fallback) AND failed polls. Any failure
        resets the healthy-poll hysteresis progress. A failure in ``half_open``
        (the probe failed) immediately re-opens the endpoint. A failure while
        already ``open`` REFRESHES the cooldown (continuous failures extend the
        open window instead of letting the cooldown lapse). While ``closed`` the
        endpoint trips on EITHER N consecutive failures OR a full-window failure
        rate above threshold (brownout)."""
        if not endpoint:
            return
        now = time.monotonic() if now is None else now
        st = self._get(endpoint)
        st.consecutive_healthy_polls = 0
        st.consecutive_failures += 1
        self._push_outcome(st, ok=False)

        if st.state is BreakerState.HALF_OPEN:
            st.state = BreakerState.OPEN
            st.opened_at = now
            st.probe_in_flight = False
            st.probe_started_at = None
            self._emit(endpoint, BreakerState.OPEN)
            logger.warning("health: endpoint %s re-opened (half-open probe failed)", endpoint)
            return
        if st.state is BreakerState.OPEN:
            # Already tripped: refresh the cooldown so a steady failure stream keeps
            # the endpoint open rather than half-opening mid-outage.
            st.opened_at = now
            self._emit(endpoint, BreakerState.OPEN)
            return
        # CLOSED: trip on consecutive-failure OR rolling failure-rate (brownout).
        if st.consecutive_failures >= self._config.fail_threshold:
            st.state = BreakerState.OPEN
            st.opened_at = now
            self._emit(endpoint, BreakerState.OPEN)
            logger.warning(
                "health: endpoint %s OPEN after %d consecutive failures",
                endpoint, st.consecutive_failures,
            )
        elif self._rate_tripped(st):
            fails = sum(1 for ok in st.outcomes if not ok)
            st.state = BreakerState.OPEN
            st.opened_at = now
            self._emit(endpoint, BreakerState.OPEN)
            logger.warning(
                "health: endpoint %s OPEN on brownout (%d/%d failures in window > %.0f%%)",
                endpoint, fails, len(st.outcomes), self._config.fail_rate_threshold * 100,
            )

    def record_success(self, endpoint: str) -> None:
        """A clean end-to-end request success — the strongest healthy signal, so
        it resets the endpoint ``closed`` immediately (no hysteresis: unlike a
        lightweight ``/health`` poll, a real success proves the whole path). The
        outcome is still appended to the rolling window (WITHOUT clearing it) so a
        brownout box's failure-rate keeps accumulating across its intermittent
        successes."""
        if not endpoint:
            return
        st = self._get(endpoint)
        was_open = st.state is not BreakerState.CLOSED
        st.consecutive_failures = 0
        st.consecutive_healthy_polls = 0
        st.probe_in_flight = False
        st.probe_started_at = None
        st.state = BreakerState.CLOSED
        st.opened_at = None
        self._push_outcome(st, ok=True)
        if was_open:
            self._emit(endpoint, BreakerState.CLOSED)
            logger.info("health: endpoint %s reset CLOSED on live success", endpoint)

    def release_probe(self, endpoint: str) -> None:
        """Idempotently free the single half-open probe token for ``endpoint``.

        Called by the fallback walkers in a ``finally`` for every chain tier so a
        probe granted by ``is_open`` (on the open->half_open transition) is released
        no matter HOW the probed request ended — a clean success, ANY classified
        failure, a caller cancellation, or a tier that never ran because a
        concurrency reorder moved it and an earlier tier already returned/raised.
        This ONLY clears the probe slot (and its time-box stamp); it deliberately
        does NOT touch breaker state / counters / the rolling window, so it can never
        perturb the ``record_success`` / ``record_failure`` transitions. No-op when no
        probe is in flight (or the endpoint is unknown / untracked)."""
        if not endpoint:
            return
        st = self._by_endpoint.get(endpoint)
        if st is not None and st.probe_in_flight:
            st.probe_in_flight = False
            st.probe_started_at = None

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
            st.probe_in_flight = False
            st.probe_started_at = None
            self._emit(endpoint, BreakerState.CLOSED)

    def record_failed_poll(self, endpoint: str, *, now: Optional[float] = None) -> None:
        """A non-200 / unreachable ``/health`` — same evidence as a request
        failure, so it feeds the same trip counter (whole-box death trips it) and
        the same cooldown refresh when already open."""
        self.record_failure(endpoint, now=now)

    # ── read side (the filter consumes this) ─────────────────────────────────
    def is_open(self, endpoint: str, *, now: Optional[float] = None) -> bool:
        """Should ``endpoint`` be pruned right now?

        ``open`` past its cooldown lazily transitions to ``half_open`` and grants
        the SINGLE probe token to THIS caller (returns False → not pruned) so one
        request can re-validate the box. While ``half_open`` with the probe token
        already held by another caller, this returns True (pruned) — so the dead
        box is probed by exactly one request, not re-flooded. ``closed`` is never
        pruned; a still-cooling ``open`` (or a half-open whose probe is in flight
        elsewhere) is."""
        if not endpoint:
            return False
        st = self._by_endpoint.get(endpoint)
        if st is None:
            return False
        if st.state is BreakerState.OPEN:
            now = time.monotonic() if now is None else now
            if st.opened_at is not None and (now - st.opened_at) >= self._config.cooldown_s:
                st.state = BreakerState.HALF_OPEN
                st.probe_in_flight = True          # grant the single probe to THIS caller
                st.probe_started_at = now
                self._emit(endpoint, BreakerState.HALF_OPEN)
                logger.info("health: endpoint %s HALF_OPEN (cooldown elapsed, single probe allowed)", endpoint)
                return False
            return True
        if st.state is BreakerState.HALF_OPEN:
            now = time.monotonic() if now is None else now
            # Time-box backstop: if the probe token has been held longer than
            # ``probe_max_s`` the fallback walker's ``finally`` release was missed
            # (a lost token). Auto-release it here so the endpoint is re-probed
            # instead of staying HALF_OPEN — and thus pruned — forever (the exact
            # failure this system prevents when the poller is off).
            if (
                st.probe_in_flight
                and st.probe_started_at is not None
                and (now - st.probe_started_at) >= self._config.probe_max_s
            ):
                st.probe_in_flight = False
                st.probe_started_at = None
                logger.warning(
                    "health: endpoint %s half-open probe token timed out after %.0fs; "
                    "auto-releasing (backstop)", endpoint, self._config.probe_max_s,
                )
            if st.probe_in_flight:
                return True                        # another caller holds the probe -> prune
            # Probe token free (a prior probe resolved without a definitive
            # success/failure, was released by the walker, or timed out above) ->
            # grant it to THIS caller.
            st.probe_in_flight = True
            st.probe_started_at = now
            return False
        return False

    def state_of(self, endpoint: str) -> BreakerState:
        st = self._by_endpoint.get(endpoint)
        return st.state if st is not None else BreakerState.CLOSED

    def snapshot(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for ep, st in self._by_endpoint.items():
            n = len(st.outcomes)
            fails = sum(1 for ok in st.outcomes if not ok)
            out[ep] = {
                "state": st.state.value,
                "consecutive_failures": st.consecutive_failures,
                "consecutive_healthy_polls": st.consecutive_healthy_polls,
                "window_size": n,
                "window_failures": fails,
                "failure_rate": (fails / n) if n else 0.0,
                "probe_in_flight": st.probe_in_flight,
            }
        return out


def _default_config() -> BreakerConfig:
    return BreakerConfig(
        fail_threshold=settings.health_breaker_fail_threshold,
        cooldown_s=settings.health_breaker_cooldown_ms / 1000.0,
        healthy_polls_required=settings.health_poller_healthy_polls,
        fail_rate_window=int(os.getenv("HEALTH_FAIL_RATE_WINDOW", "20")),
        fail_rate_threshold=float(os.getenv("HEALTH_FAIL_RATE_THRESHOLD", "0.5")),
        probe_max_s=float(os.getenv("HEALTH_PROBE_MAX_S", "30")),
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


def release_probe(endpoint: str) -> None:
    """Free any half-open probe token held for ``endpoint`` (idempotent).

    Self-gated on the same flags that let ``prune_unhealthy`` grant a probe in the
    first place, so the flags-off path stays byte-identical (no-op)."""
    if not (settings.health_breaker_enabled or settings.health_poller_enabled):
        return
    _registry.release_probe(endpoint)


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
