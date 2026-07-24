"""P3: concurrency-gauge trigger — a pre-flight REORDER filter.

Ports ``bh-voice-prod``'s ``agents/routing.py`` vLLM Prometheus ``/metrics``
scrape into ``llm_core``, but as a **deprioritize reorder**, not bh's hard flip:

* **bh:** ``is_gemma_candidate`` picks a % canary, then *"gemma busy -> route this
  session to the closed-source model instead"* — an inversion of primary/secondary
  driven by a max-concurrency gate.
* **here:** *"the vLLM tier is saturated -> move it toward the BACK of the
  already-resolved chain so the managed tier (already the NEXT tier) is tried
  first while the box is hot."* There is **no inversion**: "primary" is just index
  0 of the chain, and this filter only REORDERS — it never drops a tier and never
  returns empty. A saturated-but-up vLLM primary still runs if it is the only
  tier. bh's "flip to closed-source when gemma busy" is reproduced purely as
  "deprioritize the vLLM tier so the managed tier already sitting next in the
  chain is tried first" — no separate primary/secondary swap exists to invert.

Two hardening changes vs bh (plan §2):
  (a) the metrics URL comes from **explicit config** (``ConcurrencyGate.metrics_url``
      on the step's triggers), never bh's fragile ``re.sub(r'/v1/?$', '', base)``
      derivation off the inference endpoint;
  (b) unreadable metrics **fail open** — treated as NOT saturated (chain order
      left unchanged), never a forced flip to managed. (bh's ``azure_fallback`` on
      a metrics read error was the opposite bias; the plan reverses it so an
      observability blip cannot dump 100% of traffic onto the managed tier.)

Two aggregation/shed improvements over the naive single-scrape bang-bang:

  1. **Fleet-aggregate load via Prometheus (optional).** A single vLLM
     ``/metrics`` scrape behind an nginx ``least_conn`` LB samples the
     LEAST-loaded replica, under-reads the true fleet load, and sheds late/never.
     When ``CONCURRENCY_PROMETHEUS_URL`` is set, the gauge instead comes from the
     Prometheus HTTP API (``/api/v1/query`` of ``CONCURRENCY_PROMETHEUS_QUERY``,
     default ``sum(vllm:num_requests_running + vllm:num_requests_waiting)``) — the
     summed fleet aggregate. The single-endpoint ``/metrics`` scrape remains the
     fallback when the Prometheus URL is unset. Both paths share the same short
     Redis cache and the same FAIL-OPEN posture (any read error -> ``None``).

  2. **Smooth probabilistic shed (not bang-bang at the cap).** The old filter
     flipped 100% of gated traffic the instant the gauge reached
     ``max_concurrency`` and back the instant it dropped — with a 2s-cached gauge
     shared across workers, that synchronized flip/flip-back is a herd
     oscillation. Instead, each call deprioritizes with a PROBABILITY that rises
     with load: 0 below ``shed_start`` (``CONCURRENCY_SHED_START_FRAC`` * cap,
     default 0.7*cap), ramping linearly to 1.0 AT the cap (and staying 1.0 above
     it). So as load approaches the ceiling, only a self-proportioning *fraction*
     of gated calls shed — "shed SOME gemma" — smoothing the boundary instead of
     an all-or-nothing cliff, while the configured cap is still a hard 100%-shed
     ceiling. (Normal Python runtime ``random`` — the "no ``Math.random()``"
     rule is about deterministic-config contexts, not this load shed.)

Composition (fixed order, plan §2): ``health-prune -> concurrency-reorder ->
materialize -> classify-walk``. Health prunes known-DOWN tiers FIRST, so a down
tier is already gone before this reorder runs and can never be reordered back to
the front — this filter only ever touches saturated-but-UP vLLM tiers.

Gated by ``CONCURRENCY_GAUGE_ENABLED`` (default off) and only active where a
``ConcurrencyGate`` is configured on the step; both off => identity (zero
behaviour change). Kept import-clean (stdlib + httpx + ``app.config`` + the app
cache + ``config_model`` + ``app.metrics``) so the voice repo can mirror the same
public API and the eventual repo-merge stays mechanical.
"""

from __future__ import annotations

import os
import random
import re
from typing import Optional

from app import metrics
from app.config import settings
from app.core.cache import cache
from app.llm_core.config_model import ConcurrencyGate, Provider, Step
from helpers.utils import get_logger

logger = get_logger(__name__)

# Per-source cache namespace (bh used a single fixed key; we key by the concrete
# read source — the single ``/metrics`` URL, or the Prometheus query — so
# independent boxes / the fleet aggregate cache independently). Short TTL, shared
# across workers.
_CACHE_KEY_PREFIX = "llm_core_concurrency:"

# Default Prometheus aggregate query: the summed fleet in-flight request count.
_DEFAULT_PROM_QUERY = "sum(vllm:num_requests_running + vllm:num_requests_waiting)"

# Fraction of ``max_concurrency`` at which probabilistic shedding STARTS ramping
# from 0. At/above the cap the shed probability is a hard 1.0. Env-tunable.
_SHED_START_FRAC = float(os.getenv("CONCURRENCY_SHED_START_FRAC", "0.7"))

# vLLM Prometheus gauges summed into one in-flight number (identical to bh).
_NUM_RE = re.compile(
    r"^(vllm:num_requests_running|vllm:num_requests_waiting)\{.*\}\s+([\d.eE+-]+)$"
)


async def _fetch_concurrency(metrics_url: str) -> Optional[int]:
    """Sum ``num_requests_running + num_requests_waiting`` from a single vLLM
    ``/metrics`` endpoint, or ``None`` on ANY read failure (the fail-open signal).

    NOTE: behind an nginx ``least_conn`` LB this samples ONE (the least-loaded)
    replica; prefer the Prometheus aggregate path when a fleet-wide read matters."""
    import httpx  # lazy: keep module import side-effect-free

    timeout_s = settings.concurrency_metrics_timeout_ms / 1000.0
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(metrics_url)
            resp.raise_for_status()
    except Exception as e:  # HTTP error / timeout / connection refused -> fail-open
        logger.warning("concurrency: metrics fetch failed for %s: %s", metrics_url, e)
        return None

    total = 0
    for line in resp.text.splitlines():
        m = _NUM_RE.match(line)
        if m:
            total += int(float(m.group(2)))
    return total


async def _fetch_concurrency_prometheus(prom_url: str, query: str) -> Optional[int]:
    """Query the Prometheus HTTP API for a scalar/vector aggregate and sum its
    result values into one in-flight number, or ``None`` on ANY read failure
    (fail-open). This reads the WHOLE fleet's load (``sum(...)`` across replicas),
    unlike a single ``/metrics`` scrape that a ``least_conn`` LB biases low."""
    import httpx  # lazy: keep module import side-effect-free

    timeout_s = settings.concurrency_metrics_timeout_ms / 1000.0
    url = prom_url.rstrip("/") + "/api/v1/query"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, params={"query": query})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:  # HTTP error / timeout / bad JSON -> fail-open
        logger.warning("concurrency: prometheus query failed for %s (%s): %s", url, query, e)
        return None

    if not isinstance(payload, dict) or payload.get("status") != "success":
        logger.warning("concurrency: prometheus non-success payload for %s: %r", url, payload)
        return None

    result = (payload.get("data") or {}).get("result") or []
    total = 0.0
    saw_value = False
    for series in result:
        value = series.get("value") if isinstance(series, dict) else None
        if value and len(value) >= 2:
            try:
                total += float(value[1])
                saw_value = True
            except (TypeError, ValueError):
                continue
    if not saw_value:
        # Empty result vector (no series yet / query matched nothing). Treat as a
        # read miss -> fail-open (None) rather than asserting a real "0 in flight".
        logger.info("concurrency: prometheus query %s returned no series (fail-open)", query)
        return None
    return int(total)


async def get_concurrency(metrics_url: str) -> Optional[int]:
    """Short-TTL Redis-cached read of the vLLM in-flight request count.

    Source selection: if ``CONCURRENCY_PROMETHEUS_URL`` is set, the value is the
    FLEET AGGREGATE from the Prometheus HTTP API (``CONCURRENCY_PROMETHEUS_QUERY``);
    otherwise it is the single ``metrics_url`` ``/metrics`` scrape (the fallback).

    Cache errors degrade to a direct fetch (a Redis blip must never break the
    request path); a fetch failure returns ``None`` — the fail-open signal the
    reorder honors (treat as NOT saturated). A successful fresh read publishes
    ``metrics.set_inflight`` for the read source."""
    prom_url = os.getenv("CONCURRENCY_PROMETHEUS_URL")
    prom_url = prom_url.strip() if prom_url else ""

    if prom_url:
        query = os.getenv("CONCURRENCY_PROMETHEUS_QUERY", _DEFAULT_PROM_QUERY)
        cache_source = f"prom:{prom_url}:{query}"
        inflight_label = prom_url

        async def _do_fetch() -> Optional[int]:
            return await _fetch_concurrency_prometheus(prom_url, query)
    else:
        if not metrics_url:
            return None
        cache_source = metrics_url
        inflight_label = metrics_url

        async def _do_fetch() -> Optional[int]:
            return await _fetch_concurrency(metrics_url)

    key = f"{_CACHE_KEY_PREFIX}{cache_source}"
    try:
        cached = await cache.get(key)
    except Exception as e:  # Redis down -> direct fetch, don't break routing
        logger.warning("concurrency: cache read failed for %s: %s", cache_source, e)
        cached = None
    if cached is not None:
        return cached

    value = await _do_fetch()
    if value is not None:
        metrics.set_inflight(inflight_label, value)  # no-op if prom lib absent; never raises
        try:
            await cache.set(key, value, ttl=settings.concurrency_metrics_cache_ttl_s)
        except Exception as e:  # persistence best-effort
            logger.warning("concurrency: cache write failed for %s: %s", cache_source, e)
    return value


def _is_vllm(tier) -> bool:
    return getattr(tier, "provider", None) is Provider.VLLM


def _shed_probability(gauge: int, max_concurrency: int) -> float:
    """Probability that a gated call deprioritizes the vLLM tier at ``gauge``
    in-flight requests.

    * ``gauge <= shed_start`` (``_SHED_START_FRAC`` * cap): 0.0 — no shed.
    * ``shed_start < gauge < cap``: linear ramp in (0, 1) — shed a rising,
      self-proportioning fraction (the smooth band).
    * ``gauge >= cap``: 1.0 — the configured cap is a hard 100%-shed ceiling.

    A non-positive cap degrades to "any load sheds fully" (defensive)."""
    if max_concurrency <= 0:
        return 1.0 if gauge > 0 else 0.0
    shed_start = _SHED_START_FRAC * max_concurrency
    if gauge >= max_concurrency:
        return 1.0
    if gauge <= shed_start:
        return 0.0
    return (gauge - shed_start) / (max_concurrency - shed_start)


async def reprioritize_by_load(
    step: Optional[Step], tiers: list, gate: Optional[ConcurrencyGate]
) -> list:
    """Pre-flight FILTER #2: DEPRIORITIZE a saturated vLLM tier (reorder to back).

    Contract (differs from the P2 health prune, which DROPS a down tier): never
    drops a tier, never returns empty — only ever REORDERS. Returns ``tiers``
    unchanged (identity) when:

    * ``CONCURRENCY_GAUGE_ENABLED`` is off;
    * the step has no ``ConcurrencyGate`` configured (``gate is None``) — a step
      without a gate is untouched;
    * the gauge is unreadable (``None``) — **fail-open**: treat as NOT saturated;
    * the probabilistic shed roll declines this call (below the ramp, or a losing
      draw inside the smooth band).

    When the shed roll fires, WHERE the shed request is routed depends on the gate
    (M3 — the overflow target is separately configurable, req #3 "set separately,
    no mixing of variables from the session-% selection"):

    * ``gate.overflow_tier`` is SET -> the shed request is routed to THAT
      explicitly chosen model — independent of the session-% profile's own chain.
      The overflow tier is moved to the FRONT (tried first), the original tiers
      following as further fallback: ``[overflow_tier, *original]`` (deduped if the
      overflow tier is already present, so it never appears twice).
    * ``gate.overflow_tier`` is UNSET (``None``, the default) -> byte-identical to
      the prior behaviour: the vLLM tiers are STABLY moved behind the non-vLLM
      (managed) tiers, so the managed tier — already the next tier in the chain —
      is tried first while the box is hot. In the single vLLM-primary +
      managed-fallback config this is exactly "deprioritize the primary"; "primary"
      is only index 0, and no primary/secondary inversion exists (see the module
      docstring).

    Shed probability rises with load (``_shed_probability``): a self-proportioning
    fraction near the cap, a hard 1.0 at/above it.

    Runs on the already-HEALTH-PRUNED inert ``Tier`` list, BEFORE materialize, so
    a tier the health filter already dropped is gone and can never be reordered
    back to the front here.
    """
    if not settings.concurrency_gauge_enabled:
        return tiers
    if gate is None:
        return tiers            # step without a configured gate -> untouched
    if not tiers:
        return tiers

    gauge = await get_concurrency(gate.metrics_url)

    # ── tracing-only (no behaviour change): record the gauge read + whether the
    # vLLM tier was deprioritized onto the current turn's trace, at each outcome.
    from app.llm_core import trace as _trace

    def _record(deprioritized: bool) -> None:
        _trace.record_concurrency(
            step,
            gauge=gauge,
            max_concurrency=gate.max_concurrency,
            deprioritized=deprioritized,
            metrics_url=gate.metrics_url,
        )

    if gauge is None:
        # Fail-open: metrics unreadable -> assume NOT saturated -> order unchanged.
        # (Deliberately NOT a forced flip to managed — the plan's reversal of bh.)
        logger.info(
            "concurrency: metrics unreadable for step=%s (%s); fail-open, order unchanged",
            getattr(step, "value", step), gate.metrics_url,
        )
        _record(False)
        return tiers

    # Probabilistic proportional shed: below the ramp -> never; in the band -> a
    # rising fraction; at/above the cap -> always. Smooths the 2s-cache herd flip.
    p = _shed_probability(gauge, gate.max_concurrency)
    if p <= 0.0 or random.random() >= p:
        _record(False)
        return tiers            # this call does not shed -> primary stays primary

    # ── M3: separately-configurable overflow target ──────────────────────────
    # The shed roll fired. WHERE the shed request goes depends on the gate:
    #
    #   * ``overflow_tier`` SET -> route the shed request to THAT explicitly chosen
    #     model, independent of the session-% profile's own chain (req #3 "set
    #     separately, no mixing"). Move the overflow tier to the FRONT (tried first)
    #     with the original tiers following as further fallback -> [overflow, *orig].
    #     Dedupe: if the overflow tier is already in the chain, it is lifted to the
    #     front (removed from its old position) so it never appears twice.
    #
    #   * ``overflow_tier`` UNSET (None, the default) -> today's behaviour, byte-
    #     identical: DEPRIORITIZE the saturated vLLM tier(s) behind the managed
    #     tier(s) already sitting next in the profile's chain.
    if gate.overflow_tier is not None:
        overflow = gate.overflow_tier
        reordered = [overflow] + [t for t in tiers if t != overflow]
        logger.info(
            "concurrency: step=%s vLLM gauge %d vs cap %d (p_shed=%.2f); routing shed request to configured overflow tier %s (front)",
            getattr(step, "value", step), gauge, gate.max_concurrency, p,
            getattr(overflow, "label", None) or getattr(overflow, "model", overflow),
        )
        _record(True)
        metrics.record_deprioritized(step)  # no-op if prom lib absent; never raises
        return reordered

    vllm = [t for t in tiers if _is_vllm(t)]
    others = [t for t in tiers if not _is_vllm(t)]
    if not vllm or not others:
        # All-vLLM or no-vLLM chain: nothing to reorder behind. Never churn /
        # empty — a saturated-but-only vLLM tier still runs (degrade-safe).
        _record(False)
        return tiers

    logger.info(
        "concurrency: step=%s vLLM gauge %d vs cap %d (p_shed=%.2f); deprioritizing %d vLLM tier(s) behind managed",
        getattr(step, "value", step), gauge, gate.max_concurrency, p, len(vllm),
    )
    _record(True)
    metrics.record_deprioritized(step)  # no-op if prom lib absent; never raises
    return others + vllm
