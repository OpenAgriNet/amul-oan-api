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

Composition (fixed order, plan §2): ``health-prune -> concurrency-reorder ->
materialize -> classify-walk``. Health prunes known-DOWN tiers FIRST, so a down
tier is already gone before this reorder runs and can never be reordered back to
the front — this filter only ever touches saturated-but-UP vLLM tiers.

Gated by ``CONCURRENCY_GAUGE_ENABLED`` (default off) and only active where a
``ConcurrencyGate`` is configured on the step; both off => identity (zero
behaviour change). Kept import-clean (stdlib + httpx + ``app.config`` + the app
cache + ``config_model``) so the voice repo can mirror the same public API and the
eventual repo-merge stays mechanical.
"""

from __future__ import annotations

import re
from typing import Optional

from app.config import settings
from app.core.cache import cache
from app.llm_core.config_model import ConcurrencyGate, Provider, Step
from helpers.utils import get_logger

logger = get_logger(__name__)

# Per-metrics-url cache namespace (bh used a single fixed key; we key by URL so
# independent boxes cache independently). Short TTL, shared across workers.
_CACHE_KEY_PREFIX = "llm_core_concurrency:"

# vLLM Prometheus gauges summed into one in-flight number (identical to bh).
_NUM_RE = re.compile(
    r"^(vllm:num_requests_running|vllm:num_requests_waiting)\{.*\}\s+([\d.eE+-]+)$"
)


async def _fetch_concurrency(metrics_url: str) -> Optional[int]:
    """Sum ``num_requests_running + num_requests_waiting`` from a vLLM ``/metrics``
    endpoint, or ``None`` on ANY read failure (the fail-open signal)."""
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


async def get_concurrency(metrics_url: str) -> Optional[int]:
    """Short-TTL Redis-cached read of a vLLM engine's in-flight request count.

    Cache errors degrade to a direct fetch (a Redis blip must never break the
    request path); a fetch failure returns ``None`` — the fail-open signal the
    reorder honors (treat as NOT saturated)."""
    if not metrics_url:
        return None
    key = f"{_CACHE_KEY_PREFIX}{metrics_url}"
    try:
        cached = await cache.get(key)
    except Exception as e:  # Redis down -> direct fetch, don't break routing
        logger.warning("concurrency: cache read failed for %s: %s", metrics_url, e)
        cached = None
    if cached is not None:
        return cached

    value = await _fetch_concurrency(metrics_url)
    if value is not None:
        try:
            await cache.set(key, value, ttl=settings.concurrency_metrics_cache_ttl_s)
        except Exception as e:  # persistence best-effort
            logger.warning("concurrency: cache write failed for %s: %s", metrics_url, e)
    return value


def _is_vllm(tier) -> bool:
    return getattr(tier, "provider", None) is Provider.VLLM


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
    * the gauge is below ``gate.max_concurrency``.

    Only when the gauge is at/above threshold are the vLLM tiers STABLY moved
    behind the non-vLLM (managed) tiers, so the managed tier — already the next
    tier in the chain — is tried first while the box is hot. In the single
    vLLM-primary + managed-fallback config this is exactly "deprioritize the
    primary"; "primary" is only index 0, and no primary/secondary inversion
    exists (see the module docstring).

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
    if gauge < gate.max_concurrency:
        _record(False)
        return tiers            # below threshold -> primary stays primary

    vllm = [t for t in tiers if _is_vllm(t)]
    others = [t for t in tiers if not _is_vllm(t)]
    if not vllm or not others:
        # All-vLLM or no-vLLM chain: nothing to reorder behind. Never churn /
        # empty — a saturated-but-only vLLM tier still runs (degrade-safe).
        _record(False)
        return tiers

    logger.info(
        "concurrency: step=%s vLLM gauge %d >= %d; deprioritizing %d vLLM tier(s) behind managed",
        getattr(step, "value", step), gauge, gate.max_concurrency, len(vllm),
    )
    _record(True)
    return others + vllm
