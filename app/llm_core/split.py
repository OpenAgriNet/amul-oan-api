"""P1: weighted named-profile split + config-driven attempt chain.

This is the generalization of two hardwired pieces:

* ``pipeline_router._deterministic_variant`` — a *single* OSS/legacy bit from
  ``int(sha256(session_id)[:8], 16) % 100 < OSS_PIPELINE_PCT`` — becomes an
  assignment into one of *N* weighted :class:`NamedProfile` s via **cumulative
  weight buckets over the SAME hash**. Bit-compatible by construction: the shim's
  ``[oss(pct), managed(100-pct)]`` config puts the ``oss`` profile in buckets
  ``[0, pct)`` and ``managed`` in ``[pct, 100)`` — exactly today's
  ``bucket < pct -> oss`` boundary.

* ``fallback.attempt_chain``'s hardwired ``[oss, managed]`` becomes the resolved
  profile's ordered ``StepConfig.tiers``. Provider clients remain lazy until an
  attempt is actually reached.

Stickiness is the deterministic hash bucket itself — no Redis state. Same
``session_id`` + same weights -> same profile (stable within a config version);
a weight change re-maps the bucket so continuing sessions FOLLOW the new % on a
redeploy / config change rather than freezing on the old model. (``pipeline_router``
pinned the profile name in Redis, which froze sessions across weight changes —
deliberately dropped: it defeats the refresh-on-change contract.)

"""

from __future__ import annotations

import hashlib
from typing import Optional

from helpers.utils import get_logger
from app.llm_core import runtime
from app.llm_core.config_model import PipelineConfig, Step
from app.llm_core.factory import STEP_CLIENT_KIND, tier_client_kind

logger = get_logger(__name__)

def _bucket(session_id: str) -> int:
    """The exact bucket pipeline_router uses: 0-99 from a stable sha256 of the id.

    Kept character-for-character identical to
    ``pipeline_router._deterministic_variant`` so the two split implementations
    place any given session in the same slice of the 0-99 space."""
    digest = hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def deterministic_profile(session_id: str, pipeline: PipelineConfig) -> str:
    """Assign a session to a profile by cumulative weight buckets over ``_bucket``.

    Profiles are consumed in declared order; profile ``i`` owns the half-open
    bucket range ``[sum(weights[:i]), sum(weights[:i+1]))``. Weights sum to 100
    (enforced by ``PipelineConfig``), so the final profile is the catch-all — the
    trailing return is a defensive fail-safe only."""
    bucket = _bucket(session_id)
    cumulative = 0
    for profile in pipeline.profiles:
        cumulative += profile.weight
        if bucket < cumulative:
            return profile.name
    return pipeline.profiles[-1].name


async def resolve_profile(
    session_id: str, pipeline: Optional[PipelineConfig] = None
) -> str:
    """Deterministic weighted-profile assignment for a session (profile NAME).

    The ``sha256(session_id)`` bucket IS the sticky key: same ``session_id`` +
    same weights -> same profile, so a session stays on one model within a config
    version (no mid-session flapping) with zero Redis state. A weight change
    re-maps the bucket, so continuing sessions FOLLOW the new % on a redeploy /
    config change instead of freezing on the old model -- e.g. flipping a model
    0 -> 50% moves ~50% of in-flight sessions, not 0%.

    Deliberately no Redis profile-name pin (``pipeline_router`` had one; it froze
    sessions across weight changes, defeating the refresh-on-change contract).
    Kept ``async`` so the call seams are unchanged.
    """
    return deterministic_profile(session_id, pipeline or runtime.get_pipeline())


def _profile_for(pipeline: PipelineConfig, name: str):
    """The named profile, fail-safe to ``managed`` then the first profile —
    so a stale/absent name never raises here."""
    return pipeline.by_name(name) or pipeline.by_name("managed") or pipeline.profiles[0]


async def resolve_chain(
    session_id: str,
    step: Step,
    pipeline: Optional[PipelineConfig] = None,
    *,
    profile_name: Optional[str] = None,
) -> list:
    """Resolve an ordered chain without constructing any provider clients.

    Resolves the session's sticky weighted profile, looks up the step's tiers
    (profile override, else ``defaults``), and returns inert execution targets in
    primary-first order. Provider clients are built only when a target is reached.

    (C) When ``profile_name`` is supplied, that profile is selected DIRECTLY (via
    ``_profile_for``, fail-safe to managed) and the session is NOT re-bucketed. This
    is the correctness fix for long session ids: the router resolves the profile
    NAME from the FULL ``session_id``, but the fallback walkers are handed a
    200-char-capped ``session_id`` — re-bucketing on the capped id could pick a
    different profile than the primary path. Honoring the resolved name keeps the
    fallback chain on the same profile the router chose. When ``profile_name`` is
    None the sticky weighted split is resolved from ``session_id`` as before."""
    pipeline = pipeline or runtime.get_pipeline()
    if profile_name is not None:
        name = profile_name
    else:
        name = await resolve_profile(session_id, pipeline)
    profile = _profile_for(pipeline, name)

    # tracing-only (no behaviour change): the weighted profile this turn resolved.
    from app.llm_core import trace as _trace
    _trace.record_profile(profile.name, profile.weight)

    step_cfg = pipeline.step_config(profile, step)
    if step_cfg is None:
        raise ValueError(f"no config for step={step.value} in profile={profile.name}")

    # Health-prune inert tiers before creating execution targets.
    # Drop tiers whose endpoint is currently `open` (per-endpoint breaker). No-op
    # unless a HEALTH_* flag is on; contract: never empties the chain. Runs on the
    # inert Tiers so a pruned tier's client is never even built.
    from app.llm_core import health
    tiers = health.prune_unhealthy(step, list(step_cfg.tiers))

    # Then reorder by load before the classify/fallback walk. A saturated but
    # healthy vLLM tier can move behind
    # the managed tier, reading the gauge from the step's explicit ConcurrencyGate.
    # A no-op unless CONCURRENCY_GAUGE_ENABLED and a gate is configured on the step;
    # never drops a tier / empties the chain. Because health has already pruned any
    # DOWN tier, a down tier is gone here and can never be reordered back to front.
    from app.llm_core import concurrency
    tiers = await concurrency.reprioritize_by_load(
        step, tiers, step_cfg.triggers.concurrency_gate
    )

    from app.llm_core.execution import ExecutionTarget

    chain = [
        ExecutionTarget(tier, tier_client_kind(STEP_CLIENT_KIND[step], tier))
        for tier in tiers
    ]
    # tracing-only: the resolved primary tier + full chain for this step.
    _trace.record_step_chain(step, chain)
    return chain
