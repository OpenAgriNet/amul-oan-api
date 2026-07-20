"""P1: weighted named-profile split + config-driven attempt chain.

This is the generalization of two hardwired pieces:

* ``pipeline_router._deterministic_variant`` — a *single* OSS/legacy bit from
  ``int(sha256(session_id)[:8], 16) % 100 < OSS_PIPELINE_PCT`` — becomes an
  assignment into one of *N* weighted :class:`NamedProfile` s via **cumulative
  weight buckets over the SAME hash**. Bit-compatible by construction: the shim's
  ``[oss(pct), managed(100-pct)]`` config puts the ``oss`` profile in buckets
  ``[0, pct)`` and ``managed`` in ``[pct, 100)`` — exactly today's
  ``bucket < pct -> oss`` boundary.

* ``fallback.attempt_chain``'s hardwired ``[oss, managed]`` — becomes the
  resolved profile's ``StepConfig.tiers`` materialized through the P0 factory
  (:func:`app.llm_core.factory.materialize`), preserving order (primary first).

Stickiness mirrors ``pipeline_router`` **exactly**: deterministic hash bucket,
Redis-sticky under a new key prefix (``pipeline_profile:``) storing the profile
NAME, shared across instances, fail-safe to the deterministic bucket on any Redis
error. TTL comes from ``PipelineConfig.sticky_ttl_s`` (= ``OSS_VARIANT_TTL`` via
the shim). A config *weight* change never re-buckets an already-sticky session:
the stored name is honored as long as that profile still exists.

Gated by ``PROFILES_ENABLED`` at the call seams (router split + the fallback
walkers); nothing here reads that flag — it is inert until a caller invokes it.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from app.core.cache import cache
from helpers.utils import get_logger
from app.llm_core import runtime
from app.llm_core.config_model import PipelineConfig, Step
from app.llm_core.factory import MaterializedTier, materialize
from app.llm_core.resolver import STEP_CLIENT_KIND

logger = get_logger(__name__)

# New sticky namespace (distinct from pipeline_router's ``pipeline_variant:``).
_PROFILE_KEY_PREFIX = "pipeline_profile:"


def _profile_name_for_variant(variant: str, pipeline: PipelineConfig) -> str:
    """Map a legacy variant string (``oss``/``legacy``) to a configured profile NAME.

    Mirrors ``resolver._profile_name_for_variant``: the OSS variant selects the
    OSS-primary profile (``oss``, else the first profile); anything else selects
    the managed profile (``managed``, else the last profile). Used both to honor an
    already-resolved router variant (skipping a divergent re-bucket, fix C) and to
    migrate a legacy ``pipeline_router`` sticky key (fix A)."""
    if variant == "oss":
        p = pipeline.by_name("oss")
        return p.name if p else pipeline.profiles[0].name
    p = pipeline.by_name("managed")
    return p.name if p else pipeline.profiles[-1].name


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
    """Sticky weighted-profile assignment for a session (profile NAME).

    Mirrors ``pipeline_router.resolve_pipeline_variant`` key-for-key:

    * no session id      -> deterministic bucket (no Redis);
    * Redis hit (valid)  -> the stored profile name (sticky; a later weight change
                            does not move the session);
    * Redis miss         -> deterministic bucket, persisted best-effort;
    * any Redis error    -> deterministic bucket, never raised into the caller.

    A stored name that is no longer a configured profile is ignored (re-bucketed),
    guarding against a profile being renamed/removed out from under a sticky key.
    """
    pipeline = pipeline or runtime.get_pipeline()

    if not session_id:
        return deterministic_profile(session_id, pipeline)

    key = f"{_PROFILE_KEY_PREFIX}{session_id}"
    valid = {p.name for p in pipeline.profiles}

    try:
        stored = await cache.get(key)
        if stored in valid:
            return stored
    except Exception as e:  # Redis down / timeout -> deterministic fallback
        logger.warning("llm_core.split: cache read failed for %s: %s", session_id, e)
        return deterministic_profile(session_id, pipeline)

    name = deterministic_profile(session_id, pipeline)

    try:
        await cache.set(key, name, ttl=pipeline.sticky_ttl_s)
    except Exception as e:  # persistence is best-effort; assignment still stable
        logger.warning("llm_core.split: cache write failed for %s: %s", session_id, e)

    return name


def _profile_for(pipeline: PipelineConfig, name: str):
    """The named profile, fail-safe to ``managed`` then the first profile —
    matching resolver's fail-safe so a stale/absent name never raises here."""
    return pipeline.by_name(name) or pipeline.by_name("managed") or pipeline.profiles[0]


async def resolve_chain(
    session_id: str,
    step: Step,
    pipeline: Optional[PipelineConfig] = None,
    *,
    variant: Optional[str] = None,
) -> list[MaterializedTier]:
    """The P1 seam: (session, step) -> ordered materialized tier chain.

    Resolves the session's sticky weighted profile, looks up the step's tiers
    (profile override, else ``defaults``), and materializes them via the P0
    factory (primary first, never empty). This is the config-driven successor to
    ``fallback.attempt_chain(variant, pipeline)``; ``MaterializedTier`` satisfies
    the ``Attempt`` interface the fallback walkers read (``.kind`` / ``.model`` /
    ``.model_name`` / ``.provider`` / ``.endpoint`` / ``.timeout``).

    (C) When ``variant`` is supplied, the profile is selected DIRECTLY from that
    already-resolved router variant (``oss``/``legacy``) and the session is NOT
    re-bucketed. This is the correctness fix for long session ids: the router
    resolves the variant from the FULL ``session_id``, but the fallback walkers are
    handed a 200-char-capped ``session_id`` — re-bucketing on the capped id could
    pick a different profile than the primary path. Honoring the resolved variant
    keeps the fallback chain on the same profile the router chose. When ``variant``
    is None the sticky weighted split is resolved as before."""
    pipeline = pipeline or runtime.get_pipeline()
    if variant is not None:
        name = _profile_name_for_variant(variant, pipeline)
    else:
        name = await resolve_profile(session_id, pipeline)
    profile = _profile_for(pipeline, name)

    step_cfg = pipeline.step_config(profile, step)
    if step_cfg is None:
        raise ValueError(f"no config for step={step.value} in profile={profile.name}")

    # ── P2 pre-flight FILTER: health prune (before materialize) ──────────────
    # Drop tiers whose endpoint is currently `open` (per-endpoint breaker). No-op
    # unless a HEALTH_* flag is on; contract: never empties the chain. Runs on the
    # inert Tiers so a pruned tier's client is never even built.
    from app.llm_core import health
    tiers = health.prune_unhealthy(step, list(step_cfg.tiers))

    # ── P3 pre-flight FILTER: concurrency-gauge REORDER (after prune, before
    # materialize; fixed order health-prune -> concurrency-reorder -> materialize
    # -> classify-walk). It only DEPRIORITIZES a saturated-but-UP vLLM tier behind
    # the managed tier, reading the gauge from the step's explicit ConcurrencyGate.
    # A no-op unless CONCURRENCY_GAUGE_ENABLED and a gate is configured on the step;
    # never drops a tier / empties the chain. Because health has already pruned any
    # DOWN tier, a down tier is gone here and can never be reordered back to front.
    from app.llm_core import concurrency
    tiers = await concurrency.reprioritize_by_load(
        step, tiers, step_cfg.triggers.concurrency_gate
    )

    return materialize(STEP_CLIENT_KIND[step], tiers)


def variant_for_profile(name: str) -> str:
    """Back-compat bridge: map a profile name to the legacy variant string the
    downstream chat/voice code still branches on (``is_oss`` / prompt selection /
    token caps). The shim names the OSS profile ``oss`` and the closed-source
    profile ``managed``; every non-``oss`` profile reads as ``legacy``. P4 removes
    this bridge once downstream consumes the resolved profile/config directly."""
    return "oss" if name == "oss" else "legacy"


async def resolve_variant(
    session_id: str, pipeline: Optional[PipelineConfig] = None
) -> str:
    """Weighted-split analog of ``pipeline_router.resolve_pipeline_variant``.

    Same sticky assignment as :func:`resolve_profile`, mapped back to the legacy
    ``"oss"``/``"legacy"`` variant string so the existing downstream code path is
    unchanged. With the shim's seeded 2-profile config this is distribution-
    identical to ``resolve_pipeline_variant`` (same bit-compatible bucket)."""
    return variant_for_profile(await resolve_profile(session_id, pipeline))
