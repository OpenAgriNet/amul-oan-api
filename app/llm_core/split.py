"""Profile-name based chain resolver for llm_core.

``/api/chat/`` now routes with an explicit profile name (currently ``"oss"``).
This module materializes the ordered tier chain for that profile and step, then
applies health pruning and concurrency re-prioritization before fallback walkers
consume it.
"""

from __future__ import annotations

from typing import Optional

from helpers.utils import get_logger
from app.llm_core import runtime
from app.llm_core.config_model import PipelineConfig, Step
from app.llm_core.factory import MaterializedTier, materialize
from app.llm_core.resolver import STEP_CLIENT_KIND

logger = get_logger(__name__)


def _profile_for(pipeline: PipelineConfig, name: str):
    """The named profile, fail-safe to ``managed`` then the first profile —
    matching resolver's fail-safe so a stale/absent name never raises here."""
    return pipeline.by_name(name) or pipeline.by_name("managed") or pipeline.profiles[0]


async def resolve_chain(
    session_id: str,
    step: Step,
    pipeline: Optional[PipelineConfig] = None,
    *,
    profile_name: str = "managed",
) -> list[MaterializedTier]:
    """Materialize an ordered tier chain for the selected profile and step.

    ``session_id`` is preserved in the seam for call-site compatibility and
    logging parity; chain selection now depends only on ``profile_name``.
    """
    pipeline = pipeline or runtime.get_pipeline()
    profile = _profile_for(pipeline, profile_name)

    # tracing-only (no behaviour change): the weighted profile this turn resolved.
    from app.llm_core import trace as _trace
    _trace.record_profile(profile.name, profile.weight)

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

    chain = materialize(STEP_CLIENT_KIND[step], tiers)
    # tracing-only: the resolved primary tier + full chain for this step.
    _trace.record_step_chain(step, chain)
    return chain
