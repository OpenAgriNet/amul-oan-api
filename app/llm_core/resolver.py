"""Resolver — the single seam the engine consumes.

``resolve_chain(step, variant)`` selects the profile for a session's resolved
variant, looks up the step's tiers (profile override, else defaults), and
materializes them into ``list[MaterializedTier]`` (primary first, never empty).

P0 has **no weighted split yet**: the variant is resolved upstream exactly as
today (``pipeline_router.resolve_pipeline_variant``) and passed in; here we just
map ``variant -> profile`` ("oss" -> profile ``oss``, anything else -> ``managed``,
fail-safe to managed when the OSS profile is absent). The chain shape ([oss,
managed] for OSS, [managed] otherwise) matches ``fallback.attempt_chain`` so P1
can drop the weighted split + config-driven tiers in behind this same API.
"""

from __future__ import annotations

from typing import Any

from app.llm_core import runtime
from app.llm_core.config_model import Step, StepClientKind
from app.llm_core.factory import MaterializedTier, materialize

# Which client kind each step materializes to. For every step but POST_TRANSLATION
# this is a single fixed kind for the whole chain. POST_TRANSLATION's chain is
# mixed-provider ([TranslateGemma, LLM-overflow]); the value here is the PRIMARY's
# kind (TRANSLATEGEMMA), and ``factory._tier_client_kind`` redirects a non-TG
# overflow tier under this step to RAW_OPENAI per tier at materialize time.
STEP_CLIENT_KIND: dict[Step, StepClientKind] = {
    Step.AGENT: StepClientKind.AGENT,
    Step.MODERATION: StepClientKind.AGENT,
    Step.SUGGESTIONS: StepClientKind.AGENT,
    Step.PRE_TRANSLATION: StepClientKind.RAW_OPENAI,
    Step.POST_TRANSLATION: StepClientKind.TRANSLATEGEMMA,
}


def _profile_name_for_variant(variant: str) -> str:
    return "oss" if variant == "oss" else "managed"


def resolve_chain(step: Step, variant: str = "legacy") -> list[MaterializedTier]:
    """Materialized tier chain for a step under a session's resolved variant."""
    pipeline = runtime.get_pipeline()
    name = _profile_name_for_variant(variant)
    profile = pipeline.by_name(name) or pipeline.by_name("managed") or pipeline.profiles[0]

    step_cfg = pipeline.step_config(profile, step)
    if step_cfg is None:
        raise ValueError(f"no config for step={step.value} in profile={profile.name}")

    kind = STEP_CLIENT_KIND[step]
    chain = materialize(kind, list(step_cfg.tiers))
    # tracing-only (no behaviour change): record the resolved profile + step chain
    # for the non-fallback primary-tier seam (primary_tier/primary_handle callers).
    from app.llm_core import trace as _trace
    _trace.record_profile(profile.name, profile.weight)
    _trace.record_step_chain(step, chain)
    return chain


def chain_for(step: Step, variant: str = "legacy") -> list[MaterializedTier]:
    """Thin alias kept for the API shape P1 will generalize (weighted split)."""
    return resolve_chain(step, variant)


def primary_tier(step: Step, variant: str = "legacy") -> MaterializedTier:
    """The primary (index-0) materialized tier — identity with today's
    ``get_model_for_variant`` single-model selection at a non-fallback call site."""
    return resolve_chain(step, variant)[0]


def primary_handle(step: Step, variant: str = "legacy") -> Any:
    """Primary tier's live handle (pydantic-ai Model / AsyncOpenAI / TGDescriptor)."""
    return primary_tier(step, variant).handle


def primary_provider(step: Step, variant: str = "legacy") -> str:
    """Primary tier's provider string (identity with ``provider_for_variant``)."""
    return primary_tier(step, variant).provider
