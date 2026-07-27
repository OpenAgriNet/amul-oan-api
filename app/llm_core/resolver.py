"""Resolver — the single seam the engine consumes.

``resolve_chain(step, profile_name)`` selects the named profile DIRECTLY, looks up
the step's tiers (profile override, else defaults), and materializes them into
``list[MaterializedTier]`` (primary first, never empty).

The routing token is the actual profile NAME (the N-way split's output), selected
via ``pipeline.by_name(profile_name)`` with a fail-safe to ``managed`` then
``profiles[0]``. There is no longer a 2-way ``variant`` collapse: a 3rd profile
(e.g. ``qwen``) is served with ITS tiers, not bucketed back to oss/managed. An
absent/stale name resolves to ``managed`` — identical to the old variant mapping,
so threading ``"oss"``/``"legacy"`` remains bit-identical (``"legacy"`` -> managed).
"""

from __future__ import annotations

from typing import Any

from app.llm_core import runtime
from app.llm_core.config_model import Step, StepClientKind, Tier
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


def _profile(pipeline, profile_name: str):
    """The named profile, fail-safe to ``managed`` then ``profiles[0]``. An absent
    name (e.g. the legacy ``"legacy"`` sentinel, or a stale/capped name) resolves to
    ``managed`` — identical to the removed ``_profile_name_for_variant`` mapping."""
    return pipeline.by_name(profile_name) or pipeline.by_name("managed") or pipeline.profiles[0]


def resolve_chain(step: Step, profile_name: str = "managed") -> list[MaterializedTier]:
    """Materialized tier chain for a step under a session's resolved profile NAME.

    ``profile_name`` is the routing token (the actual configured profile name, e.g.
    ``oss``/``managed``/``qwen``); the profile is selected DIRECTLY, so a 3rd profile
    is served — not collapsed back to oss/managed via a variant string."""
    pipeline = runtime.get_pipeline()
    profile = _profile(pipeline, profile_name)

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


def post_translation_tiers(profile_name: str = "managed") -> list[Tier]:
    """The INERT POST_TRANSLATION tier chain — NOT materialized.

    Post-translation is unique: TranslateGemma serves ~every turn, and its cheap
    :class:`TGDescriptor` primary is provider-independent, whereas the managed-LLM
    overflow tier is an ``AsyncOpenAI`` client that is both expensive to build and
    can legitimately fail to construct in a healthy-TG deployment (no OPENAI key,
    incomplete Azure env, unset INFERENCE_ENDPOINT_URL, anthropic/gemini provider).
    Eagerly materializing the whole chain on EVERY translate call would waste that
    construction and let a misconfigured overflow tier take a healthy primary down.
    So the translation adapter takes the tiers INERT and builds each handle LAZILY
    only as the fallback walker reaches it (see ``translation._PostTranslationTier``).

    Deliberately does NOT ``record_profile``: post-translation runs AFTER the agent
    step and is profile-INVARIANT (it lives in ``defaults``), so recording a profile
    here would last-write-wins CLOBBER the turn's real ``pipeline_profile`` (set by
    the agent-step ``resolve_chain``) to "managed". The step chain is still recorded
    by the translation adapter's ``_post_translation_chain`` via ``record_step_chain``."""
    pipeline = runtime.get_pipeline()
    profile = _profile(pipeline, profile_name)
    step_cfg = pipeline.step_config(profile, Step.POST_TRANSLATION)
    if step_cfg is None:
        raise ValueError("no config for step=post_translation")
    return list(step_cfg.tiers)


def chain_for(step: Step, profile_name: str = "managed") -> list[MaterializedTier]:
    """Thin alias kept for the API shape P1 will generalize (weighted split)."""
    return resolve_chain(step, profile_name)


def primary_tier(step: Step, profile_name: str = "managed") -> MaterializedTier:
    """The primary (index-0) materialized tier for a session's resolved profile NAME
    — identity with today's ``get_model_for_variant`` single-model selection at a
    non-fallback call site, generalized to N profiles."""
    return resolve_chain(step, profile_name)[0]


def primary_handle(step: Step, profile_name: str = "managed") -> Any:
    """Primary tier's live handle (pydantic-ai Model / AsyncOpenAI / TGDescriptor)."""
    return primary_tier(step, profile_name).handle


def primary_provider(step: Step, profile_name: str = "managed") -> str:
    """Primary tier's provider string (identity with ``provider_for_variant``)."""
    return primary_tier(step, profile_name).provider
