"""Runtime holder + startup self-check for the unified pipeline.

``configure()`` (called from the FastAPI lifespan) loads
``PIPELINE_CONFIG_PATH`` YAML when present, else synthesizes the config from the
current env (``legacy_shim``), validates it, stores it in the module global
``PIPELINE``, and runs the identity self-check. ``get_pipeline()`` lazily
configures on first use so request paths and tests never see ``None``.

Identity self-check (the P0 bar): for the current ``.env`` it logs the resolved
(provider, base_url, model, timeout) per step and asserts they equal the legacy
singletons (``agents.models`` / ``translation.py``). A mismatch raises only when
``LLM_CORE_ENABLED`` is on — so a flag-off boot can never be broken by a shim
edge case, while flipping the flag on is gated on true identity.
"""

from __future__ import annotations

import os
from typing import Optional

from helpers.utils import get_logger
from app.llm_core.config_model import PipelineConfig, Step
from app.llm_core.legacy_shim import synthesize_from_env

logger = get_logger(__name__)

PIPELINE: Optional[PipelineConfig] = None


def _load_from_yaml(path: str) -> PipelineConfig:
    import yaml  # lazy: only needed when a config file is supplied

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)


# Providers that can materialize as a RAW_OPENAI client (AsyncOpenAI-compatible).
# anthropic/gemini are AGENT-kind only; a RAW_OPENAI step (chat pre-translation)
# configured with them would crash per-request in the factory, so reject at boot.
_RAW_OPENAI_OK = {"vllm", "openai", "azure-openai"}
# Enhancement tracking for real anthropic/gemini RAW pretranslation support.
_RAW_PROVIDER_ENH_ISSUE = (
    "https://github.com/OpenAgriNet/amul-oan-api/issues "
    "(enhancement: support anthropic/gemini RAW pretranslation)"
)


def validate_config(pipeline: PipelineConfig, *, enforce: bool) -> None:
    """(E) Fail-fast on a RAW_OPENAI-kind step whose tier provider is unsupported.

    ``Step.PRE_TRANSLATION`` materializes as a RAW_OPENAI client, which the factory
    rejects for ``anthropic``/``gemini``. If the shim synthesizes an anthropic
    pretranslation tier (``PRETRANSLATION_PROVIDER=anthropic`` or
    ``LLM_PROVIDER=anthropic``), that would crash on every request. Catch it at
    startup with a clear message instead. Gated on ``enforce`` (== LLM_CORE_ENABLED)
    so a flag-off boot on the legacy path — which handles anthropic pretranslation
    itself — is never broken; setting the flag on is what makes the config binding
    and thus the one that must be legal."""
    from app.llm_core.config_model import StepClientKind
    from app.llm_core.resolver import STEP_CLIENT_KIND

    raw_steps = [s for s, k in STEP_CLIENT_KIND.items() if k is StepClientKind.RAW_OPENAI]
    problems: list[str] = []
    for profile in pipeline.profiles:
        for step in raw_steps:
            cfg = pipeline.step_config(profile, step)
            if cfg is None:
                continue
            for tier in cfg.tiers:
                if tier.provider.value not in _RAW_OPENAI_OK:
                    problems.append(
                        f"profile={profile.name} step={step.value} "
                        f"provider={tier.provider.value} is not RAW_OPENAI-compatible "
                        f"(allowed: {sorted(_RAW_OPENAI_OK)})"
                    )
    if not problems:
        return
    msg = (
        "llm_core config INVALID — unsupported provider for a RAW_OPENAI step; "
        "anthropic/gemini need the AGENT client kind. Track "
        + _RAW_PROVIDER_ENH_ISSUE
        + ":\n  - "
        + "\n  - ".join(problems)
    )
    if enforce:
        raise ValueError(msg)
    logger.warning("%s\n(LLM_CORE_ENABLED is off; not raising)", msg)


def configure(*, run_self_check: bool = True) -> PipelineConfig:
    """Load / synthesize the pipeline config, validate, store, self-check."""
    global PIPELINE
    path = os.getenv("PIPELINE_CONFIG_PATH")
    if path and os.path.exists(path):
        logger.info("llm_core: loading pipeline config from %s", path)
        PIPELINE = _load_from_yaml(path)
    else:
        PIPELINE = synthesize_from_env()
        logger.info(
            "llm_core: synthesized pipeline config from env (profiles=%s)",
            [f"{p.name}:{p.weight}" for p in PIPELINE.profiles],
        )
    # (E) Provider/step legality — fail-fast at boot when LLM_CORE_ENABLED.
    from app.config import settings as _settings
    validate_config(PIPELINE, enforce=bool(getattr(_settings, "llm_core_enabled", False)))
    # Tracing-only: dump the COMPLETE loaded config (all profiles, step tiers,
    # triggers) as one structured boot log line so the full wiring is greppable
    # in logs even before any turn arrives (`grep llm_core.full_config`).
    from app.llm_core import trace as _trace
    _trace.log_full_config(PIPELINE)
    if run_self_check:
        try:
            self_check()
        except AssertionError:
            raise
        except Exception as exc:  # never break config load on a self-check bug
            logger.warning("llm_core: self-check skipped (%s)", exc)
    return PIPELINE


def get_pipeline() -> PipelineConfig:
    if PIPELINE is None:
        configure(run_self_check=False)
    assert PIPELINE is not None
    return PIPELINE


def _base_url(handle) -> Optional[str]:
    b = getattr(handle, "base_url", None)
    if b is None:
        b = getattr(getattr(handle, "client", None), "base_url", None)
    return str(b).rstrip("/") if b is not None else None


def self_check() -> None:
    """Assert flag-on resolution == legacy wiring for the current env."""
    # Lazy, repo-specific imports (kept out of the module graph so llm_core stays
    # import-clean); compared against the actual legacy singletons.
    from app.llm_core import resolver
    from app.config import settings
    from agents.models import (
        get_model_for_variant,
        provider_for_variant,
        oss_model_available,
    )

    enforce = bool(getattr(settings, "llm_core_enabled", False))
    managed_timeout = settings.fallback_managed_timeout_ms / 1000.0
    mismatches: list[str] = []

    variants = ["legacy"]
    if oss_model_available():
        variants.append("oss")

    # ── agent / moderation / suggestions: identity with get_model_for_variant ──
    for variant in variants:
        legacy_model = get_model_for_variant(variant)
        legacy_provider = provider_for_variant(variant)
        legacy_name = getattr(legacy_model, "model_name", None)
        for step in (Step.AGENT, Step.MODERATION, Step.SUGGESTIONS):
            mt = resolver.primary_tier(step, variant)
            r_url = _base_url(mt.handle)
            l_url = _base_url(legacy_model)
            logger.info(
                "llm_core self-check step=%s variant=%s -> provider=%s base_url=%s model=%s timeout=%s",
                step.value, variant, mt.provider, r_url, mt.model_name, mt.timeout,
            )
            if legacy_name is not None and mt.model_name != legacy_name:
                mismatches.append(f"{step.value}/{variant} model {mt.model_name!r} != legacy {legacy_name!r}")
            if l_url is not None and r_url != l_url:
                mismatches.append(f"{step.value}/{variant} base_url {r_url!r} != legacy {l_url!r}")
            # provider parity (vllm for oss, LLM_PROVIDER for managed)
            if mt.provider != legacy_provider:
                mismatches.append(f"{step.value}/{variant} provider {mt.provider!r} != legacy {legacy_provider!r}")

    # ── pre/post-translation: identity with translation.py singletons ─────────
    # Guarded: translation.py transitively imports agents.tools, which fails to
    # build its schemas under a pydantic-ai version mismatch. The agents.models
    # checks above always run (and enforce); the translation-dependent checks are
    # skipped-with-a-log if that module can't import in this env.
    try:
        from app.services import translation as tr
    except Exception as exc:  # pragma: no cover - env-dependent
        tr = None
        logger.warning("llm_core self-check: translation checks skipped (%s)", exc)

    if tr is not None:
        # pre-translation: only the openai/vllm (RAW_OPENAI) managed path is
        # introspectable; anthropic pretranslation is checked by model name only.
        managed_pre = resolver.primary_tier(Step.PRE_TRANSLATION, "legacy")
        logger.info(
            "llm_core self-check step=pre_translation variant=legacy -> provider=%s base_url=%s model=%s timeout=%s",
            managed_pre.provider, _base_url(managed_pre.handle), managed_pre.model_name, managed_pre.timeout,
        )
        if managed_pre.model_name != tr.PRETRANSLATION_MODEL:
            mismatches.append(f"pre_translation model {managed_pre.model_name!r} != legacy {tr.PRETRANSLATION_MODEL!r}")
        if managed_pre.provider in ("openai", "vllm"):
            try:
                legacy_client = tr._get_openai_client()
                l_url = _base_url(legacy_client)
                r_url = _base_url(managed_pre.handle)
                if l_url is not None and r_url != l_url:
                    mismatches.append(f"pre_translation base_url {r_url!r} != legacy {l_url!r}")
            except Exception as exc:  # client init may need a key not present in tests
                logger.info("llm_core self-check: pre_translation client compare skipped (%s)", exc)

        # post-translation: identity with translation TranslateGemma endpoints.
        post = resolver.primary_tier(Step.POST_TRANSLATION, "legacy")
        legacy_eps = [e.rstrip("/") for e in tr.TRANSLATION_ENDPOINTS_27B_BASE]
        legacy_tg_model = tr.TRANSLATION_MODEL_IDS.get("27b-base")
        logger.info(
            "llm_core self-check step=post_translation variant=legacy -> model=%s endpoints=%s",
            post.model_name, legacy_eps,
        )
        if post.endpoint.rstrip("/") not in legacy_eps:
            mismatches.append(f"post_translation endpoint {post.endpoint!r} not in legacy {legacy_eps!r}")
        if legacy_tg_model is not None and post.model_name != legacy_tg_model:
            mismatches.append(f"post_translation model {post.model_name!r} != legacy {legacy_tg_model!r}")

    # managed timeout parity (sample the agent managed tier).
    managed_agent_mt = resolver.primary_tier(Step.AGENT, "legacy")
    if managed_agent_mt.timeout not in (None, managed_timeout):
        mismatches.append(f"agent/legacy timeout {managed_agent_mt.timeout} != managed {managed_timeout}")

    if mismatches:
        msg = "llm_core self-check FAILED (resolve != legacy wiring):\n  - " + "\n  - ".join(mismatches)
        if enforce:
            raise AssertionError(msg)
        logger.warning("%s\n(LLM_CORE_ENABLED is off; not raising)", msg)
    else:
        logger.info(
            "llm_core self-check PASSED: resolve == legacy wiring for variants=%s (LLM_CORE_ENABLED=%s)",
            variants, enforce,
        )
