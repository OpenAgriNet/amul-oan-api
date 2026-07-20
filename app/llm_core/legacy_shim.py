"""Identity shim: synthesize a :class:`PipelineConfig` from TODAY's env vars.

``synthesize_from_env()`` reads the env exactly as the current wiring reads it —
``agents/models`` (managed + OSS agent models), ``translation.py``
(pre/post-translation), ``pipeline_router`` (the OSS %-split) and the
``FALLBACK_*`` timeouts — and emits an equivalent config so that, with
``LLM_CORE_ENABLED`` on, the resolver reproduces the legacy provider / base_url /
model / timeout for the current environment. No legacy env reading is removed;
this is a parallel, additive reader.

Profiles: ``[oss(weight=OSS_PIPELINE_PCT), managed(100-pct)]`` when OSS is
configured (``OSS_INFERENCE_ENDPOINT_URL`` set), else ``[managed(100)]`` — matching
``pipeline_router`` / ``oss_model_available()``. For the OSS profile each LLM step
carries ``[oss, managed]`` tiers (mirroring ``fallback.attempt_chain``); managed
carries ``[managed]``. Post-translation (TranslateGemma) is profile-invariant and
lives in ``defaults``.

Kept free of ``agents.*`` / ``app.services.*`` imports — reads os.getenv only —
so the core stays import-clean.
"""

from __future__ import annotations

import os

from app.llm_core.config_model import (
    ApiStyle,
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
    Triggers,
)


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── provider → (api_key_env, endpoint) for the MANAGED agent tier ─────────────
def _managed_agent_tier(timeout_ms: int, label: str) -> Tier:
    provider = (_env("LLM_PROVIDER", "openai") or "openai").lower()
    model = _env("LLM_MODEL_NAME", "gpt-4.1") or "gpt-4.1"

    if provider == "vllm":
        return Tier(provider=Provider.VLLM, model=model, endpoint=_env("INFERENCE_ENDPOINT_URL"),
                    api_key_env="INFERENCE_API_KEY", timeout_ms=timeout_ms, label=label)
    if provider == "anthropic":
        return Tier(provider=Provider.ANTHROPIC, model=model,
                    api_key_env="ANTHROPIC_API_KEY", timeout_ms=timeout_ms, label=label)
    if provider == "gemini":
        return Tier(provider=Provider.GEMINI, model=model,
                    api_key_env="GEMINI_API_KEY", timeout_ms=timeout_ms, label=label)
    if provider == "azure-openai":
        return Tier(provider=Provider.AZURE, model=_env("AZURE_OPENAI_DEPLOYMENT_NAME", model) or model,
                    endpoint=_env("AZURE_OPENAI_ENDPOINT"), api_key_env="AZURE_OPENAI_API_KEY",
                    api_version=_env("AZURE_OPENAI_API_VERSION"), timeout_ms=timeout_ms, label=label)
    # default: openai
    return Tier(provider=Provider.OPENAI, model=model, endpoint=None,
                api_key_env="OPENAI_API_KEY", timeout_ms=timeout_ms, label=label)


def _oss_agent_tier(timeout_ms: int, label: str) -> Tier:
    return Tier(
        provider=Provider.VLLM,
        model=_env("OSS_LLM_MODEL_NAME", "gemma-4-31b-it") or "gemma-4-31b-it",
        endpoint=_env("OSS_INFERENCE_ENDPOINT_URL"),
        api_key_env="OSS_INFERENCE_API_KEY",
        timeout_ms=timeout_ms,
        label=label,
    )


# ── pre-translation tiers (translation.py) ────────────────────────────────────
def _pretranslation_model_default(provider: str) -> str:
    if provider == "anthropic":
        return _env("ANTHROPIC_PRETRANSLATION_MODEL", "claude-haiku-4-5") or "claude-haiku-4-5"
    if provider == "vllm":
        return _env("LLM_MODEL_NAME", "gemma-4-31b-it") or "gemma-4-31b-it"
    return "gpt-4.1-mini"


def _managed_pretranslation_tier(timeout_ms: int) -> Tier:
    llm_provider = (_env("LLM_PROVIDER", "openai") or "openai").lower()
    provider = (_env("PRETRANSLATION_PROVIDER", llm_provider) or llm_provider).lower()
    model = _env("PRETRANSLATION_MODEL", _pretranslation_model_default(provider)) or _pretranslation_model_default(provider)

    if provider == "vllm":
        return Tier(provider=Provider.VLLM, model=model, endpoint=_env("INFERENCE_ENDPOINT_URL"),
                    api_key_env="INFERENCE_API_KEY", timeout_ms=timeout_ms, label="managed-pretranslation")
    if provider == "anthropic":
        return Tier(provider=Provider.ANTHROPIC, model=model,
                    api_key_env="ANTHROPIC_API_KEY", timeout_ms=timeout_ms, label="managed-pretranslation")
    return Tier(provider=Provider.OPENAI, model=model, endpoint=None,
                api_key_env="OPENAI_API_KEY", timeout_ms=timeout_ms, label="managed-pretranslation")


def _oss_pretranslation_tier(timeout_ms: int) -> Tier:
    return Tier(
        provider=Provider.VLLM,
        model=_env("OSS_PRETRANSLATION_MODEL", _env("OSS_LLM_MODEL_NAME", "gemma-4-31b-it")) or "gemma-4-31b-it",
        endpoint=_env("OSS_INFERENCE_ENDPOINT_URL"),
        api_key_env="OSS_INFERENCE_API_KEY",
        timeout_ms=timeout_ms,
        label="oss-pretranslation",
    )


# ── post-translation (TranslateGemma) — profile-invariant → defaults ──────────
def _post_translation_tiers() -> list[Tier]:
    raw = (_env("TRANSLATEGEMMA_27B_BASE_ENDPOINTS", "") or "").strip()
    if raw:
        endpoints = [e.strip() for e in raw.split(",") if e.strip()]
    else:
        endpoints = [_env("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1") or "http://localhost:18002/v1"]
    model_id = _env("TRANSLATEGEMMA_27B_BASE_MODEL", "translategemma-27b-base") or "translategemma-27b-base"
    return [
        Tier(provider=Provider.TRANSLATEGEMMA, model=model_id, endpoint=ep,
             api_style=ApiStyle.TEXT_COMPLETION, timeout_ms=60000, label="translategemma-27b-base")
        for ep in endpoints
    ]


def _oss_configured() -> bool:
    return bool(_env("OSS_INFERENCE_ENDPOINT_URL"))


def synthesize_from_env() -> PipelineConfig:
    """Build a behaviour-identical PipelineConfig from the current environment."""
    managed_ms = _int_env("FALLBACK_MANAGED_TIMEOUT_MS", 20000)
    oss_chat_ms = _int_env("FALLBACK_CHAT_OSS_TIMEOUT_MS", 8000)
    oss_mod_ms = _int_env("FALLBACK_MODERATION_OSS_TIMEOUT_MS", 5000)
    oss_pre_ms = _int_env("FALLBACK_PRETRANSLATION_OSS_TIMEOUT_MS", 10000)
    oss_sug_ms = _int_env("FALLBACK_SUGGESTIONS_OSS_TIMEOUT_MS", 6000)

    fallback_enabled = (os.getenv("FALLBACK_ENABLED", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    sticky_ttl = _int_env("OSS_VARIANT_TTL", 60 * 60 * 24 * 7)

    # Managed tiers per step (single-tier managed profile).
    managed_agent = _managed_agent_tier(managed_ms, "managed-agent")
    managed_pre = _managed_pretranslation_tier(managed_ms)

    def managed_steps() -> dict:
        agent_cfg = StepConfig(tiers=[managed_agent], triggers=Triggers(ttft_deadline_ms=managed_ms))
        return {
            Step.AGENT: agent_cfg,
            Step.MODERATION: StepConfig(tiers=[managed_agent]),
            Step.SUGGESTIONS: StepConfig(tiers=[managed_agent]),
            Step.PRE_TRANSLATION: StepConfig(tiers=[managed_pre]),
        }

    post_tiers = _post_translation_tiers()
    defaults = {Step.POST_TRANSLATION: StepConfig(tiers=post_tiers)}

    if not _oss_configured():
        managed = NamedProfile(name="managed", weight=100, steps=managed_steps())
        return PipelineConfig(
            profiles=[managed],
            defaults=defaults,
            sticky_ttl_s=sticky_ttl,
            fallback_enabled=fallback_enabled,
        )

    # OSS configured: two profiles. OSS profile carries [oss, managed] per step
    # (mirrors fallback.attempt_chain); managed carries [managed].
    pct = max(0, min(100, _int_env("OSS_PIPELINE_PCT", 0)))
    oss_steps = {
        Step.AGENT: StepConfig(
            tiers=[_oss_agent_tier(oss_chat_ms, "oss-agent"), managed_agent],
            triggers=Triggers(ttft_deadline_ms=oss_chat_ms),
        ),
        Step.MODERATION: StepConfig(tiers=[_oss_agent_tier(oss_mod_ms, "oss-moderation"), managed_agent]),
        Step.SUGGESTIONS: StepConfig(tiers=[_oss_agent_tier(oss_sug_ms, "oss-suggestions"), managed_agent]),
        Step.PRE_TRANSLATION: StepConfig(tiers=[_oss_pretranslation_tier(oss_pre_ms), managed_pre]),
    }
    oss_profile = NamedProfile(name="oss", weight=pct, steps=oss_steps)
    managed_profile = NamedProfile(name="managed", weight=100 - pct, steps=managed_steps())
    return PipelineConfig(
        profiles=[oss_profile, managed_profile],
        defaults=defaults,
        sticky_ttl_s=sticky_ttl,
        fallback_enabled=fallback_enabled,
    )
