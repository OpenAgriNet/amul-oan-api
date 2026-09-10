"""Identity shim: synthesize a :class:`PipelineConfig` from TODAY's env vars.

``synthesize_from_env()`` reads the env exactly as the current wiring reads it —
the former construction-time agent model wiring, ``translation.py``
(pre/post-translation), ``pipeline_router`` (the OSS %-split) and the
``FALLBACK_*`` timeouts — and emits an equivalent config for the core execution
boundary.

Profiles: ``[oss(weight=OSS_PIPELINE_PCT), managed(100-pct)]`` when OSS is
configured (``OSS_INFERENCE_ENDPOINT_URL`` set), else ``[managed(100)]`` — matching
``pipeline_router`` / ``oss_model_available()``. For the OSS profile each LLM step
carries ``[oss, managed]`` tiers (mirroring ``fallback.attempt_chain``); managed
carries ``[managed]``. Post-translation (TranslateGemma) is profile-invariant and
lives in ``defaults``.

Kept free of ``agents.*`` / ``app.services.*`` imports — reads centralized
``app.config`` values only —
so the core stays import-clean.
"""

from __future__ import annotations

import logging

from app.llm_core.config_model import (
    AdmissionPolicy,
    ConcurrencyGate,
    NamedProfile,
    PipelineConfig,
    Provider,
    ProfileCapabilities,
    Step,
    StepConfig,
    Tier,
    Triggers,
)
from app.config import get_config_value


logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    return get_config_value(name, default)


def _int_env(name: str, default: int) -> int:
    raw = get_config_value(name)
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
                    api_key_env="ANTHROPIC_API_KEY", timeout_ms=timeout_ms,
                    admission=AdmissionPolicy.MANAGED, label=label)
    if provider == "gemini":
        return Tier(provider=Provider.GEMINI, model=model,
                    api_key_env="GEMINI_API_KEY", timeout_ms=timeout_ms,
                    admission=AdmissionPolicy.MANAGED, label=label)
    if provider == "azure-openai":
        return Tier(provider=Provider.AZURE, model=_env("AZURE_OPENAI_DEPLOYMENT_NAME", model) or model,
                    endpoint=_env("AZURE_OPENAI_ENDPOINT"), api_key_env="AZURE_OPENAI_API_KEY",
                    api_version=_env("AZURE_OPENAI_API_VERSION"), timeout_ms=timeout_ms,
                    admission=AdmissionPolicy.MANAGED, label=label)
    # default: openai
    return Tier(provider=Provider.OPENAI, model=model, endpoint=None,
                api_key_env="OPENAI_API_KEY", timeout_ms=timeout_ms,
                admission=AdmissionPolicy.MANAGED, label=label)


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
                    api_key_env="ANTHROPIC_API_KEY", timeout_ms=timeout_ms,
                    admission=AdmissionPolicy.MANAGED, label="managed-pretranslation")
    return Tier(provider=Provider.OPENAI, model=model, endpoint=None,
                api_key_env="OPENAI_API_KEY", timeout_ms=timeout_ms,
                admission=AdmissionPolicy.MANAGED, label="managed-pretranslation")


def _oss_pretranslation_tier(timeout_ms: int) -> Tier:
    return Tier(
        provider=Provider.VLLM,
        model=_env("OSS_PRETRANSLATION_MODEL", _env("OSS_LLM_MODEL_NAME", "gemma-4-31b-it")) or "gemma-4-31b-it",
        endpoint=_env("OSS_INFERENCE_ENDPOINT_URL"),
        api_key_env="OSS_INFERENCE_API_KEY",
        timeout_ms=timeout_ms,
        label="oss-pretranslation",
    )


# ── post-translation — profile-invariant → defaults ──────────────────────────
# Chain: [TranslateGemma(LB), managed-LLM overflow]. TranslateGemma is deployed
# BEHIND AN NGINX LB, so the SINGULAR ``TRANSLATEGEMMA_27B_BASE_ENDPOINT`` IS that
# LB (it fans out to replicas server-side) — there is exactly one client-facing
# endpoint, never a client-side list. The overflow tier is the managed LLM doing
# en→target translation via chat.completions with the SAME glossary/rules prompt;
# it serves only when TranslateGemma fails before the first streamed token.
def _post_translation_tiers(fallback_enabled: bool) -> list[Tier]:
    # TranslateGemma is fronted by an nginx LB, so post-translation reads the
    # SINGULAR endpoint. The old client-side plural list (+random.choice) is gone;
    # warn loudly if a stale env still sets only the plural var, which would
    # otherwise be silently ignored and drop TG to the localhost default.
    if _env("TRANSLATEGEMMA_27B_BASE_ENDPOINTS") and not _env("TRANSLATEGEMMA_27B_BASE_ENDPOINT"):
        logger.warning(
            "TRANSLATEGEMMA_27B_BASE_ENDPOINTS (plural) is set but the singular "
            "TRANSLATEGEMMA_27B_BASE_ENDPOINT is not — the plural var is DEPRECATED "
            "and ignored; TranslateGemma will fall back to the localhost default. "
            "Set TRANSLATEGEMMA_27B_BASE_ENDPOINT to the nginx LB URL."
        )
    endpoint = _env("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1") or "http://localhost:18002/v1"
    model_id = _env("TRANSLATEGEMMA_27B_BASE_MODEL", "translategemma-27b-base") or "translategemma-27b-base"
    # ``timeout_ms`` is the 60s overall/total per-attempt cap; ``ttft_ms`` is the
    # distinct, SHORT first-token deadline (single-digit seconds) so a saturated-
    # but-alive TG overflows fast instead of blocking a voice turn for the full 60s.
    tg = Tier(
        provider=Provider.TRANSLATEGEMMA, model=model_id, endpoint=endpoint,
        timeout_ms=60000,
        ttft_ms=_int_env("FALLBACK_POST_TRANSLATION_TG_TTFT_MS", 5000),
        label="translategemma",
    )
    if not fallback_enabled:
        return [tg]

    # Post-translation requires an OpenAI-compatible raw client. Keep the old
    # provider when it is compatible, but let deployments select it independently
    # from the agent. For Anthropic/Gemini, omit the optional overflow unless a
    # compatible provider is explicitly configured.
    agent_provider = (_env("LLM_PROVIDER", "openai") or "openai").lower()
    configured_provider = _env("POST_TRANSLATION_LLM_PROVIDER")
    if configured_provider is None and agent_provider not in {"openai", "azure-openai", "vllm"}:
        return [tg]
    provider = (configured_provider or agent_provider).lower()
    if provider not in {"openai", "azure-openai", "vllm"}:
        raise ValueError(
            "POST_TRANSLATION_LLM_PROVIDER must be openai, azure-openai, or vllm "
            f"when fallback is enabled; got {provider!r}"
        )
    llm_ms = _int_env("FALLBACK_POST_TRANSLATION_LLM_TIMEOUT_MS", 30000)
    model_override = _env("POST_TRANSLATION_LLM_MODEL")
    if provider == "vllm":
        fallback = Tier(
            provider=Provider.VLLM,
            model=model_override or "gemma-4-31b-it",
            endpoint=_env("INFERENCE_ENDPOINT_URL"),
            api_key_env="INFERENCE_API_KEY",
            timeout_ms=llm_ms,
            label="llm-fallback",
        )
    elif provider == "azure-openai":
        fallback = Tier(
            provider=Provider.AZURE,
            model=model_override or _env("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1") or "gpt-4.1",
            endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key_env="AZURE_OPENAI_API_KEY",
            api_version=_env("AZURE_OPENAI_API_VERSION"),
            timeout_ms=llm_ms,
            admission=AdmissionPolicy.MANAGED,
            label="llm-fallback",
        )
    else:
        fallback = Tier(
            provider=Provider.OPENAI,
            model=model_override or "gpt-4.1",
            api_key_env="OPENAI_API_KEY",
            timeout_ms=llm_ms,
            admission=AdmissionPolicy.MANAGED,
            label="llm-fallback",
        )
    return [tg, fallback]


def _oss_configured() -> bool:
    return bool(_env("OSS_INFERENCE_ENDPOINT_URL"))


def _agent_concurrency_gate() -> ConcurrencyGate | None:
    """The P3 concurrency gate for the AGENT step, from an EXPLICIT metrics URL.

    ``AGENT_CONCURRENCY_METRICS_URL`` (the vLLM Prometheus ``/metrics``, e.g.
    ``http://10.185.25.197:8020/metrics``) arms the gate; ``CONCURRENCY_MAX`` (or
    10) is the in-flight threshold. Unset -> ``None`` -> the gauge is a harmless
    no-op. Never derived by stripping ``/v1`` off the inference endpoint (plan §2)."""
    metrics_url = _env("AGENT_CONCURRENCY_METRICS_URL")
    if not metrics_url:
        return None
    return ConcurrencyGate(metrics_url=metrics_url, max_concurrency=_int_env("CONCURRENCY_MAX", 10))


def _capabilities(agent_tier: Tier) -> ProfileCapabilities:
    override = _env("CHAT_HISTORY_MAX_TOKENS")
    if override and override.isdigit():
        history_max_tokens = int(override)
    elif agent_tier.provider is Provider.VLLM and "gemma" in agent_tier.model.lower():
        history_max_tokens = _int_env("CHAT_HISTORY_MAX_TOKENS_VLLM_GEMMA", 10_000)
    else:
        history_max_tokens = 80_000
    return ProfileCapabilities(
        requires_translation=agent_tier.provider is Provider.VLLM,
        history_max_tokens=history_max_tokens,
    )


def synthesize_from_env() -> PipelineConfig:
    """Build a behaviour-identical PipelineConfig from the current environment."""
    managed_ms = _int_env("FALLBACK_MANAGED_TIMEOUT_MS", 20000)
    oss_chat_ms = _int_env("FALLBACK_CHAT_OSS_TIMEOUT_MS", 8000)
    oss_mod_ms = _int_env("FALLBACK_MODERATION_OSS_TIMEOUT_MS", 5000)
    oss_pre_ms = _int_env("FALLBACK_PRETRANSLATION_OSS_TIMEOUT_MS", 10000)
    oss_sug_ms = _int_env("FALLBACK_SUGGESTIONS_OSS_TIMEOUT_MS", 6000)

    fallback_enabled = str(get_config_value("FALLBACK_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    # Managed tiers per step (single-tier managed profile).
    managed_agent = _managed_agent_tier(managed_ms, "managed-agent")
    managed_pre = _managed_pretranslation_tier(managed_ms)

    def managed_steps() -> dict:
        agent_cfg = StepConfig(tiers=[managed_agent])
        return {
            Step.AGENT: agent_cfg,
            Step.MODERATION: StepConfig(tiers=[managed_agent]),
            Step.SUGGESTIONS: StepConfig(tiers=[managed_agent]),
            Step.PRE_TRANSLATION: StepConfig(tiers=[managed_pre]),
        }

    post_tiers = _post_translation_tiers(fallback_enabled)
    defaults = {Step.POST_TRANSLATION: StepConfig(tiers=post_tiers)}

    if not _oss_configured():
        managed = NamedProfile(
            name="managed",
            weight=100,
            capabilities=_capabilities(managed_agent),
            steps=managed_steps(),
        )
        return PipelineConfig(
            profiles=[managed],
            defaults=defaults,
            fallback_enabled=fallback_enabled,
        )

    # OSS configured: two profiles. OSS profile carries [oss, managed] per step
    # (mirrors fallback.attempt_chain); managed carries [managed].
    pct = max(0, min(100, _int_env("OSS_PIPELINE_PCT", 0)))
    agent_gate = _agent_concurrency_gate()  # None unless AGENT_CONCURRENCY_METRICS_URL is set
    oss_steps = {
        Step.AGENT: StepConfig(
            tiers=[_oss_agent_tier(oss_chat_ms, "oss-agent"), managed_agent],
            triggers=Triggers(concurrency_gate=agent_gate),
        ),
        Step.MODERATION: StepConfig(tiers=[_oss_agent_tier(oss_mod_ms, "oss-moderation"), managed_agent]),
        Step.SUGGESTIONS: StepConfig(tiers=[_oss_agent_tier(oss_sug_ms, "oss-suggestions"), managed_agent]),
        Step.PRE_TRANSLATION: StepConfig(tiers=[_oss_pretranslation_tier(oss_pre_ms), managed_pre]),
    }
    oss_profile = NamedProfile(
        name="oss",
        weight=pct,
        capabilities=_capabilities(oss_steps[Step.AGENT].tiers[0]),
        steps=oss_steps,
    )
    managed_profile = NamedProfile(
        name="managed",
        weight=100 - pct,
        capabilities=_capabilities(managed_agent),
        steps=managed_steps(),
    )
    return PipelineConfig(
        profiles=[oss_profile, managed_profile],
        defaults=defaults,
        fallback_enabled=fallback_enabled,
    )
