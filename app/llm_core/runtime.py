"""Runtime holder for the unified pipeline.

``configure()`` (called from the FastAPI lifespan) loads
``PIPELINE_CONFIG_PATH`` YAML when present, else synthesizes the config from the
current env (``legacy_shim``), validates it, then atomically stores it in the
module global ``PIPELINE``. ``get_pipeline()`` lazily
configures on first use so request paths and tests never see ``None``.

Provider clients remain lazy and are constructed only when an execution target is
actually invoked. Agents carry no construction-time model; this module is the only
runtime model selection path.
"""

from __future__ import annotations

import os
from typing import Optional

from helpers.utils import get_logger
from app.config import get_config_value
from app.llm_core.config_model import PipelineConfig, Step
from app.llm_core.legacy_shim import synthesize_from_env

logger = get_logger(__name__)

PIPELINE: Optional[PipelineConfig] = None
# The config captured at ``configure()`` BEFORE any live (redis) refresh can
# override it — the deploy's permanent boot fallback. ``config_source`` reverts to
# THIS (not the last live config) when the live key is cleared/absent, so `clear`
# is a true emergency rollback to the boot config.
BOOT_PIPELINE: Optional[PipelineConfig] = None


def _load_from_yaml(path: str) -> PipelineConfig:
    import yaml  # lazy: only needed when a config file is supplied

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)


# Providers with a concrete pretranslation protocol adapter.
_AGENT_OK = {"vllm", "openai", "azure-openai", "anthropic", "gemini"}
_PRETRANSLATION_OK = {"vllm", "openai", "azure-openai", "anthropic"}
_POST_TRANSLATION_OK = {"vllm", "openai", "azure-openai", "translategemma"}


def validate_config(pipeline: PipelineConfig) -> None:
    """Structurally validate the normalized tiers that can execute."""
    problems: list[str] = []
    for profile in pipeline.profiles:
        for step in Step:
            plan = pipeline.step_plan(profile, step)
            if plan is None:
                continue
            allowed = (
                _PRETRANSLATION_OK
                if step is Step.PRE_TRANSLATION
                else _POST_TRANSLATION_OK
                if step is Step.POST_TRANSLATION
                else _AGENT_OK
            )
            adapter = {
                Step.PRE_TRANSLATION: "pretranslation",
                Step.POST_TRANSLATION: "post-translation",
            }.get(step, step.value)
            for tier in plan.candidates:
                provider = tier.provider.value
                if provider not in allowed:
                    problems.append(
                        f"profile={profile.name} step={step.value} "
                        f"provider={provider} has no {adapter} adapter "
                        f"(allowed: {sorted(allowed)})"
                    )
                    continue
                if tier.provider.value == "vllm" and not tier.endpoint:
                    problems.append(
                        f"profile={profile.name} step={step.value} provider=vllm "
                        "requires endpoint"
                    )
                if tier.provider.value == "azure-openai" and (
                    not tier.endpoint or not tier.api_version or not tier.api_key_env
                ):
                    problems.append(
                        f"profile={profile.name} step={step.value} provider=azure-openai "
                        "requires endpoint, api_version and api_key_env"
                    )
                if tier.provider.value == "translategemma" and not tier.endpoint:
                    problems.append(
                        f"profile={profile.name} step={step.value} "
                        "provider=translategemma requires endpoint"
                    )
    if not problems:
        return
    msg = (
        "llm_core config INVALID — unsupported provider adapter:\n  - "
        + "\n  - ".join(problems)
    )
    raise ValueError(msg)


def validate_content(cfg: PipelineConfig) -> None:
    """Validate active config without constructing or caching provider clients."""
    validate_config(cfg)


def _truthy_env(name: str) -> bool:
    v = get_config_value(name)
    return v is not None and v.strip().lower() in {"1", "true", "yes", "on"}


class BootRefused(RuntimeError):
    """Intentional hard-gate boot failure (e.g. REQUIRE_OVERFLOW_ARMED with overflow
    DISARMED). The startup call site re-raises this instead of swallowing it."""


def _assert_boot_posture(pipeline: PipelineConfig) -> None:
    """Emit a LOUD one-line 'overflow ARMED / DISARMED' posture summary at boot.

    The whole overflow system — the OSS->managed attempt chain AND the health +
    concurrency guards, which fire ONLY via the fallback walkers — is inert unless
    ``FALLBACK_ENABLED`` is on. A deploy from defaults could therefore ship dark
    with nothing in the logs saying so. This line makes the armament state greppable
    at startup (``grep 'llm_core posture'``): INFO when armed, WARNING when disarmed.

    Honors the opt-in ``REQUIRE_OVERFLOW_ARMED``: when truthy, a DISARMED boot is a
    hard error (raises) so prod can gate on it and never ship overflow-off."""
    from app.config import settings

    def _onoff(b: bool) -> str:
        return "on" if b else "off"

    fallback_on = pipeline.fallback_enabled
    if settings.concurrency_gauge_enabled:
        conc = "on(metrics_url set)" if settings.agent_concurrency_metrics_url else "on(metrics_url unset — no-op)"
    else:
        conc = "off"
    guards = (
        f"health_breaker={_onoff(settings.health_breaker_enabled)} "
        f"health_poller={_onoff(settings.health_poller_enabled)} "
        f"concurrency={conc}"
    )
    if fallback_on:
        logger.info("llm_core posture: overflow=ARMED fallback=on %s", guards)
    else:
        logger.warning(
            "llm_core posture: overflow=DISARMED (FALLBACK_ENABLED=false) — "
            "health/concurrency guards inert (they fire only via the fallback "
            "walkers); %s", guards,
        )
        if _truthy_env("REQUIRE_OVERFLOW_ARMED"):
            raise BootRefused(
                "llm_core boot refused: REQUIRE_OVERFLOW_ARMED=true but overflow is "
                "DISARMED (FALLBACK_ENABLED=false). Set FALLBACK_ENABLED=true to arm "
                "the unified overflow/fallback path, or unset REQUIRE_OVERFLOW_ARMED."
            )


def configure() -> PipelineConfig:
    """Load, normalize, validate, and atomically publish pipeline config."""
    global PIPELINE, BOOT_PIPELINE
    try:
        path = get_config_value("PIPELINE_CONFIG_PATH")
        if path and os.path.exists(path):
            logger.info("llm_core: loading pipeline config from %s", path)
            candidate = _load_from_yaml(path)
        else:
            candidate = synthesize_from_env()
            logger.info(
                "llm_core: synthesized pipeline config from env (profiles=%s)",
                [f"{p.name}:{p.weight}" for p in candidate.profiles],
            )
        validate_content(candidate)
    except Exception as exc:
        raise BootRefused(f"llm_core boot config rejected: {exc}") from exc

    # The posture gate is validation too: run it before the single commit point so
    # any rejection leaves both known-good globals untouched.
    _assert_boot_posture(candidate)

    # Publish only after the complete candidate has passed.
    PIPELINE = candidate
    # Capture the boot config as the permanent fallback BEFORE any live redis
    # refresh can override PIPELINE (get_pipeline -> config_source.maybe_refresh).
    # config_source reverts to THIS on a cleared/absent live key (emergency
    # rollback), never to a stale last-live config.
    BOOT_PIPELINE = PIPELINE
    # Tracing-only: dump the COMPLETE loaded config (all profiles, step tiers,
    # triggers) as one structured boot log line so the full wiring is greppable
    # in logs even before any turn arrives (`grep llm_core.full_config`).
    from app.llm_core import trace as _trace
    _trace.log_full_config(PIPELINE)
    # M2: note whether the live redis-backed config source is enabled (default OFF).
    # When on, weight changes PUT to the channel key take effect within the TTL with
    # no redeploy; when off, get_pipeline() serves the boot config only.
    from app.llm_core import config_source
    if config_source.enabled():
        logger.info(
            "llm_core: live redis config source ENABLED (channel=%s key=%s refresh=%ss) "
            "— weight changes PUT to that key take effect within the TTL, no redeploy",
            config_source.channel(), config_source.key(), config_source.refresh_interval_s(),
        )
    else:
        logger.info(
            "llm_core: live redis config source disabled (%s unset) — serving boot config only",
            config_source.ENABLED_ENV,
        )
    return PIPELINE


def get_pipeline() -> PipelineConfig:
    global PIPELINE
    if PIPELINE is None:
        configure()
    assert PIPELINE is not None
    # M2 (live config): consult the redis-backed source. TTL-gated (hits redis at
    # most once per PIPELINE_CONFIG_REFRESH_S window) and fail-safe (any error ->
    # returns the last-good PIPELINE unchanged, never raises). When
    # PIPELINE_CONFIG_REDIS_ENABLED is unset/false this is an immediate identity
    # no-op, so behaviour is byte-identical to boot-config-only.
    from app.llm_core import config_source
    PIPELINE = config_source.maybe_refresh(PIPELINE)
    return PIPELINE
