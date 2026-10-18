"""Runtime holder for the unified pipeline.

``configure()`` (called from the FastAPI lifespan) loads
``PIPELINE_CONFIG_PATH`` YAML when present, else synthesizes the config from the
current env (``legacy_shim``), validates it, then atomically stores it in the
module global ``PIPELINE``. ``get_pipeline()`` lazily
configures on first use so request paths and tests never see ``None``.

With ``PIPELINE_CONFIG_DIR`` one process serves chat and voice, each on its own
file there (``chat.yaml``, ``voice.yaml``), and ``PIPELINES`` / ``BOOT_PIPELINES``
/ ``SETTINGS`` hold one entry per surface instead.

Provider clients remain lazy and are constructed only when an execution target is
actually invoked. Agents carry no construction-time model; this module is the only
runtime model selection path.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from helpers.utils import get_logger
from app.config import get_config_value
from app.llm_core.config_model import PipelineConfig, Step, SurfaceSettings, VoiceSettings
from app.llm_core.legacy_shim import synthesize_from_env

logger = get_logger(__name__)

PIPELINE: Optional[PipelineConfig] = None
# The config captured at ``configure()`` BEFORE any live (redis) refresh can
# override it — the deploy's permanent boot fallback. ``config_source`` reverts to
# THIS (not the last live config) when the live key is cleared/absent, so `clear`
# is a true emergency rollback to the boot config.
BOOT_PIPELINE: Optional[PipelineConfig] = None

# With PIPELINE_CONFIG_DIR: each surface's live config, boot config and settings,
# by surface name. Empty without it, when PIPELINE is the deployment's one config
# and every surface runs on it.
CONFIG_DIR_ENV = "PIPELINE_CONFIG_DIR"
PIPELINES: dict[str, PipelineConfig] = {}
BOOT_PIPELINES: dict[str, PipelineConfig] = {}
SETTINGS: dict[str, SurfaceSettings] = {}
_NO_SETTINGS = {"chat": SurfaceSettings(), "voice": VoiceSettings()}


_ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _env_ref(match: re.Match) -> str:
    value = get_config_value(match.group(1))
    if not value:
        raise ValueError(f"{match.group(1)} is unset but the pipeline config file refers to it")
    return value


def _substitute_env(value):
    """Replace each ``${NAME}`` in the file's strings with that setting, so the file
    can name per-deployment endpoints instead of carrying them."""
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, str):
        return _ENV_REF.sub(_env_ref, value)
    return value


def _settings_model(surface: str) -> type[SurfaceSettings]:
    return VoiceSettings if surface == "voice" else SurfaceSettings


def _load_from_yaml(path: str, surface: str) -> tuple[PipelineConfig, SurfaceSettings]:
    """The file's PipelineConfig and its ``settings:`` block, validated as
    ``surface``'s. The two are kept apart: a live (redis) config is a PipelineConfig
    only, so a live push never changes a surface's settings."""
    import yaml  # lazy: only needed when a config file is supplied

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    block = data.pop("settings", None) or {}
    return PipelineConfig(**_substitute_env(data)), _settings_model(surface)(**block)


# Providers with a concrete pretranslation protocol adapter.
_AGENT_OK = {"vllm", "openai", "azure-openai", "anthropic", "gemini"}
_PRETRANSLATION_OK = {"vllm", "openai", "azure-openai", "anthropic"}
# Voice's classifier calls are bare ``chat.completions`` requests.
_RAW_OPENAI_OK = {"vllm", "openai", "azure-openai"}
_POST_TRANSLATION_OK = {
    "vllm", "openai", "azure-openai", "anthropic", "gemini", "translategemma"
}


def _built_as_raw_openai(step: Step, surface: Optional[str] = None) -> bool:
    """Whether this config's step tiers are built as bare OpenAI clients.

    ``non_meaningful`` always is. Moderation and pretranslation are on the voice
    channel (``surface``, else this deployment's), where they are voice's own
    ``chat.completions`` calls rather than chat's moderation agent and
    provider-native pretranslation.
    """
    from app.llm_core import config_source
    from app.llm_core.config_model import StepClientKind
    from app.llm_core.factory import STEP_CLIENT_KIND

    if step in (Step.MODERATION, Step.PRE_TRANSLATION) and (surface or config_source.channel()) == "voice":
        return True
    return STEP_CLIENT_KIND[step] is StepClientKind.RAW_OPENAI


def validate_config(pipeline: PipelineConfig, surface: Optional[str] = None) -> None:
    """Structurally validate the normalized tiers that can execute, by ``surface``'s
    rules (this deployment's channel when left out)."""
    problems: list[str] = []
    for profile in pipeline.profiles:
        if profile.weight > 0 and pipeline.step_plan(profile, Step.AGENT) is None:
            problems.append(f"profile={profile.name} has no agent plan")
        for step in Step:
            plan = pipeline.step_plan(profile, step)
            if plan is None:
                continue
            allowed = (
                _RAW_OPENAI_OK
                if _built_as_raw_openai(step, surface)
                else _PRETRANSLATION_OK
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


def validate_content(cfg: PipelineConfig, surface: Optional[str] = None) -> None:
    """Validate active config without constructing or caching provider clients.

    ``surface`` is the config's own surface when the deployment loads one per
    surface, and its rules apply. Left out, the config is the deployment's only
    one, checked by PIPELINE_CHANNEL's rules as before."""
    validate_config(cfg, surface)
    if _truthy_env("REQUIRE_OVERFLOW_ARMED") and not cfg.fallback_enabled:
        raise ValueError("REQUIRE_OVERFLOW_ARMED=true but fallback_enabled=false")
    if surface is None:
        validate_voice(cfg)
    elif surface == "voice":
        validate_voice(cfg, check_channel=False)


# What a voice call runs on (app/voice): moderation, pretranslation, the agent and
# output translation on the call's own profile, and the non-meaningful and consent
# classifiers on the managed one.
_VOICE_PROFILE_STEPS = (Step.MODERATION, Step.PRE_TRANSLATION, Step.AGENT, Step.POST_TRANSLATION)


def validate_voice(cfg: PipelineConfig, *, check_channel: bool = True) -> None:
    """With VOICE_ROUTE_ENABLED, refuse a config voice can't run on.

    The route on its own would boot fine on chat's channel and chat's plans, which
    have no non_meaningful step, and voice would then fail open or call the wrong
    models. Boot and every live config go through this. Voice's own file (with
    PIPELINE_CONFIG_DIR) is voice's by where it is, so ``check_channel`` is off."""
    from app.config import settings
    from app.llm_core import config_source

    if not settings.voice_route_enabled:
        return
    problems: list[str] = []
    if check_channel and config_source.channel() != "voice":
        problems.append(f"PIPELINE_CHANNEL is {config_source.channel()!r}, voice needs 'voice'")
    for profile in cfg.profiles:
        if profile.weight <= 0:
            continue
        for step in _VOICE_PROFILE_STEPS:
            if cfg.step_plan(profile, step) is None:
                problems.append(f"profile={profile.name} has no {step.value} plan")
    managed = cfg.by_name("managed") or cfg.profiles[0]
    if cfg.step_plan(managed, Step.NON_MEANINGFUL) is None:
        problems.append(f"profile={managed.name} has no non_meaningful plan")
    if problems:
        raise ValueError(
            "VOICE_ROUTE_ENABLED but the pipeline config can't run voice:\n  - " + "\n  - ".join(problems)
        )


def _truthy_env(name: str) -> bool:
    v = get_config_value(name)
    return v is not None and v.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_env(name: str, default: int) -> int:
    raw = get_config_value(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default
    return value if value > 0 else default


def normalize_config(pipeline: PipelineConfig) -> PipelineConfig:
    """Resolve plan-wide capabilities once for every config ingress."""
    history_override = _positive_int_env("CHAT_HISTORY_MAX_TOKENS", 0)
    history_default = history_override or 80_000
    gemma_default = history_override or _positive_int_env(
        "CHAT_HISTORY_MAX_TOKENS_VLLM_GEMMA", 10_000
    )
    profiles = [
        profile
        if pipeline.step_plan(profile, Step.AGENT) is None
        else profile.model_copy(update={
            "capabilities": pipeline.effective_capabilities(
                profile,
                history_default_tokens=history_default,
                history_vllm_gemma_tokens=gemma_default,
            )
        })
        for profile in pipeline.profiles
    ]
    return pipeline.model_copy(update={"profiles": profiles})


class BootRefused(RuntimeError):
    """Intentional hard-gate boot failure (e.g. REQUIRE_OVERFLOW_ARMED with overflow
    DISARMED). The startup call site re-raises this instead of swallowing it."""


def _log_boot_posture(pipeline: PipelineConfig) -> None:
    """Emit a LOUD one-line 'overflow ARMED / DISARMED' posture summary at boot.

    The whole overflow system — the OSS->managed attempt chain AND the health +
    concurrency guards, which fire ONLY via the fallback walkers — is inert unless
    ``FALLBACK_ENABLED`` is on. A deploy from defaults could therefore ship dark
    with nothing in the logs saying so. This line makes the armament state greppable
    at startup (``grep 'llm_core posture'``): INFO when armed, WARNING when disarmed.

    The shared config validator separately enforces ``REQUIRE_OVERFLOW_ARMED``."""
    from app.config import settings

    def _onoff(b: bool) -> str:
        return "on" if b else "off"

    fallback_on = pipeline.fallback_enabled
    plans = (
        pipeline.step_plan(profile, step)
        for profile in pipeline.profiles
        for step in Step
    )
    conc = "on" if any(
        plan is not None and plan.concurrency_gate is not None for plan in plans
    ) else "off"
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


def _load(path: Optional[str], surface: Optional[str]) -> tuple[PipelineConfig, SurfaceSettings]:
    """Load, normalize and validate one config: the file at ``path`` if there is
    one, else the env shim. ``surface`` is as in ``validate_content``."""
    from app.llm_core import config_source

    if path and os.path.exists(path):
        logger.info("llm_core: loading pipeline config from %s", path)
        candidate, surface_settings = _load_from_yaml(path, surface or config_source.channel())
    else:
        candidate, surface_settings = synthesize_from_env(), SurfaceSettings()
        logger.info(
            "llm_core: synthesized pipeline config from env (profiles=%s)",
            [f"{p.name}:{p.weight}" for p in candidate.profiles],
        )
    candidate = normalize_config(candidate)
    validate_content(candidate, surface)
    return candidate, surface_settings


def _load_surfaces(config_dir: str) -> dict[str, tuple[PipelineConfig, SurfaceSettings]]:
    """Chat's and voice's configs from PIPELINE_CONFIG_DIR. Chat's file is optional
    (the env shim builds it); voice's is required while the voice route is on."""
    from app.config import settings
    from app.llm_core import config_source

    if get_config_value("PIPELINE_CONFIG_PATH") or get_config_value(config_source.CHANNEL_ENV):
        raise ValueError(f"{CONFIG_DIR_ENV} replaces PIPELINE_CONFIG_PATH and PIPELINE_CHANNEL; unset them")
    voice_path = os.path.join(config_dir, "voice.yaml")
    if settings.voice_route_enabled and not os.path.exists(voice_path):
        raise ValueError(f"VOICE_ROUTE_ENABLED but {voice_path} is missing")
    loaded = {"chat": _load(os.path.join(config_dir, "chat.yaml"), "chat")}
    if os.path.exists(voice_path):
        loaded["voice"] = _load(voice_path, "voice")
    return loaded


def _log_boot(pipeline: PipelineConfig, surface: Optional[str]) -> None:
    # Tracing-only: dump the COMPLETE loaded config (all profiles, step tiers,
    # triggers) as one structured boot log line so the full wiring is greppable
    # in logs even before any turn arrives (`grep llm_core.full_config`).
    from app.llm_core import trace as _trace
    if surface is not None:
        logger.info("llm_core: boot config for the %s surface", surface)
    _trace.log_full_config(pipeline)
    _log_boot_posture(pipeline)
    # M2: note whether the live redis-backed config source is enabled (default OFF).
    # When on, weight changes PUT to the channel key take effect within the TTL with
    # no redeploy; when off, get_pipeline() serves the boot config only.
    from app.llm_core import config_source
    if config_source.enabled():
        logger.info(
            "llm_core: live redis config source ENABLED (channel=%s key=%s refresh=%ss) "
            "— weight changes PUT to that key take effect within the TTL, no redeploy",
            surface or config_source.channel(), config_source.key(surface), config_source.refresh_interval_s(),
        )
    else:
        logger.info(
            "llm_core: live redis config source disabled (%s unset) — serving boot config only",
            config_source.ENABLED_ENV,
        )


def configure() -> PipelineConfig:
    """Load, normalize, validate, and atomically publish pipeline config.

    With PIPELINE_CONFIG_DIR, one config per surface, each checked by its own
    surface's rules and published with its settings; returns chat's. Without it,
    the deployment's one config, as before."""
    global PIPELINE, BOOT_PIPELINE, PIPELINES, BOOT_PIPELINES, SETTINGS
    config_dir = get_config_value(CONFIG_DIR_ENV)
    try:
        if config_dir:
            loaded = _load_surfaces(config_dir)
        else:
            candidate, _ = _load(get_config_value("PIPELINE_CONFIG_PATH"), None)
    except Exception as exc:
        raise BootRefused(f"llm_core boot config rejected: {exc}") from exc

    # Publish only after every candidate has passed.
    if config_dir:
        PIPELINES = {surface: cfg for surface, (cfg, _) in loaded.items()}
        BOOT_PIPELINES = dict(PIPELINES)
        SETTINGS = {surface: surface_settings for surface, (_, surface_settings) in loaded.items()}
        PIPELINE = BOOT_PIPELINE = None
        for surface, cfg in PIPELINES.items():
            _log_boot(cfg, surface)
        return PIPELINES["chat"]
    PIPELINES, BOOT_PIPELINES, SETTINGS = {}, {}, {}
    PIPELINE = candidate
    # Capture the boot config as the permanent fallback BEFORE any live redis
    # refresh can override PIPELINE (get_pipeline -> config_source.maybe_refresh).
    # config_source reverts to THIS on a cleared/absent live key (emergency
    # rollback), never to a stale last-live config.
    BOOT_PIPELINE = PIPELINE
    _log_boot(PIPELINE, None)
    return PIPELINE


def get_pipeline(surface: str = "chat") -> PipelineConfig:
    """The config ``surface``'s turns run on: its own with PIPELINE_CONFIG_DIR (chat's
    for a surface without a file), else the deployment's one."""
    global PIPELINE
    if PIPELINE is None and not PIPELINES:
        configure()
    # M2 (live config): consult the redis-backed source. TTL-gated (hits redis at
    # most once per PIPELINE_CONFIG_REFRESH_S window) and fail-safe (any error ->
    # returns the last-good PIPELINE unchanged, never raises). When
    # PIPELINE_CONFIG_REDIS_ENABLED is unset/false this is an immediate identity
    # no-op, so behaviour is byte-identical to boot-config-only.
    from app.llm_core import config_source
    if PIPELINES:
        surface = surface if surface in PIPELINES else "chat"
        PIPELINES[surface] = config_source.maybe_refresh(PIPELINES[surface], surface)
        return PIPELINES[surface]
    assert PIPELINE is not None
    PIPELINE = config_source.maybe_refresh(PIPELINE)
    return PIPELINE


def surfaces() -> tuple[str, ...]:
    """The surfaces with a config of their own; just chat without PIPELINE_CONFIG_DIR."""
    return tuple(PIPELINES) or ("chat",)


def surface_settings(surface: str) -> SurfaceSettings:
    """The ``settings:`` block of the surface's file; all unset without
    PIPELINE_CONFIG_DIR, so every reader falls back to its env setting."""
    return SETTINGS.get(surface) or _NO_SETTINGS.get(surface) or SurfaceSettings()


def surface_setting(surface: str, name: str, fallback):
    """One of the surface's settings, or ``fallback`` (its env setting) when the
    file leaves it out."""
    value = getattr(surface_settings(surface), name)
    return fallback if value is None else value
