"""Runtime holder + startup self-check for the unified pipeline.

``configure()`` (called from the FastAPI lifespan) loads
``PIPELINE_CONFIG_PATH`` YAML when present, else synthesizes the config from the
current env (``legacy_shim``), validates it, stores it in the module global
``PIPELINE``, and runs the identity self-check. ``get_pipeline()`` lazily
configures on first use so request paths and tests never see ``None``.

The startup self-check logs the resolved (provider, base URL, model, timeout)
for every configured step and verifies that every primary handle can be built.
Agents carry no construction-time model; this module is the only runtime model
selection path.
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
_PRETRANSLATION_OK = {"vllm", "openai", "azure-openai", "anthropic"}
_POST_TRANSLATION_OK = {"vllm", "openai", "azure-openai", "translategemma"}


def validate_config(pipeline: PipelineConfig) -> None:
    """Fail fast when a configured tier has no protocol adapter."""
    from app.llm_core.config_model import StepClientKind
    from app.llm_core.factory import STEP_CLIENT_KIND

    raw_steps = [
        step
        for step, kind in STEP_CLIENT_KIND.items()
        if kind is StepClientKind.PRE_TRANSLATION
    ]
    problems: list[str] = []
    for profile in pipeline.profiles:
        for step in raw_steps:
            cfg = pipeline.step_config(profile, step)
            if cfg is None:
                continue
            for tier in cfg.tiers:
                if tier.provider.value not in _PRETRANSLATION_OK:
                    problems.append(
                        f"profile={profile.name} step={step.value} "
                        f"provider={tier.provider.value} has no pretranslation adapter "
                        f"(allowed: {sorted(_PRETRANSLATION_OK)})"
                    )
        post = pipeline.step_config(profile, Step.POST_TRANSLATION)
        if post is not None:
            for tier in post.tiers:
                if tier.provider.value not in _POST_TRANSLATION_OK:
                    problems.append(
                        f"profile={profile.name} step={Step.POST_TRANSLATION.value} "
                        f"provider={tier.provider.value} has no post-translation adapter "
                        f"(allowed: {sorted(_POST_TRANSLATION_OK)})"
                    )
    if not problems:
        return
    msg = (
        "llm_core config INVALID — unsupported provider adapter:\n  - "
        + "\n  - ".join(problems)
    )
    raise ValueError(msg)


def validate_content(cfg: PipelineConfig) -> None:
    """Run the SAME content gates the boot path applies against a CANDIDATE config
    (a live redis config, or an ops-script payload) BEFORE it goes live — raising on
    any unbuildable content. This is the single validator both ``config_source``
    (fail-CLOSED live load) and ``scripts/set_pipeline_config.py`` (refuse-to-write)
    call, so a schema-valid but unbuildable config can never go live and break
    requests.

    Two checks, mirroring boot:
      (a) ``validate_config(cfg)`` — provider/step legality; and
      (b) a resolvability probe — for every profile, every configured step, and every
          tier, build its handle; the factory raises on
          an unbuildable tier (vllm tier with no endpoint, azure tier missing
          api_key_env/api_version, etc.), exactly as the boot self-check would.

    Raises (never swallows) so callers can fail closed."""
    validate_config(cfg)
    from app.llm_core.factory import STEP_CLIENT_KIND, build_handle, tier_client_kind

    for profile in cfg.profiles:
        for step in Step:
            step_config = cfg.step_config(profile, step)
            if step_config is None:
                continue
            for tier in step_config.tiers:
                build_handle(tier, tier_client_kind(STEP_CLIENT_KIND[step], tier))


def _truthy_env(name: str) -> bool:
    v = get_config_value(name)
    return v is not None and v.strip().lower() in {"1", "true", "yes", "on"}


class BootRefused(RuntimeError):
    """Intentional hard-gate boot failure (e.g. REQUIRE_OVERFLOW_ARMED with overflow
    DISARMED). Distinct type so the best-effort ``configure()`` call site in main.py
    can re-raise it (a deliberate refusal to boot) while still swallowing genuine
    non-fatal configure/self-check edge cases."""


def _assert_boot_posture() -> None:
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

    fallback_on = bool(PIPELINE and PIPELINE.fallback_enabled)
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


def configure(*, run_self_check: bool = True) -> PipelineConfig:
    """Load / synthesize the pipeline config, validate, store, self-check."""
    global PIPELINE, BOOT_PIPELINE
    path = get_config_value("PIPELINE_CONFIG_PATH")
    if path and os.path.exists(path):
        logger.info("llm_core: loading pipeline config from %s", path)
        PIPELINE = _load_from_yaml(path)
    else:
        PIPELINE = synthesize_from_env()
        logger.info(
            "llm_core: synthesized pipeline config from env (profiles=%s)",
            [f"{p.name}:{p.weight}" for p in PIPELINE.profiles],
        )
    # (E) Provider/step legality — fail-fast at boot. The unified pipeline is the
    # only path after P4 (the LLM_CORE_ENABLED kill-switch was removed), so the
    # config binding is always the live one and must always be legal: enforce.
    validate_config(PIPELINE)
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
    # Boot posture assertion: LOUD ARMED/DISARMED overflow summary (+ hard-gate via
    # REQUIRE_OVERFLOW_ARMED). Placed after config load so a hard-gate raise fires
    # before the (non-fatal) self-check.
    _assert_boot_posture()
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
    if run_self_check:
        try:
            self_check()
        except AssertionError:
            raise
        except Exception as exc:  # never break config load on a self-check bug
            logger.warning("llm_core: self-check skipped (%s)", exc)
    return PIPELINE


def get_pipeline() -> PipelineConfig:
    global PIPELINE
    if PIPELINE is None:
        configure(run_self_check=False)
    assert PIPELINE is not None
    # M2 (live config): consult the redis-backed source. TTL-gated (hits redis at
    # most once per PIPELINE_CONFIG_REFRESH_S window) and fail-safe (any error ->
    # returns the last-good PIPELINE unchanged, never raises). When
    # PIPELINE_CONFIG_REDIS_ENABLED is unset/false this is an immediate identity
    # no-op, so behaviour is byte-identical to boot-config-only.
    from app.llm_core import config_source
    PIPELINE = config_source.maybe_refresh(PIPELINE)
    return PIPELINE


def _base_url(handle) -> Optional[str]:
    b = getattr(handle, "base_url", None)
    if b is None:
        b = getattr(getattr(handle, "client", None), "base_url", None)
    return str(b).rstrip("/") if b is not None else None


def self_check() -> None:
    """Startup validation: every profile's every step must resolve to a live
    primary tier (build a handle without raising) for the current config.

    This is the P4 successor to the P0/P1 identity self-check. There is no longer a
    legacy wiring to compare against — the unified pipeline is the only path — so
    the check now just logs the resolved (provider, base_url, model, timeout) per
    configured step and WARNS on any step that fails to resolve. It is
    intentionally non-fatal: a handle-build edge case (e.g. a fallback-tier key
    absent in this env) must never block startup, exactly as the flag-off boot was
    robust before. Genuine config-shape errors are already caught by
    ``PipelineConfig``'s validator at load time.
    """
    from app.llm_core.factory import STEP_CLIENT_KIND, build_handle, tier_client_kind

    pipeline = get_pipeline()
    failures: list[str] = []

    for profile in pipeline.profiles:
        for step in Step:
            step_cfg = pipeline.step_config(profile, step)
            if step_cfg is None:
                continue  # a profile need not configure every step (post-trans lives in defaults)
            try:
                # Resolve BY PROFILE NAME (N-way): a broken 3rd-profile tier (bad
                # provider/endpoint/key) is caught here at boot, not just oss/managed.
                tier = step_cfg.tiers[0]
                handle = build_handle(
                    tier, tier_client_kind(STEP_CLIENT_KIND[step], tier)
                )
                logger.info(
                    "llm_core self-check profile=%s step=%s -> provider=%s base_url=%s model=%s timeout=%s",
                    profile.name,
                    step.value,
                    tier.provider.value,
                    _base_url(handle),
                    tier.model,
                    tier.timeout_ms / 1000.0 if tier.timeout_ms is not None else None,
                )
            except Exception as exc:
                failures.append(f"{profile.name}/{step.value}: {type(exc).__name__}: {exc}")

    if failures:
        logger.warning(
            "llm_core self-check: %d step(s) did not resolve in this env (non-fatal):\n  - %s",
            len(failures), "\n  - ".join(failures),
        )
    else:
        logger.info(
            "llm_core self-check PASSED: every configured step resolves (profiles=%s)",
            [p.name for p in pipeline.profiles],
        )
