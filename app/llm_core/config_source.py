"""M2: redis-backed LIVE pipeline-config source (no-redeploy % changes).

The boot config (``runtime.configure`` — env-shim or ``PIPELINE_CONFIG_PATH``
YAML) is the initial AND permanent fallback. This module lets an operator PUT a
new :class:`PipelineConfig` into Redis and have ``get_pipeline()`` pick it up
within a short TTL — WITHOUT a redeploy — so a weight change (e.g. an OSS profile
0 -> 50%) goes live in seconds. Because ``split.deterministic_profile`` re-buckets
every request against the CURRENT weights (no Redis session pin), continuing
sessions FOLLOW the new % automatically (the refresh-on-change contract).

Design
------
* **Redis key** ``llm_pipeline_config:{channel}`` holds the JSON of a
  ``PipelineConfig`` (``model_dump(mode="json")``). Secrets are NEVER in it — a
  tier only names its ``api_key_env``; the VALUE is read from the environment at
  materialize time, exactly as for the boot config.
* **channel** — ``PIPELINE_CHANNEL`` env, defaulting to the repo's identity
  (``voice`` if the ``Step`` enum has the voice-only ``non_meaningful`` step,
  else ``chat``) so each deployment self-identifies without any per-repo code
  edit. Chat and voice therefore read DISTINCT keys off a shared Redis.
* **TTL** — ``maybe_refresh`` is gated by ``time.monotonic()``: it touches Redis
  at most once per ``PIPELINE_CONFIG_REFRESH_S`` (default 10s; a non-positive value
  degrades to the default so it can never GET-per-request) window. Inside the
  window it returns the caller's config unchanged (zero Redis I/O), so calling it
  on every request is cheap.
* **Boot-time validation on the LIVE path (fail-CLOSED on bad content)** — a
  live config is not merely schema-checked: after parse it is run through
  ``runtime.validate_content`` (the SAME content gates the boot path applies —
  provider/step legality + a resolvability probe that builds every profile/step
  primary handle). A schema-valid but UNBUILDABLE config (vllm tier with no
  endpoint, absent ``api_key_env``, anthropic/gemini on a RAW_OPENAI step, a
  profile missing a required step) is therefore REJECTED — treated exactly like a
  read failure (last-good kept, rate-limited WARNING) so a bad push can never go
  live and break requests.
* **Fail-safe (never raises to the caller)** — on a TRANSIENT read failure (redis
  disabled, client init error, redis down, invalid JSON, ``ValidationError``,
  weights != 100, content-invalid) ``maybe_refresh`` returns the last-good config
  unchanged and logs a rate-limited WARNING. The last successfully-loaded config +
  last-refresh time are cached, so a redis blip leaves the last-good config
  serving; nothing here can break a request path.
* **Clear / key-absent reverts to BOOT (emergency rollback)** — when the source is
  ENABLED but the key is ABSENT (operator ran ``clear``, or it was never set),
  ``maybe_refresh`` returns ``runtime.BOOT_PIPELINE`` — the config captured at
  ``configure()`` BEFORE any live refresh — NOT the last LIVE config. So deleting
  the key is a true rollback to the deploy's boot config. (A present-but-unreadable
  key is a transient error, above, and keeps last-good — it does not nuke to boot.)
* **Short dedicated socket timeout** — the config-source redis client is built with
  its own short connect+socket timeout (``PIPELINE_CONFIG_REDIS_TIMEOUT_S``, default
  0.5s), NOT the app-wide ``redis_socket_timeout`` (up to 10s), so a slow/down redis
  costs the request hot path ≤0.5s per TTL window, not up to 10s.
* **Default OFF** — with ``PIPELINE_CONFIG_REDIS_ENABLED`` unset/false,
  ``maybe_refresh`` is an immediate identity no-op (no redis client is even
  built), so ``get_pipeline()`` is behaviorally identical to boot-config-only.

Kept import-clean (stdlib + pydantic + ``config_model`` at import time; ``redis``
and ``app.config`` are imported lazily inside the client builder) so it stays
byte-identical across the chat and voice repos and the eventual repo-merge is a
mechanical convergence. The synchronous ``redis`` client is deliberate:
``get_pipeline()`` is sync and on the request path, and the read is TTL-gated to
at most once per window with a short socket timeout.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from helpers.utils import get_logger
from app.llm_core.config_model import PipelineConfig, Step

logger = get_logger(__name__)

# ── env knobs ────────────────────────────────────────────────────────────────
ENABLED_ENV = "PIPELINE_CONFIG_REDIS_ENABLED"   # default off
CHANNEL_ENV = "PIPELINE_CHANNEL"                # default self-identifies (chat/voice)
REFRESH_ENV = "PIPELINE_CONFIG_REFRESH_S"       # TTL seconds, default 10
TIMEOUT_ENV = "PIPELINE_CONFIG_REDIS_TIMEOUT_S" # dedicated short socket timeout, default 0.5s

_DEFAULT_REFRESH_S = 10.0
_DEFAULT_REDIS_TIMEOUT_S = 0.5
_KEY_PREFIX = "llm_pipeline_config:"

# Sentinel: the redis key is ABSENT (cleared / never-set) — distinct from both a
# valid config and a transient read error (None). Signals a revert to BOOT.
_KEY_ABSENT = object()

# Repo self-identification: the voice Step enum has the voice-only NON_MEANINGFUL
# step, chat has SUGGESTIONS instead — so this one expression yields "voice" in
# the voice repo and "chat" in the chat repo with ZERO per-repo code difference.
_DEFAULT_CHANNEL = "voice" if hasattr(Step, "NON_MEANINGFUL") else "chat"

# Rate-limit the fail-safe WARNING so a persistent redis/config fault (hit once
# per TTL window) cannot spam the logs.
_WARN_INTERVAL_S = 60.0

# ── module state (cache) ─────────────────────────────────────────────────────
_last_refresh_monotonic: float = 0.0   # 0.0 => never refreshed this process
_last_good: Optional[PipelineConfig] = None
_last_warn_monotonic: float = 0.0
_redis_client = None                   # lazily built; None until first use
_redis_init_failed: bool = False       # latch so we don't retry a broken import
# Reentrancy guard: the LIVE-path content validation (``runtime.validate_content``)
# resolves the candidate config through ``resolver`` -> ``runtime.get_pipeline`` ->
# back into ``maybe_refresh``. While that probe runs, ``maybe_refresh`` must be an
# identity no-op (return ``current``) so it neither re-reads redis nor recurses.
_suppress_refresh: bool = False


def _truthy(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Whether the live redis config source is turned on (default OFF)."""
    return _truthy(ENABLED_ENV)


def channel() -> str:
    """The config channel this deployment reads — ``PIPELINE_CHANNEL`` or the
    repo default (``chat`` / ``voice``). Chat and voice read distinct keys."""
    v = os.getenv(CHANNEL_ENV)
    v = v.strip() if v else ""
    return v or _DEFAULT_CHANNEL


def key(chan: Optional[str] = None) -> str:
    """Redis key for a channel (defaults to this deployment's channel)."""
    return f"{_KEY_PREFIX}{chan or channel()}"


def refresh_interval_s() -> float:
    """TTL window in seconds (``PIPELINE_CONFIG_REFRESH_S``, default 10). A bad or
    NON-POSITIVE value degrades to the default rather than raising — a 0 (or
    negative) window would otherwise GET redis on EVERY request (defeating the TTL
    and putting a blocking read on every hot-path call), so clamp it to the
    default."""
    raw = os.getenv(REFRESH_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_REFRESH_S
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_REFRESH_S
    return val if val > 0 else _DEFAULT_REFRESH_S


def redis_timeout_s() -> float:
    """Dedicated SHORT connect+socket timeout for the config-source redis client
    (``PIPELINE_CONFIG_REDIS_TIMEOUT_S``, default 0.5s). Deliberately independent of
    the app-wide ``redis_socket_timeout`` (up to 10s): this read sits on the request
    hot path (once per TTL window), so a slow/down redis must cost ≤0.5s, not 10s. A
    bad or non-positive value degrades to the default rather than raising."""
    raw = os.getenv(TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_REDIS_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_REDIS_TIMEOUT_S
    return val if val > 0 else _DEFAULT_REDIS_TIMEOUT_S


def _warn(msg: str, *args) -> None:
    """Rate-limited WARNING — at most once per ``_WARN_INTERVAL_S`` so a stuck
    fault cannot spam the logs."""
    global _last_warn_monotonic
    now = time.monotonic()
    if _last_warn_monotonic and (now - _last_warn_monotonic) < _WARN_INTERVAL_S:
        return
    _last_warn_monotonic = now
    logger.warning(msg, *args)


def build_redis_client():
    """Build a synchronous redis client from the SAME connection env the app uses
    (``app.config.settings`` — host/port/db/password), but with a DEDICATED short
    connect+socket timeout (``redis_timeout_s()``, default 0.5s) rather than the
    app-wide ``redis_socket_timeout`` — so a slow/down redis on the request hot path
    costs ≤0.5s, not up to 10s. ``decode_responses=True`` mirrors ``app.core.cache``.
    Also used by the ops script so the two never drift. Returns ``None`` if
    redis/config can't import (never raises)."""
    try:
        import redis  # sync client; the app already depends on redis (redis.asyncio)
        from app.config import settings

        timeout = redis_timeout_s()
        return redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            decode_responses=True,
        )
    except Exception as e:  # import / settings failure -> fail-safe (no live source)
        _warn("pipeline config: redis client build failed: %s", e)
        return None


def _get_redis():
    """Lazily build + cache the sync redis client. Tests monkeypatch THIS function
    (return a fake redis) so no real redis is contacted."""
    global _redis_client, _redis_init_failed
    if _redis_client is not None:
        return _redis_client
    if _redis_init_failed:
        return None
    client = build_redis_client()
    if client is None:
        _redis_init_failed = True
        return None
    _redis_client = client
    return _redis_client


def _try_load():
    """GET + parse + fully validate the live config. Tri-state return (never raises):

      * a :class:`PipelineConfig` — a valid, BUILDABLE live config to apply;
      * ``_KEY_ABSENT`` — the key is absent (cleared / never-set): the caller must
        revert to BOOT (not last-live);
      * ``None`` — a TRANSIENT read failure (no client / redis down / invalid JSON /
        weights!=100 / content-invalid): the caller keeps the last-good config.

    Content validation makes the LIVE path FAIL-CLOSED: a schema-valid but
    unbuildable config (``runtime.validate_content`` raises) is treated as a read
    failure so it can never go live."""
    client = _get_redis()
    if client is None:
        return None  # no client -> transient; keep last-good
    k = key()
    try:
        raw = client.get(k)
    except Exception as e:  # redis down / timeout -> transient; keep last-good
        _warn("pipeline config: redis GET failed for %s: %s; keeping last-good", k, e)
        return None
    if raw is None:
        return _KEY_ABSENT  # cleared / never-set -> caller reverts to BOOT
    try:
        data = json.loads(raw)
        cfg = PipelineConfig(**data)  # validates weights==100 + unique names
    except Exception as e:  # invalid JSON / ValidationError / weights!=100 -> transient
        _warn("pipeline config: invalid live config at %s (%s); keeping last-good", k, e)
        return None
    # FAIL-CLOSED content gate: run the SAME checks the boot path applies (provider/
    # step legality + a resolvability probe that builds each profile/step primary
    # handle). A schema-valid but unbuildable config is rejected like a read failure
    # so a bad push cannot go live. ``validate_content`` is per-repo in ``runtime``;
    # calling ONLY it here keeps this module byte-identical across chat and voice.
    try:
        from app.llm_core import runtime
        runtime.validate_content(cfg)
    except Exception as e:  # unbuildable content -> fail-closed; keep last-good
        _warn("pipeline config: content-invalid live config at %s (%s); keeping last-good", k, e)
        return None
    logger.info(
        "pipeline config: loaded LIVE config from redis %s (profiles=%s)",
        k, [f"{p.name}:{p.weight}" for p in cfg.profiles],
    )
    return cfg


def maybe_refresh(current: PipelineConfig) -> PipelineConfig:
    """Return the live config if the source is enabled and a valid one is present,
    else the appropriate fallback. TTL-gated (hits redis at most once per window)
    and fail-safe (never raises).

    Contract:
      * source disabled -> immediate identity (no redis client built);
      * a validation probe is in flight (reentrant call) -> identity no-op;
      * within the TTL window -> return ``current`` (zero redis I/O — ``current``
        is already the last-good config, since ``runtime`` stores our return);
      * past the TTL -> GET the key:
          - a valid, BUILDABLE config becomes the new last-good and is returned;
          - key ABSENT (cleared / never-set) -> ``runtime.BOOT_PIPELINE`` (revert to
            the deploy's boot config — emergency rollback, NOT the last-live config);
          - a transient failure (redis down / invalid / content-invalid) keeps the
            last-good (``current``) serving.
    """
    global _last_refresh_monotonic, _last_good, _suppress_refresh
    if _suppress_refresh:
        # Reentrant call from a content-validation probe: never re-read or recurse.
        return current
    if not enabled():
        return current

    now = time.monotonic()
    if _last_refresh_monotonic and (now - _last_refresh_monotonic) < refresh_interval_s():
        # TTL not elapsed: cheap no-op. `current` is what runtime last stored.
        return current

    _last_refresh_monotonic = now
    _suppress_refresh = True  # guard the validate_content probe inside _try_load
    try:
        loaded = _try_load()
    finally:
        _suppress_refresh = False

    if loaded is _KEY_ABSENT:
        # Cleared / never-set -> revert to the BOOT config (NOT the last LIVE one),
        # so `clear` is a true emergency rollback. Guard against BOOT not yet
        # captured (pre-configure) by degrading to `current`.
        from app.llm_core import runtime
        boot = runtime.BOOT_PIPELINE
        if boot is not None:
            _last_good = boot
            return boot
        return current
    if loaded is not None:
        _last_good = loaded
        return loaded
    # Transient failure: keep last-good. Prefer our cached last-good over `current`
    # only if we somehow have a newer one; normally they are the same object.
    return _last_good if _last_good is not None else current


def reset() -> None:
    """Clear cached state + client (TEST/ops helper — e.g. after flipping env vars
    in a test, or to force the next ``maybe_refresh`` to re-read). Not called on
    the request path."""
    global _last_refresh_monotonic, _last_good, _last_warn_monotonic
    global _redis_client, _redis_init_failed, _suppress_refresh
    _last_refresh_monotonic = 0.0
    _last_good = None
    _last_warn_monotonic = 0.0
    _redis_client = None
    _redis_init_failed = False
    _suppress_refresh = False
