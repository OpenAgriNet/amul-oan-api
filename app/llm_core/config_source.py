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
  at most once per ``PIPELINE_CONFIG_REFRESH_S`` (default 10s) window. Inside the
  window it returns the caller's config unchanged (zero Redis I/O), so calling it
  on every request is cheap.
* **Fail-safe (never raises to the caller)** — on ANY failure (redis disabled,
  client init error, redis down, missing key, invalid JSON, pydantic
  ``ValidationError``, weights != 100) ``maybe_refresh`` returns the last-good
  config unchanged and logs a rate-limited WARNING. The last successfully-loaded
  config + last-refresh time are cached, so a redis blip leaves the last-good
  config serving; nothing here can break a request path.
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

_DEFAULT_REFRESH_S = 10.0
_KEY_PREFIX = "llm_pipeline_config:"

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
    negative value degrades to the default rather than raising."""
    raw = os.getenv(REFRESH_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_REFRESH_S
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_REFRESH_S
    return val if val >= 0 else _DEFAULT_REFRESH_S


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
    (``app.config.settings`` — host/port/db/password), mirroring ``app.core.cache``
    (``decode_responses=True``, short socket timeouts). Also used by the ops
    script so the two never drift. Returns ``None`` if redis/config can't import
    (never raises)."""
    try:
        import redis  # sync client; the app already depends on redis (redis.asyncio)
        from app.config import settings

        return redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            socket_timeout=settings.redis_socket_timeout,
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


def _try_load() -> Optional[PipelineConfig]:
    """GET + parse + validate the live config, or ``None`` on ANY failure (the
    fail-safe signal). Never raises."""
    client = _get_redis()
    if client is None:
        return None
    k = key()
    try:
        raw = client.get(k)
    except Exception as e:  # redis down / timeout -> fail-safe
        _warn("pipeline config: redis GET failed for %s: %s; keeping last-good", k, e)
        return None
    if raw is None:
        return None  # key absent -> boot/last-good config stays; not worth warning
    try:
        data = json.loads(raw)
        cfg = PipelineConfig(**data)  # validates weights==100 + unique names
    except Exception as e:  # invalid JSON / ValidationError / weights!=100 -> fail-safe
        _warn("pipeline config: invalid live config at %s (%s); keeping last-good", k, e)
        return None
    logger.info(
        "pipeline config: loaded LIVE config from redis %s (profiles=%s)",
        k, [f"{p.name}:{p.weight}" for p in cfg.profiles],
    )
    return cfg


def maybe_refresh(current: PipelineConfig) -> PipelineConfig:
    """Return the live config if the source is enabled and a valid one is present,
    else ``current`` unchanged. TTL-gated (hits redis at most once per window) and
    fail-safe (any error keeps the last-good config). NEVER raises.

    Contract:
      * source disabled -> immediate identity (no redis client built);
      * within the TTL window -> return ``current`` (zero redis I/O — ``current``
        is already the last-good config, since ``runtime`` stores our return);
      * past the TTL -> GET the key; a valid config becomes the new last-good and
        is returned; any failure keeps the last-good (``current``) serving.
    """
    global _last_refresh_monotonic, _last_good
    if not enabled():
        return current

    now = time.monotonic()
    if _last_refresh_monotonic and (now - _last_refresh_monotonic) < refresh_interval_s():
        # TTL not elapsed: cheap no-op. `current` is what runtime last stored.
        return current

    _last_refresh_monotonic = now
    loaded = _try_load()
    if loaded is not None:
        _last_good = loaded
        return loaded
    # Any failure: keep last-good. Prefer our cached last-good over `current` only
    # if we somehow have a newer one; normally they are the same object.
    return _last_good if _last_good is not None else current


def reset() -> None:
    """Clear cached state + client (TEST/ops helper — e.g. after flipping env vars
    in a test, or to force the next ``maybe_refresh`` to re-read). Not called on
    the request path."""
    global _last_refresh_monotonic, _last_good, _last_warn_monotonic
    global _redis_client, _redis_init_failed
    _last_refresh_monotonic = 0.0
    _last_good = None
    _last_warn_monotonic = 0.0
    _redis_client = None
    _redis_init_failed = False
