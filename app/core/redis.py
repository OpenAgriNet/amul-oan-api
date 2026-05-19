"""
Shared Redis client factory.

Single source of truth for Redis connections across the application.
All modules that need a raw ``redis.asyncio.Redis`` client should use
``get_shared_redis_client()`` instead of creating their own instance.

The aiocache-based ``cache`` in ``app.core.cache`` remains for
key-value caching operations (get/set with TTL).  This module
provides the lower-level client needed by modules like
``scheme_ingestion`` that use Redis features beyond simple k/v
(e.g. locks, SET NX, direct GET/SET with no serializer).
"""

from __future__ import annotations

from typing import Optional

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

_redis_client = None


async def get_shared_redis_client():
    """Return a singleton ``redis.asyncio.Redis`` client.

    Uses the centralised settings from ``app.config.settings`` so
    there is no risk of configuration drift between modules.

    Raises:
        RuntimeError: If the ``redis`` package is not installed.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as redis
    except ModuleNotFoundError as exc:
        logger.error("redis package is not installed")
        raise RuntimeError("redis is not installed") from exc

    logger.info(
        "Creating shared Redis client host=%s port=%s db=%s prefix=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_db,
        settings.redis_key_prefix,
    )

    _redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        socket_timeout=settings.redis_socket_timeout,
        retry_on_timeout=settings.redis_retry_on_timeout,
        max_connections=settings.redis_max_connections,
    )
    return _redis_client


async def close_shared_redis_client() -> None:
    """Gracefully close the shared Redis connection (call on shutdown)."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception as exc:
            logger.warning("Error closing shared Redis client: %s", exc)
        _redis_client = None
