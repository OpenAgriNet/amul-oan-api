"""
Core cache instance configuration using Redis and aiocache.

This module provides the cache instance that other parts of the application can use.
Uses enhanced Redis configuration with connection pooling and timeouts.
"""
from typing import Any

from aiocache import Cache
from aiocache.serializers import JsonSerializer
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)
FARMER_ANIMAL_API_CACHE_TTL = settings.farmer_animal_api_cache_ttl
_CACHE_SENTINEL = "__cached_api_response__"

cache = Cache(
    cache_class=Cache.REDIS, # pyright: ignore
    endpoint=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    serializer=JsonSerializer(),
    ttl=settings.default_cache_ttl,
    timeout=settings.redis_socket_timeout,
    pool_max_size=settings.redis_max_connections,
    key_builder=lambda key, namespace: f"{settings.redis_key_prefix}{namespace}:{key}" if namespace else f"{settings.redis_key_prefix}{key}",
)

logger.info(
    f"Cache configured with Redis at {settings.redis_host}:{settings.redis_port} "
    f"(DB: {settings.redis_db}, Prefix: {settings.redis_key_prefix}, "
    f"Max Connections: {settings.redis_max_connections})"
    + (" (password set)" if settings.redis_password else "")
)


def build_api_cache_key(api_name: str, number: str) -> str:
    return f"{api_name}:{number}"


async def get_cached_api_response(cache_key: str) -> tuple[bool, Any]:
    try:
        cached_value = await cache.get(cache_key)
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", cache_key, str(e))
        return False, None

    if isinstance(cached_value, dict) and cached_value.get(_CACHE_SENTINEL) is True:
        logger.info("Cache hit for %s", cache_key)
        return True, cached_value.get("value")

    logger.info("Cache miss for %s", cache_key)
    return False, None


async def set_cached_api_response(cache_key: str, value: Any) -> None:
    try:
        await cache.set(
            cache_key,
            {_CACHE_SENTINEL: True, "value": value},
            ttl=FARMER_ANIMAL_API_CACHE_TTL,
        )
        logger.info(
            "Cache set for %s with ttl=%s", cache_key, FARMER_ANIMAL_API_CACHE_TTL
        )
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", cache_key, str(e))
