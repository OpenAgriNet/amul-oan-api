"""Unified cache/client access and common cache helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from aiocache import Cache
from aiocache.serializers import JsonSerializer
from redis.asyncio import Redis

from app.config import settings
from app.redis.config import (
    AI_CALL_COOLDOWN_TTL_SECONDS,
    DEFAULT_CACHE_TTL_SECONDS,
    FARMER_ANIMAL_API_CACHE_TTL_SECONDS,
    FARMER_CACHE_TTL_SECONDS,
    key,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

_CACHE_SENTINEL = "__cached_api_response__"
FARMER_NAMESPACE = "farmer"
AI_CALL_NAMESPACE = "ai_call_booked"


cache = Cache(
    cache_class=Cache.REDIS,  # pyright: ignore
    endpoint=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    serializer=JsonSerializer(),
    ttl=DEFAULT_CACHE_TTL_SECONDS,
    timeout=settings.redis_socket_timeout,
    pool_max_size=settings.redis_max_connections,
    key_builder=lambda value, namespace: (
        f"{settings.redis_key_prefix}{namespace}:{value}"
        if namespace
        else f"{settings.redis_key_prefix}{value}"
    ),
)

redis_client = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    socket_connect_timeout=settings.redis_socket_connect_timeout,
    socket_timeout=settings.redis_socket_timeout,
    max_connections=settings.redis_max_connections,
    retry_on_timeout=settings.redis_retry_on_timeout,
    decode_responses=True,
)


async def get_cache(cache_key: str, *, namespace: Optional[str] = None):
    return await cache.get(cache_key, namespace=namespace)


async def set_cache(
    cache_key: str,
    value: Any,
    *,
    ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    namespace: Optional[str] = None,
) -> bool:
    await cache.set(cache_key, value, ttl=ttl, namespace=namespace)
    return True


def build_api_cache_key(api_name: str, number: str) -> str:
    return key("api", f"{api_name}:{number}")


async def get_cached_api_response(cache_key: str) -> tuple[bool, Any]:
    try:
        cached_value = await get_cache(cache_key)
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
        await set_cache(
            cache_key,
            {_CACHE_SENTINEL: True, "value": value},
            ttl=FARMER_ANIMAL_API_CACHE_TTL_SECONDS,
        )
        logger.info(
            "Cache set for %s with ttl=%s",
            cache_key,
            FARMER_ANIMAL_API_CACHE_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", cache_key, str(e))


def farmer_cache_key(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()


async def get_farmer_cache(phone: str):
    return await get_cache(farmer_cache_key(phone), namespace=FARMER_NAMESPACE)


async def set_farmer_cache(phone: str, value: Any) -> bool:
    return await set_cache(
        farmer_cache_key(phone),
        value,
        ttl=FARMER_CACHE_TTL_SECONDS,
        namespace=FARMER_NAMESPACE,
    )


async def get_ai_call_cooldown(session_id: str):
    return await get_cache(session_id, namespace=AI_CALL_NAMESPACE)


async def set_ai_call_cooldown(session_id: str, value: Any) -> bool:
    return await set_cache(
        session_id,
        value,
        ttl=AI_CALL_COOLDOWN_TTL_SECONDS,
        namespace=AI_CALL_NAMESPACE,
    )

