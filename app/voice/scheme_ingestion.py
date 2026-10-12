"""Read-only cache access for milk producer schemes."""

import json
from typing import Any

from app.config import settings
from app.voice.models.union import UnionName, canonical_union_name
from helpers.utils import get_logger

logger = get_logger(__name__)

SCHEME_CACHE_NAMESPACE = "milk_producer_schemes"
_redis_client = None


class SchemeIngestionError(Exception):
    """Base error for scheme ingestion failures."""


class SchemeDependencyError(SchemeIngestionError):
    """Raised when an optional dependency is unavailable."""


class SchemeCacheError(SchemeIngestionError):
    """Raised when Redis cache access fails."""


# Cache keys must match amul-oan-api scheme ingestion (chat writes, voice reads).
# Tool + farmer-context gating derive from this map so a union cannot be
# exposed without a readable cache source, or vice versa.
SUPPORTED_UNION_SOURCE_KEYS = {
    UnionName.BANAS.value: ("banasdairy.coop/home/inputactivities#milkproducers",),
    UnionName.KUTCH.value: ("sarhaddairy.coop/for-our-milk-producers",),
    UnionName.SUMUL.value: ("sumul.com/farmer-section",),
    UnionName.SURENDRANAGAR.value: ("sursagardairy.com/farmer/milkproducers",),
    UnionName.BHARUCH.value: ("dudhdharadairy.in/for_our_milk_producers",),
    UnionName.SABAR.value: ("sabardairy.org/for-our-milk-producers",),
}
SUPPORTED_SCHEME_UNIONS = frozenset(SUPPORTED_UNION_SOURCE_KEYS)


def _build_prefixed_key(namespace: str, key: str) -> str:
    normalized_prefix = settings.redis_key_prefix.rstrip(":-")
    if normalized_prefix:
        return f"{normalized_prefix}:{namespace}:{key}"
    return f"{namespace}:{key}"


def build_scheme_cache_key(source_key: str) -> str:
    return _build_prefixed_key(SCHEME_CACHE_NAMESPACE, source_key)


def get_source_keys_for_union(union_name: str) -> tuple[str, ...]:
    return SUPPORTED_UNION_SOURCE_KEYS.get(canonical_union_name(union_name), ())


async def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as redis
    except ModuleNotFoundError as exc:
        raise SchemeDependencyError("redis is not installed") from exc

    try:
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
    except Exception as exc:
        raise SchemeCacheError("failed to initialize Redis client") from exc
    return _redis_client


async def get_cached_source_records(source_key: str, redis_client=None) -> list[dict[str, Any]]:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    try:
        cached = await client.get(cache_key)
    except Exception as exc:
        raise SchemeCacheError(f"failed to read scheme cache for {source_key}") from exc
    if not cached:
        return []
    try:
        parsed = json.loads(cached)
    except json.JSONDecodeError:
        logger.warning("Invalid scheme cache payload source_key=%s cache_key=%s", source_key, cache_key)
        return []
    if not isinstance(parsed, list):
        logger.warning("Unexpected scheme cache payload type source_key=%s payload_type=%s", source_key, type(parsed).__name__)
        return []
    return parsed


async def get_cached_scheme_records_for_union(union_name: str, redis_client=None) -> list[dict[str, Any]]:
    normalized_union_name = canonical_union_name(union_name)
    source_keys = get_source_keys_for_union(normalized_union_name)
    if not source_keys:
        return []

    records: list[dict[str, Any]] = []
    for source_key in source_keys:
        source_records = await get_cached_source_records(source_key, redis_client=redis_client)
        records.extend(record for record in source_records if record.get("union_name") == normalized_union_name)
    return records
