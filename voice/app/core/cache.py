"""
Core cache instance configuration using Redis and aiocache.

This module provides the cache instance that other parts of the application can use.
Uses enhanced Redis configuration with connection pooling and timeouts.
"""
from aiocache import Cache
from aiocache.serializers import JsonSerializer
from redis.asyncio import Redis
from voice.app.config import settings
from voice.helpers.utils import get_logger

logger = get_logger(__name__)

# Configure the cache instance with enhanced settings from Django
cache = Cache(
    Cache.REDIS,
    endpoint=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    serializer=JsonSerializer(),
    ttl=settings.default_cache_ttl,
    # Enhanced connection settings
    timeout=settings.redis_socket_timeout,
    pool_max_size=settings.redis_max_connections,
    # Add key prefix support
    key_builder=lambda key, namespace: f"{settings.redis_key_prefix}{namespace}:{key}" if namespace else f"{settings.redis_key_prefix}{key}",
)

redis_client = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    socket_connect_timeout=settings.redis_socket_connect_timeout,
    socket_timeout=settings.redis_socket_timeout,
    max_connections=settings.redis_max_connections,
    retry_on_timeout=settings.redis_retry_on_timeout,
    decode_responses=True,
)


def build_cache_key(key: str, namespace: str | None = None) -> str:
    if namespace:
        return f"{settings.redis_key_prefix}{namespace}:{key}"
    return f"{settings.redis_key_prefix}{key}"

logger.info(
    f"Cache configured with Redis at {settings.redis_host}:{settings.redis_port} "
    f"(DB: {settings.redis_db}, Prefix: {settings.redis_key_prefix}, "
    f"Max Connections: {settings.redis_max_connections})"
)
