"""Backward-compatible cache exports. Prefer using app.redis.* modules directly."""

from app.redis.cache import (
    build_api_cache_key,
    cache,
    get_cached_api_response,
    set_cached_api_response,
)
