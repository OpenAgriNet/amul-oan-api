"""Centralized Redis settings and TTL constants."""

from app.config import settings


DEFAULT_CACHE_TTL_SECONDS = settings.default_cache_ttl
MESSAGE_HISTORY_TTL_SECONDS = settings.default_cache_ttl
SUGGESTIONS_TTL_SECONDS = settings.suggestions_cache_ttl
FARMER_CACHE_TTL_SECONDS = settings.farmer_animal_api_cache_ttl
FARMER_ANIMAL_API_CACHE_TTL_SECONDS = settings.farmer_animal_api_cache_ttl
SESSION_OWNER_TTL_SECONDS = settings.session_owner_ttl_seconds
SESSION_OWNER_REFRESH_INTERVAL_SECONDS = settings.session_owner_refresh_interval_seconds
AI_CALL_COOLDOWN_TTL_SECONDS = 60 * 30


def key(namespace: str, identifier: str) -> str:
    """Build a normalized key payload before global redis prefix is applied."""
    return f"{namespace}:{identifier}"

