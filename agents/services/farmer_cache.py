"""
Cache layer for farmer data fetched from PashuGPT APIs.
Uses Redis (via aiocache) with 24h TTL. Cache key is a SHA-256 hash of the
normalized phone number so raw phone numbers don't appear as Redis keys.

Both backends share the same Redis instance and prefix (sva-cache-), so a
farmer cached by the chat service is available to voice and vice versa.
"""
import hashlib
from typing import Optional

from app.core.cache import cache
from agents.models.farmer import FarmerDataEnvelope
from helpers.utils import get_logger

logger = get_logger(__name__)

FARMER_CACHE_TTL = 60 * 60 * 24 * 17  # 17 days
FARMER_CACHE_NAMESPACE = "farmer"


def _cache_key(phone: str) -> str:
    """Build cache key from phone number hash."""
    return hashlib.sha256(phone.encode()).hexdigest()


async def get_cached_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """Retrieve cached farmer data for a phone number."""
    key = _cache_key(phone)
    try:
        raw = await cache.get(key, namespace=FARMER_CACHE_NAMESPACE)
        if raw and isinstance(raw, dict):
            envelope = FarmerDataEnvelope.model_validate(raw)
            envelope.source = "cache"
            return envelope
    except Exception as e:
        logger.warning(f"Failed to read farmer cache for phone hash {key[:8]}...: {e}")
    return None


async def set_cached_farmer_data(phone: str, data: FarmerDataEnvelope) -> None:
    """Store farmer data in cache."""
    key = _cache_key(phone)
    try:
        await cache.set(key, data.model_dump(), ttl=FARMER_CACHE_TTL, namespace=FARMER_CACHE_NAMESPACE)
        logger.debug(f"Cached farmer data for phone hash {key[:8]}... ({len(data.farmers)} records)")
    except Exception as e:
        logger.warning(f"Failed to write farmer cache: {e}")


from agents.tools.farmer import get_farmer_data_by_mobile


async def get_or_fetch_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """Cache-first farmer data retrieval. Check Redis, else fetch from APIs and cache."""
    cached = await get_cached_farmer_data(phone)
    if cached:
        return cached

    records = await get_farmer_data_by_mobile(phone)
    if records:
        envelope = FarmerDataEnvelope.from_records(records, source="api")
        await set_cached_farmer_data(phone, envelope)
        return envelope
    return None
