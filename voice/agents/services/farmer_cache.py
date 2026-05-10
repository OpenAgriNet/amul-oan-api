"""
Cache layer for farmer data fetched from PashuGPT APIs.

Voice reads farmer context from Redis only. Freshness is controlled separately
from key expiry:
- refresh interval: 24h
- cache retention: 7d

This lets the request path return cached data immediately, mark it stale in the
read result, and schedule a background refresh without blocking the caller.
"""
import asyncio
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from voice.app.core.cache import cache, redis_client, build_cache_key
from voice.agents.models.farmer import FarmerDataEnvelope, FarmerRecord
from voice.agents.tools.farmer_animal_backends import (
    GetAITechniciansBySocietyQueryParams,
    get_ai_technicians_by_society_api,
)
from voice.helpers.utils import get_logger

logger = get_logger(__name__)

FARMER_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days retention in Redis
FARMER_REFRESH_INTERVAL = 60 * 60 * 24  # refresh once a day
FARMER_REFRESH_LOCK_TTL = 60 * 5  # dedupe concurrent refreshes for 5 minutes
FARMER_CACHE_NAMESPACE = "farmer"
FARMER_REFRESH_LOCK_NAMESPACE = "farmer-refresh"


def _cache_key(phone: str) -> str:
    """Build cache key from phone number hash."""
    return hashlib.sha256(phone.encode()).hexdigest()


def _refresh_lock_key(phone: str) -> str:
    return build_cache_key(_cache_key(phone), namespace=FARMER_REFRESH_LOCK_NAMESPACE)


def _compute_freshness(envelope: FarmerDataEnvelope) -> tuple[bool, Optional[str], Optional[str]]:
    if not envelope.fetchedAt:
        return True, "missing_fetched_at", None
    try:
        fetched_at = datetime.fromisoformat(envelope.fetchedAt.replace("Z", "+00:00"))
    except ValueError:
        return True, "invalid_fetched_at", None

    refresh_after = fetched_at + timedelta(seconds=FARMER_REFRESH_INTERVAL)
    refresh_after_iso = refresh_after.astimezone(timezone.utc).isoformat()
    is_stale = datetime.now(timezone.utc) >= refresh_after.astimezone(timezone.utc)
    return is_stale, ("expired" if is_stale else None), refresh_after_iso


async def get_cached_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """Retrieve cached farmer data for a phone number."""
    key = _cache_key(phone)
    try:
        raw = await cache.get(key, namespace=FARMER_CACHE_NAMESPACE)
        if raw and isinstance(raw, dict):
            envelope = FarmerDataEnvelope.model_validate(raw)
            envelope.source = "cache"
            envelope.lookupStatus = envelope.lookupStatus or ("found" if envelope.farmers else "not_found")
            envelope.stale, envelope.staleReason, envelope.refreshAfter = _compute_freshness(envelope)
            if envelope.farmers and "aiTechnicians" not in raw:
                envelope.stale = True
                envelope.staleReason = "missing_ai_technicians"
                envelope.refreshAfter = datetime.now(timezone.utc).isoformat()
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


from voice.agents.tools.farmer import fetch_farmer_info_raw


async def refresh_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """
    Refresh farmer data from upstream APIs and update Redis.
    Returns the refreshed envelope or None on refresh failure.
    """
    lock_key = _refresh_lock_key(phone)
    acquired = False
    try:
        acquired = await redis_client.set(lock_key, "1", ex=FARMER_REFRESH_LOCK_TTL, nx=True)
        if not acquired:
            logger.debug("Farmer refresh already in flight for phone hash %s...", _cache_key(phone)[:8])
            return None

        records = await fetch_farmer_info_raw(phone)
        if records:
            envelope = FarmerDataEnvelope.from_records(records, source="api", lookup_status="found")
            envelope.aiTechnicians = await _fetch_ai_technicians(records)
        else:
            envelope = FarmerDataEnvelope.not_found(source="api")
        await set_cached_farmer_data(phone, envelope)
        return envelope
    except Exception as e:
        logger.warning("Farmer refresh failed for phone hash %s...: %s", _cache_key(phone)[:8], e)
        return None
    finally:
        if acquired:
            try:
                await redis_client.delete(lock_key)
            except Exception:
                pass


async def get_farmer_data_cached_only(phone: str) -> Optional[FarmerDataEnvelope]:
    """Read farmer context from Redis only; never block on upstream APIs."""
    return await get_cached_farmer_data(phone)


def should_refresh_farmer_data(envelope: Optional[FarmerDataEnvelope]) -> bool:
    if envelope is None:
        return True
    return envelope.stale


async def get_or_fetch_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """Legacy cache-first retrieval. Prefer cached-only reads on the voice path."""
    cached = await get_cached_farmer_data(phone)
    if cached:
        return cached

    return await refresh_farmer_data(phone)


async def _fetch_ai_technicians(records: list[FarmerRecord]) -> list[dict]:
    token = os.getenv("PASHUGPT_TOKEN")
    if not token or not records:
        return []

    async def _fetch_for_farmer(record: FarmerRecord) -> Optional[dict]:
        data = record.model_dump()
        union_code = data.get("unionCode") or data.get("union_code")
        society_code = data.get("societyCode") or data.get("society_code")
        if not union_code or not society_code:
            return None

        try:
            technicians = await get_ai_technicians_by_society_api(
                GetAITechniciansBySocietyQueryParams(
                    unionCode=str(union_code),
                    societyCode=str(society_code),
                ),
                token,
            )
        except Exception as e:
            logger.warning(
                "AI technician lookup failed for farmer=%s union=%s society=%s: %s",
                data.get("farmerName"),
                union_code,
                society_code,
                e,
            )
            technicians = None

        return {
            "farmerName": data.get("farmerName"),
            "farmerCode": data.get("farmerCode"),
            "societyName": data.get("societyName"),
            "societyCode": str(society_code),
            "unionCode": str(union_code),
            "technicians": [technician.model_dump() for technician in (technicians or [])],
        }

    groups = await asyncio.gather(*[_fetch_for_farmer(record) for record in records])
    return [group for group in groups if group is not None]
