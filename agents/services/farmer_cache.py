"""
Cache layer for farmer data fetched from PashuGPT APIs (unified chat + voice).

Stale-while-revalidate: reads return cached data immediately and mark it stale;
a background worker refreshes off the request path so slow/unreliable upstream
APIs never block a turn. Freshness is tracked separately from Redis key expiry:
- soft refresh interval: 12h for "found", 2h for "not_found" (env-tunable)
- cache retention (hard delete): 7d (env-tunable)

All three timers are config-driven (app.config.settings). chat (/user) and voice
share the same Redis key per phone, so a farmer cached by one is visible to both.

KNOWN LIMITATION (logged, follow-up): a register-then-immediately-call flow can
keep seeing "not_found" for up to the not_found interval (~2h) — the stale
negative-cache entry is served and isn't re-checked until it crosses its refresh
mark and a read enqueues a background refresh. Lowering
FARMER_NEGATIVE_REFRESH_INTERVAL_SECONDS shrinks the window; the proper fix is
active cache-invalidation on registration (registration flow busts the phone's
cache key), which is cross-service and out of scope for the merge.
"""
import asyncio
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.cache import cache, redis_client, build_cache_key
from app.config import settings
from app.observability import start_observation
from agents.models.farmer import FarmerDataEnvelope, FarmerRecord
from agents.tools.farmer_animal_backends import (
    GetAITechniciansBySocietyQueryParams,
    get_ai_technicians_by_society_api,
    fetch_reason,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

FARMER_CACHE_TTL = settings.farmer_cache_retention_seconds  # hard retention in Redis (deletion), default 7d
FARMER_REFRESH_INTERVAL = settings.farmer_refresh_interval_seconds  # soft expiry: refresh a "found" record, default 12h
FARMER_NEGATIVE_REFRESH_INTERVAL = settings.farmer_negative_refresh_interval_seconds  # not_found refreshes sooner, default 2h
FARMER_REFRESH_LOCK_TTL = 60 * 5  # dedupe concurrent refreshes for 5 minutes
FARMER_COLD_FETCH_TIMEOUT = 4.0  # bounded blocking fetch for a cold/never-cached miss (cold ~3.1s observed)
# Beyond this age a cached record is too stale to serve: the read blocks on a
# bounded API call instead (falls back to the stale record only if that fails).
FARMER_MAX_SERVE_STALE_SECONDS = settings.farmer_max_serve_stale_seconds
FARMER_CACHE_NAMESPACE = "farmer"
FARMER_REFRESH_LOCK_NAMESPACE = "farmer-refresh"
FARMER_REFRESH_QUEUE_NAMESPACE = "farmer-refresh-queue"
# Single Redis set holding raw phone numbers awaiting a background refresh.
FARMER_REFRESH_QUEUE_KEY = build_cache_key("pending", namespace=FARMER_REFRESH_QUEUE_NAMESPACE)


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

    interval = (
        FARMER_NEGATIVE_REFRESH_INTERVAL
        if envelope.lookupStatus == "not_found"
        else FARMER_REFRESH_INTERVAL
    )
    refresh_after = fetched_at + timedelta(seconds=interval)
    refresh_after_iso = refresh_after.astimezone(timezone.utc).isoformat()
    is_stale = datetime.now(timezone.utc) >= refresh_after.astimezone(timezone.utc)
    return is_stale, ("expired" if is_stale else None), refresh_after_iso


def _envelope_age_seconds(envelope: Optional[FarmerDataEnvelope]) -> Optional[float]:
    if envelope is None or not envelope.fetchedAt:
        return None
    try:
        fetched_at = datetime.fromisoformat(envelope.fetchedAt.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds()


def exceeds_max_serve_stale(envelope: Optional[FarmerDataEnvelope]) -> bool:
    """True when a cached record is too old to serve and the read should block
    on a fresh API call (e.g. background refresh has been failing). Unknown age
    counts as too stale."""
    age = _envelope_age_seconds(envelope)
    if age is None:
        return True
    return age > FARMER_MAX_SERVE_STALE_SECONDS


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


from agents.tools.farmer import fetch_farmer_info_raw


async def _restamp_kept_record(phone: str, envelope: FarmerDataEnvelope) -> None:
    """When don't-downgrade keeps a 'found' record on an empty upstream, refresh
    its fetchedAt so reads stop block-fetching it every turn — while PRESERVING the
    remaining hard Redis TTL so a genuinely removed farmer still expires on schedule."""
    envelope.fetchedAt = datetime.now(timezone.utc).isoformat()
    key = _cache_key(phone)
    try:
        remaining = await redis_client.ttl(build_cache_key(key, namespace=FARMER_CACHE_NAMESPACE))
        if remaining and remaining > 0:
            await cache.set(key, envelope.model_dump(), ttl=remaining, namespace=FARMER_CACHE_NAMESPACE)
    except Exception as e:
        logger.warning("Failed to restamp kept farmer record (phone hash %s...): %s", key[:8], e)


async def _await_inflight_refresh(
    phone: str, lock_key: str, *, timeout: float, interval: float = 0.1
) -> tuple[Optional[FarmerDataEnvelope], bool]:
    """Poll until the in-flight refresh holding `lock_key` finishes (lock gone),
    bounded by `timeout`. Returns (latest cached value, cleared) — cleared is True
    only if the lock actually released within the window. No cap below `timeout`:
    the request path's outer asyncio.wait_for is the real limiter, and the worker
    passes its own bound — so a slow (~3s) cold-fetch holder is awaited fully
    instead of giving up early and serving stale."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    cleared = False
    while loop.time() < deadline:
        await asyncio.sleep(interval)
        try:
            if not await redis_client.exists(lock_key):
                cleared = True
                break
        except Exception:
            break
    return await get_cached_farmer_data(phone), cleared


async def refresh_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """
    Refresh farmer data from upstream APIs and update Redis.
    Returns the refreshed envelope, or the in-flight refresh's result when the
    lock is busy and clears in time, or None on actual failure / lock-still-busy.
    """
    lock_key = _refresh_lock_key(phone)
    acquired = False
    try:
        acquired = await redis_client.set(lock_key, "1", ex=FARMER_REFRESH_LOCK_TTL, nx=True)
        if not acquired:
            # Another refresh is in-flight. Wait for its result and return that,
            # rather than None — None would make max-serve-stale serve the ancient
            # record and would let the worker drop a queued phone as a no-op.
            logger.debug("Farmer refresh in flight for phone hash %s...; awaiting result", _cache_key(phone)[:8])
            env, cleared = await _await_inflight_refresh(
                phone, lock_key, timeout=FARMER_COLD_FETCH_TIMEOUT
            )
            if cleared:
                return env
            # Holder outlived our wait — re-queue so the refresh isn't lost
            # (covers the worker path) and signal "not done" to the caller.
            await enqueue_farmer_refresh(phone)
            return None

        records = await fetch_farmer_info_raw(phone)
        if records:
            envelope = FarmerDataEnvelope.from_records(records, source="api", lookup_status="found")
            envelope.aiTechnicians = await _fetch_ai_technicians(records)
            await set_cached_farmer_data(phone, envelope)
            return envelope

        # Upstream returned nothing. Never let a transient empty response wipe
        # known-good data — keep the "found" record regardless of age (it stays
        # stale and is retried). We cannot distinguish a genuine "not found" from
        # a transient failure here, so genuine removal is left to the hard Redis
        # TTL rather than an ambiguous empty response. (A confident not_found
        # signal is a follow-up in the provider-interface PR.)
        existing = await get_cached_farmer_data(phone)
        if existing is not None and existing.lookupStatus == "found":
            logger.info(
                "Skipping not_found overwrite of good cached farmer data (phone hash %s...)",
                _cache_key(phone)[:8],
            )
            # If it had already aged past the serve ceiling, a persistently-empty
            # upstream would otherwise force a blocking re-fetch on EVERY turn.
            # Restamp fetchedAt so reads serve it without blocking, while
            # PRESERVING the hard TTL so a genuinely removed farmer still expires.
            if exceeds_max_serve_stale(existing):
                await _restamp_kept_record(phone, existing)
            return existing

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
    """Cache-first retrieval used by the /user endpoint. Returns cached data if
    present, else does a full refresh (raw fetch + AI-technician enrichment)."""
    cached = await get_cached_farmer_data(phone)
    if cached:
        return cached

    return await refresh_farmer_data(phone)


async def enqueue_farmer_refresh(phone: str) -> None:
    """Queue a phone for background refresh (stale-while-revalidate).

    Pushes the raw phone onto a Redis set so a dedicated worker can refresh it
    off the request path. The set dedupes naturally, and refresh_farmer_data
    self-dedupes via its NX lock, so enqueuing the same phone repeatedly is safe.
    """
    if not phone:
        return
    try:
        await redis_client.sadd(FARMER_REFRESH_QUEUE_KEY, phone)
    except Exception as e:
        logger.warning("Failed to enqueue farmer refresh: %s", e)


async def refresh_farmer_data_bounded(
    phone: str, timeout: float = FARMER_COLD_FETCH_TIMEOUT
) -> Optional[FarmerDataEnvelope]:
    """Blocking refresh with a hard timeout, for a cold/never-cached miss.

    On timeout we defer to the background worker rather than hanging the turn:
    the in-flight refresh is cancelled (its NX lock is released in its finally),
    the phone is queued, and the caller proceeds with no farmer data this turn.
    """
    try:
        with fetch_reason("cold_fetch"):
            return await asyncio.wait_for(refresh_farmer_data(phone), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Cold farmer fetch exceeded %.1fs for phone hash %s...; deferring to worker",
            timeout,
            _cache_key(phone)[:8],
        )
        await enqueue_farmer_refresh(phone)
        return None


async def drain_farmer_refresh_queue_once(batch: int = 20) -> int:
    """Pop up to `batch` queued phones and refresh each. Returns count processed."""
    try:
        members = await redis_client.spop(FARMER_REFRESH_QUEUE_KEY, batch)
    except Exception as e:
        logger.warning("Failed to read farmer refresh queue: %s", e)
        return 0
    if not members:
        return 0
    if isinstance(members, (str, bytes)):
        members = [members]
    processed = 0
    for phone in members:
        try:
            # Root span so the nested API-call observations have a parent and
            # are queryable in Langfuse (background refreshes aren't tied to a
            # voice session); fetch_reason tags them as background_refresh.
            with start_observation(
                "farmer_background_refresh",
                input={"phone_hash": _cache_key(phone)[:12]},
                metadata={"reason": "background_refresh"},
            ):
                with fetch_reason("background_refresh"):
                    await refresh_farmer_data(phone)
            processed += 1
        except Exception:
            logger.exception("Background farmer refresh failed for a queued phone")
    return processed


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
