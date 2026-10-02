"""
Cache layer for farmer data fetched from PashuGPT APIs.

Voice reads farmer context from Redis only. Freshness is controlled separately
from key expiry:
- soft refresh interval: 12h for "found", 2h for "not_found"
- cache retention (hard delete): 7d

This lets the request path return cached data immediately, mark it stale in the
read result, and schedule a background refresh (via a Redis queue drained by a
worker) without blocking the caller.
"""
import asyncio
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.cache import cache, redis_client, build_cache_key
from app.config import settings
from app.voice.models.union import is_ai_call_banned_union
from app.observability import start_observation
from agents.voice.models.farmer import AnimalRecord, FarmerDataEnvelope, FarmerRecord
from agents.voice.tools.farmer_animal_backends import (
    BackendUnavailableError,
    GetAITechniciansBySocietyQueryParams,
    get_ai_technicians_by_society_api,
    fetch_animal_amulpashudhan,
    fetch_animal_herdman,
    merge_animal_data,
    normalize_tag,
    fetch_reason,
    current_fetch_reason,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

FARMER_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days hard retention in Redis (deletion)
FARMER_REFRESH_INTERVAL = 60 * 60 * 12  # soft expiry: refresh a "found" record after 12h
FARMER_NEGATIVE_REFRESH_INTERVAL = 60 * 60 * 2  # not_found refreshes sooner (caller may newly register)
FARMER_REFRESH_LOCK_TTL = 60 * 5  # dedupe concurrent refreshes for 5 minutes
FARMER_COLD_FETCH_TIMEOUT = settings.farmer_cold_fetch_timeout  # bounded cold-miss fetch
# Beyond this age a cached record is too stale to serve: the read blocks on a
# bounded API call instead (falls back to the stale record only if that fails).
FARMER_MAX_SERVE_STALE_SECONDS = settings.farmer_max_serve_stale_seconds
# Cap concurrent per-animal API fetches during a background enrichment so a large
# herd (or a multi-record phone) can't fan out into hundreds of simultaneous
# outbound calls (socket exhaustion / 429s).
FARMER_ANIMAL_FETCH_CONCURRENCY = 8
FARMER_CACHE_NAMESPACE = "farmer"
FARMER_REFRESH_LOCK_NAMESPACE = "farmer-refresh"
FARMER_INFLIGHT_NAMESPACE = "farmer-inflight"
FARMER_INFLIGHT_MARKER_TTL = settings.farmer_inflight_marker_ttl
# Redis is local; anything slower means it is degraded. Bound the marker calls so
# a Redis blip can't add its own client-level retry time to a voice turn.
FARMER_INFLIGHT_REDIS_TIMEOUT = 0.25
FARMER_REFRESH_QUEUE_NAMESPACE = "farmer-refresh-queue"
# Single Redis set holding raw phone numbers awaiting a background refresh.
FARMER_REFRESH_QUEUE_KEY = build_cache_key("pending", namespace=FARMER_REFRESH_QUEUE_NAMESPACE)


def _cache_key(phone: str) -> str:
    """Build cache key from phone number hash."""
    return hashlib.sha256(phone.encode()).hexdigest()


def _record_tags(record: FarmerRecord) -> list[str]:
    """Raw (unmasked) animal tags from a farmer record."""
    raw = record.tagNumbers or record.tagNo or ""
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _record_needs_animals(record: FarmerRecord) -> bool:
    """True when a farmer has tags but no per-animal records cached yet — e.g. an
    envelope written before animal enrichment existed. Used to force one
    background refresh that backfills the breeding/AI history."""
    return bool(_record_tags(record)) and not record.animals


def _has_failed_technician_lookup(envelope: FarmerDataEnvelope) -> bool:
    """True when any cached technician group came from a FAILED lookup rather
    than a society that genuinely has no technicians.

    Envelopes written before the lookupFailed flag existed omit the key; those
    are treated as successful so old cache entries are not refreshed forever.
    """
    return any(group.get("lookupFailed") for group in (envelope.aiTechnicians or []))


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
            elif _has_failed_technician_lookup(envelope):
                # A failed lookup was cached as an empty technician list. The
                # key IS present, so the check above cannot catch it and the
                # envelope is not stale by age — without this the blip would
                # persist for the full TTL and voice would keep telling callers
                # no technicians are available.
                envelope.stale = True
                envelope.staleReason = "ai_technician_lookup_failed"
                envelope.refreshAfter = datetime.now(timezone.utc).isoformat()
            elif any(_record_needs_animals(f) for f in envelope.farmers):
                # Backfill per-animal records (incl. AI history) for envelopes
                # cached before animal enrichment existed.
                envelope.stale = True
                envelope.staleReason = "missing_animals"
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


from agents.voice.tools.farmer import fetch_farmer_info_raw


async def _restamp_kept_record(phone: str, envelope: FarmerDataEnvelope) -> None:
    """When don't-downgrade keeps a 'found' record on an empty upstream, refresh
    its fetchedAt so reads stop block-fetching it every turn — while PRESERVING the
    remaining 7d Redis TTL so a genuinely removed farmer still expires on schedule."""
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
    enqueue_backfill_after_unlock = False
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

        upstream_failed = False
        try:
            records = await fetch_farmer_info_raw(phone)
        except BackendUnavailableError as e:
            # Upstream is down; this is not "the farmer does not exist". Fall
            # through to the empty branch, which must not write a not_found.
            upstream_failed = True
            records = None
            logger.warning(
                "Farmer lookup upstream unavailable for phone hash %s...: %s",
                _cache_key(phone)[:8],
                e,
            )
        if records:
            envelope = FarmerDataEnvelope.from_records(records, source="api", lookup_status="found")
            envelope.aiTechnicians = await _fetch_ai_technicians(records)
            # Per-animal enrichment (breeding/AI history) is fetched ONLY in the
            # background worker — it makes N extra API calls and must never be on
            # a voice turn. A cold/request-path refresh caches farmer-level data
            # immediately, then enqueues a background refresh to backfill the
            # animal records so they are present on the next turn (rather than
            # waiting for a later read to notice "missing_animals" staleness).
            if current_fetch_reason() == "background_refresh":
                await _enrich_records_with_animals(envelope.farmers)
            await set_cached_farmer_data(phone, envelope)
            # Defer the backfill enqueue until AFTER our refresh lock is released
            # (see finally): enqueuing while still holding the lock lets a worker
            # spop the phone, hit the lock-busy branch, and return the
            # missing_animals envelope without enriching — dropping the job.
            if current_fetch_reason() != "background_refresh" and any(
                _record_needs_animals(f) for f in envelope.farmers
            ):
                enqueue_backfill_after_unlock = True
            return envelope

        # Upstream returned nothing. Never let a transient empty response wipe
        # known-good data — keep the "found" record regardless of age (it stays
        # stale and is retried). Genuine removal is left to the 7d hard Redis TTL
        # rather than an ambiguous empty response.
        existing = await get_cached_farmer_data(phone)
        if existing is not None and existing.lookupStatus == "found":
            logger.info(
                "Skipping not_found overwrite of good cached farmer data (phone hash %s...)",
                _cache_key(phone)[:8],
            )
            # If it had already aged past the serve ceiling, a persistently-empty
            # upstream would otherwise force a blocking re-fetch on EVERY turn.
            # Restamp fetchedAt so reads serve it without blocking, while
            # PRESERVING the 7d hard TTL so a genuinely removed farmer still expires.
            if exceeds_max_serve_stale(existing):
                await _restamp_kept_record(phone, existing)
            return existing

        if upstream_failed:
            # Nothing known-good to keep, and nothing trustworthy to write.
            # Leave the cache empty so the next turn retries, rather than pinning
            # a 2h not_found (FARMER_NEGATIVE_REFRESH_INTERVAL) on a caller whose
            # lookup merely failed. Returning None reads as "unresolved"
            # downstream, which tells the caller to try again rather than that
            # their number is unregistered. See issue #282 root cause B.
            return None

        # Upstream answered and has no record for this mobile. That is a fact,
        # and it is the only empty result we are willing to cache.
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
        # Lock is now released — safe to enqueue the backfill: a worker that
        # picks it up will acquire the lock cleanly and run enrichment.
        if enqueue_backfill_after_unlock:
            await enqueue_farmer_refresh(phone)


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


def _inflight_key(phone: str) -> str:
    return build_cache_key(_cache_key(phone), namespace=FARMER_INFLIGHT_NAMESPACE)


async def is_fetch_inflight(phone: str) -> bool:
    """True when a recent cold fetch was cancelled and a worker is still on it.

    Fails open: an unreadable marker must not make every caller look unresolved.
    """
    try:
        return bool(
            await asyncio.wait_for(
                redis_client.exists(_inflight_key(phone)),
                timeout=FARMER_INFLIGHT_REDIS_TIMEOUT,
            )
        )
    except Exception:
        return False


async def _mark_fetch_inflight(phone: str) -> None:
    try:
        await asyncio.wait_for(
            redis_client.set(_inflight_key(phone), "1", ex=FARMER_INFLIGHT_MARKER_TTL),
            timeout=FARMER_INFLIGHT_REDIS_TIMEOUT,
        )
    except Exception as e:
        logger.warning("Failed to set farmer in-flight marker: %s", e)


async def refresh_farmer_data_bounded(
    phone: str, timeout: float = FARMER_COLD_FETCH_TIMEOUT
) -> Optional[FarmerDataEnvelope]:
    """Blocking refresh with a hard timeout, for a cold/never-cached miss.

    On timeout we defer to the background worker rather than hanging the turn,
    and leave a marker so the next turn doesn't block on the same fetch again.
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
        await _mark_fetch_inflight(phone)
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


async def _fetch_one_animal(tag: str, token1: Optional[str], token3: Optional[str]) -> Optional[AnimalRecord]:
    """Fetch + merge a single animal (amulpashudhan primary, herdman fallback)."""
    norm = normalize_tag(tag)
    if not norm:
        return None
    primary = None
    fallback = None
    if token1:
        try:
            primary = await fetch_animal_amulpashudhan(norm, token1)
        except Exception as e:
            logger.warning("amulpashudhan animal fetch failed for tag %s: %s", norm, e)
    if token3:
        try:
            fallback = await fetch_animal_herdman(norm, token3)
        except Exception as e:
            logger.warning("herdman animal fetch failed for tag %s: %s", norm, e)
    merged = merge_animal_data(primary, fallback)
    if not merged:
        return None
    try:
        return AnimalRecord.model_validate(merged)
    except Exception as e:
        logger.warning("Failed to validate animal record for tag %s: %s", norm, e)
        return None


async def _enrich_records_with_animals(records: list[FarmerRecord]) -> None:
    """Populate each farmer record's `animals` list with per-animal data
    (incl. lastBreedingActivity = AI date + bull id). Runs only on the background
    refresh path — never on a voice turn. Mutates `records` in place; best-effort."""
    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        return

    # One flat gather across all farmers × tags so the worker fetches the whole
    # herd concurrently rather than farmer-by-farmer — bounded by a semaphore so
    # a large herd can't open hundreds of sockets at once.
    jobs: list[tuple[FarmerRecord, str]] = []
    for record in records:
        for tag in _record_tags(record):
            jobs.append((record, tag))
    if not jobs:
        return

    sem = asyncio.Semaphore(FARMER_ANIMAL_FETCH_CONCURRENCY)

    async def _bounded(tag: str) -> Optional[AnimalRecord]:
        async with sem:
            return await _fetch_one_animal(tag, token1, token3)

    results = await asyncio.gather(
        *(_bounded(tag) for _, tag in jobs),
        return_exceptions=True,
    )
    by_record: dict[int, list[AnimalRecord]] = {}
    for (record, _tag), result in zip(jobs, results):
        if isinstance(result, AnimalRecord):
            by_record.setdefault(id(record), []).append(result)
    for record in records:
        animals = by_record.get(id(record))
        if animals:
            record.animals = animals


async def _fetch_ai_technicians(records: list[FarmerRecord]) -> list[dict]:
    token = os.getenv("PASHUGPT_TOKEN")
    if not token or not records:
        return []

    async def _fetch_for_farmer(record: FarmerRecord) -> Optional[dict]:
        data = record.model_dump()
        union_name = data.get("unionName") or data.get("union_name")
        if is_ai_call_banned_union(union_name if isinstance(union_name, str) else None):
            logger.info(
                "Skipping AI technician lookup; union is banned from AI-call booking union=%s farmer=%s",
                union_name,
                data.get("farmerName"),
            )
            return None
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

        # get_ai_technicians_by_society_api returns None when the lookup FAILED
        # and [] when the society genuinely has no technicians. Both used to be
        # flattened to [], so a transient upstream blip was cached as a
        # confident "no technicians" for the life of the envelope. Record the
        # difference so get_cached_farmer_data can mark the envelope stale and
        # let the background worker retry.
        return {
            "farmerName": data.get("farmerName"),
            "farmerCode": data.get("farmerCode"),
            "societyName": data.get("societyName"),
            "societyCode": str(society_code),
            "unionCode": str(union_code),
            "technicians": [technician.model_dump() for technician in (technicians or [])],
            "lookupFailed": technicians is None,
        }

    groups = await asyncio.gather(*[_fetch_for_farmer(record) for record in records])
    return [group for group in groups if group is not None]
