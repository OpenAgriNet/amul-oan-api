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
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.core.cache import cache, redis_client, build_cache_key
from app.config import settings
from app.observability import start_observation
from app.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from app.models.union import is_ai_call_banned_union
from agents.tools.farmer_animal_backends import (
    GetAITechniciansBySocietyQueryParams,
    get_ai_technicians_by_society_cached,
    fetch_reason,
    normalize_phone,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

FARMER_CACHE_TTL = settings.farmer_cache_retention_seconds  # hard retention in Redis (deletion), default 7d
FARMER_REFRESH_INTERVAL = settings.farmer_refresh_interval_seconds  # soft expiry: refresh a "found" record, default 12h
FARMER_NEGATIVE_REFRESH_INTERVAL = settings.farmer_negative_refresh_interval_seconds  # not_found refreshes sooner, default 2h
FARMER_REFRESH_LOCK_TTL = settings.farmer_refresh_lock_ttl_seconds  # dedupe concurrent refreshes
FARMER_COLD_FETCH_TIMEOUT = settings.farmer_cold_fetch_timeout_seconds  # bounded cold/never-cached miss
FARMER_REFRESH_QUEUE_BATCH_SIZE = settings.farmer_refresh_queue_batch_size
FARMER_REFRESH_RETRY_BASE_SECONDS = settings.farmer_refresh_retry_base_seconds
FARMER_REFRESH_RETRY_MAX_SECONDS = settings.farmer_refresh_retry_max_seconds
# Beyond this age a cached record is too stale to serve: the read blocks on a
# bounded API call instead (falls back to the stale record only if that fails).
FARMER_MAX_SERVE_STALE_SECONDS = settings.farmer_max_serve_stale_seconds
FARMER_CACHE_NAMESPACE = "farmer"
FARMER_REFRESH_LOCK_NAMESPACE = "farmer-refresh"
FARMER_REFRESH_QUEUE_NAMESPACE = "farmer-refresh-queue"
FARMER_REFRESH_ATTEMPT_NAMESPACE = "farmer-refresh-attempt"
# Single Redis set holding raw phone numbers awaiting a background refresh.
FARMER_REFRESH_QUEUE_KEY = build_cache_key("pending", namespace=FARMER_REFRESH_QUEUE_NAMESPACE)


def _normalize_cache_phone(phone: str) -> str:
    """Canonical phone for cache keys and queue membership."""
    if not phone:
        return phone
    normalized = normalize_phone(phone)
    return normalized or phone


def _cache_key(phone: str) -> str:
    """Build cache key from phone number hash."""
    return hashlib.sha256(phone.encode()).hexdigest()


def _phone_log_hash(phone: str) -> str:
    return _cache_key(phone)[:8]


def _refresh_lock_key(phone: str) -> str:
    return build_cache_key(_cache_key(phone), namespace=FARMER_REFRESH_LOCK_NAMESPACE)


def _refresh_attempt_key(phone: str) -> str:
    return build_cache_key(_cache_key(phone), namespace=FARMER_REFRESH_ATTEMPT_NAMESPACE)


def _is_authoritative_envelope(envelope: Optional[FarmerDataEnvelope]) -> bool:
    """Only ``found`` and ``not_found`` may be served to downstream consumers."""
    if envelope is None:
        return False
    return envelope.lookupStatus in {"found", "not_found"}


def _compute_backoff_seconds(attempt_count: int) -> int:
    """Exponential backoff capped at FARMER_REFRESH_RETRY_MAX_SECONDS."""
    if attempt_count <= 0:
        return FARMER_REFRESH_RETRY_BASE_SECONDS
    delay = FARMER_REFRESH_RETRY_BASE_SECONDS * (2 ** (attempt_count - 1))
    return min(delay, FARMER_REFRESH_RETRY_MAX_SECONDS)


async def _get_refresh_attempt_state(phone: str) -> dict[str, Any]:
    try:
        raw = await redis_client.get(_refresh_attempt_key(phone))
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode()
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    except Exception as e:
        logger.debug("Failed to read farmer refresh attempt state: %s", e)
    return {}


async def _set_refresh_attempt_state(phone: str, state: dict[str, Any]) -> None:
    try:
        ttl = max(FARMER_REFRESH_RETRY_MAX_SECONDS * 2, FARMER_CACHE_TTL)
        await redis_client.set(
            _refresh_attempt_key(phone),
            json.dumps(state),
            ex=ttl,
        )
    except Exception as e:
        logger.warning("Failed to write farmer refresh attempt state: %s", e)


async def _clear_refresh_attempt_state(phone: str) -> None:
    try:
        await redis_client.delete(_refresh_attempt_key(phone))
    except Exception:
        pass


async def _enqueue_backoff_active(phone: str) -> bool:
    """True when we should skip enqueue because next_retry_at is still in the future."""
    state = await _get_refresh_attempt_state(phone)
    next_retry_at = state.get("next_retry_at")
    if not next_retry_at:
        return False
    try:
        retry_after = datetime.fromisoformat(str(next_retry_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(timezone.utc) < retry_after.astimezone(timezone.utc)


async def _record_refresh_enqueue_backoff(phone: str) -> None:
    """Schedule the next allowed enqueue after a successful queue insert."""
    state = await _get_refresh_attempt_state(phone)
    attempt_count = int(state.get("attempt_count") or 0)
    delay = _compute_backoff_seconds(max(attempt_count, 1))
    next_retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
    await _set_refresh_attempt_state(
        phone,
        {
            "attempt_count": attempt_count,
            "next_retry_at": next_retry_at,
            "last_outcome": state.get("last_outcome"),
        },
    )


async def _record_refresh_failure(phone: str, *, outcome: str = "error") -> None:
    """Increment attempt count after a failed refresh and extend retry window."""
    state = await _get_refresh_attempt_state(phone)
    attempt_count = int(state.get("attempt_count") or 0) + 1
    delay = _compute_backoff_seconds(attempt_count)
    next_retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
    await _set_refresh_attempt_state(
        phone,
        {
            "attempt_count": attempt_count,
            "next_retry_at": next_retry_at,
            "last_outcome": outcome,
        },
    )


async def _normalize_for_consumer(
    phone: str,
    envelope: Optional[FarmerDataEnvelope],
) -> Optional[FarmerDataEnvelope]:
    """Return only authoritative envelopes; non-authoritative rows become None."""
    if envelope is None:
        return None
    if _is_authoritative_envelope(envelope):
        return envelope
    logger.info(
        "Farmer cache read: non_authoritative phone_hash=%s lookup_status=%s",
        _phone_log_hash(phone),
        envelope.lookupStatus,
    )
    await enqueue_farmer_refresh(phone)
    return None


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
    if envelope.lookupStatus == "unknown":
        return True, "non_authoritative", None
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


from agents.tools.farmer import FarmerFetchOutcome, fetch_farmer_info_with_outcome


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
    phone = _normalize_cache_phone(phone)
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

        records, outcome = await fetch_farmer_info_with_outcome(phone)
        if outcome == FarmerFetchOutcome.FOUND and records:
            envelope = FarmerDataEnvelope.from_records(records, source="api", lookup_status="found")
            envelope.aiTechnicians = await _fetch_ai_technicians(records)
            await set_cached_farmer_data(phone, envelope)
            await _clear_refresh_attempt_state(phone)
            return envelope

        # ERROR or ambiguous empty while upstream cannot distinguish miss vs failure.
        existing = await get_cached_farmer_data(phone)
        if existing is not None and existing.lookupStatus == "found":
            logger.info(
                "Skipping not_found overwrite of good cached farmer data (phone hash %s...)",
                _cache_key(phone)[:8],
            )
            if exceeds_max_serve_stale(existing):
                await _restamp_kept_record(phone, existing)
            await _record_refresh_failure(phone, outcome="error")
            return existing

        await _record_refresh_failure(phone, outcome="error")
        return None
    except Exception as e:
        logger.warning("Farmer refresh failed for phone hash %s...: %s", _cache_key(phone)[:8], e)
        await _record_refresh_failure(phone, outcome="exception")
        return None
    finally:
        if acquired:
            try:
                await redis_client.delete(lock_key)
            except Exception:
                pass


async def enqueue_farmer_refresh(phone: str) -> None:
    """Queue a phone for background refresh (stale-while-revalidate).

    Pushes the raw phone onto a Redis set so a dedicated worker can refresh it
    off the request path. Outcome-based exponential backoff prevents hot phones
    from being re-enqueued every read when upstream keeps failing.
    """
    phone = _normalize_cache_phone(phone)
    if not phone:
        return
    try:
        if await _enqueue_backoff_active(phone):
            logger.debug(
                "Skipping farmer refresh enqueue within backoff window for phone hash %s...",
                _phone_log_hash(phone),
            )
            return
        await redis_client.sadd(FARMER_REFRESH_QUEUE_KEY, phone)
        await _record_refresh_enqueue_backoff(phone)
    except Exception as e:
        logger.warning("Failed to enqueue farmer refresh: %s", e)


async def _revalidate_stale_read(
    phone: str,
    envelope: FarmerDataEnvelope,
    *,
    allow_block_fetch: bool,
) -> FarmerDataEnvelope:
    """SWR read policy for a cached envelope: return immediately; refresh off-path
    when soft-stale. When ``allow_block_fetch`` is True and the record exceeds
    max-serve-stale, block briefly on a bounded upstream fetch before falling
    back to the stale envelope."""
    phone_hash = _phone_log_hash(phone)

    if not envelope.stale:
        logger.info(
            "Farmer cache read: cache_hit_fresh phone_hash=%s lookup_status=%s",
            phone_hash,
            envelope.lookupStatus,
        )
        return envelope

    if allow_block_fetch and exceeds_max_serve_stale(envelope):
        logger.info(
            "Farmer cache read: cold_fetch phone_hash=%s lookup_status=%s reason=max_serve_stale",
            phone_hash,
            envelope.lookupStatus,
        )
        refreshed = await refresh_farmer_data_bounded(phone)
        if refreshed is not None:
            logger.info(
                "Farmer cache read: cold_fetch_success phone_hash=%s lookup_status=%s",
                phone_hash,
                refreshed.lookupStatus,
            )
            return refreshed
        logger.warning(
            "Farmer cache read: cold_fetch_fallback_stale phone_hash=%s lookup_status=%s",
            phone_hash,
            envelope.lookupStatus,
        )
        await enqueue_farmer_refresh(phone)
        return envelope

    await enqueue_farmer_refresh(phone)
    logger.info(
        "Farmer cache read: cache_hit_stale_enqueued phone_hash=%s lookup_status=%s stale_reason=%s",
        phone_hash,
        envelope.lookupStatus,
        envelope.staleReason,
    )
    return envelope


async def get_farmer_data_cached_only(phone: str) -> Optional[FarmerDataEnvelope]:
    """Read farmer context from Redis only; never block on upstream APIs."""
    phone = _normalize_cache_phone(phone)
    cached = await get_cached_farmer_data(phone)
    if cached is None:
        return None
    return await _revalidate_stale_read(phone, cached, allow_block_fetch=False)


def should_refresh_farmer_data(envelope: Optional[FarmerDataEnvelope]) -> bool:
    if envelope is None:
        return True
    return envelope.stale


async def get_or_fetch_farmer_data(phone: str) -> Optional[FarmerDataEnvelope]:
    """Cache-first retrieval used by the /user endpoint and loan tool.

    Fresh cache hits return immediately. Soft-stale hits are served immediately
    and enqueued for background refresh. Records older than max-serve-stale
    trigger a bounded blocking fetch; cold misses do a full refresh.

    Only authoritative envelopes (``found``, ``not_found``) are returned.
    Legacy/non-authoritative ``unknown`` rows normalize to ``None``.
    """
    phone = _normalize_cache_phone(phone)
    cached = await get_cached_farmer_data(phone)
    if cached:
        result = await _revalidate_stale_read(phone, cached, allow_block_fetch=True)
        return await _normalize_for_consumer(phone, result)

    logger.info(
        "Farmer cache read: cold_fetch phone_hash=%s reason=cache_miss",
        _phone_log_hash(phone),
    )
    refreshed = await refresh_farmer_data_bounded(phone)
    return await _normalize_for_consumer(phone, refreshed)


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


async def drain_farmer_refresh_queue_once(batch: int = FARMER_REFRESH_QUEUE_BATCH_SIZE) -> int:
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

    # Keep response shape per farmer while deduping upstream lookups by society:
    # one cache/API lookup per unique (union_code, society_code), then fan out.
    eligible_rows: list[tuple[dict, str, str]] = []
    for record in records:
        data = record.model_dump()
        union_name = data.get("unionName") or data.get("union_name")
        if is_ai_call_banned_union(union_name if isinstance(union_name, str) else None):
            logger.info(
                "Skipping AI technician lookup; union is banned from AI-call booking union=%s farmer=%s",
                union_name,
                data.get("farmerName"),
            )
            continue
        union_code = data.get("unionCode") or data.get("union_code")
        society_code = data.get("societyCode") or data.get("society_code")
        if not union_code or not society_code:
            continue
        eligible_rows.append((data, str(union_code), str(society_code)))

    if not eligible_rows:
        return []

    unique_pairs = sorted({(union_code, society_code) for _, union_code, society_code in eligible_rows})

    async def _lookup_pair(union_code: str, society_code: str) -> tuple[tuple[str, str], Optional[list]]:
        try:
            technicians = await get_ai_technicians_by_society_cached(
                GetAITechniciansBySocietyQueryParams(
                    unionCode=union_code,
                    societyCode=society_code,
                ),
                token,
            )
            return (union_code, society_code), technicians
        except Exception as e:
            logger.warning(
                "AI technician lookup failed for union=%s society=%s: %s",
                union_code,
                society_code,
                e,
            )
            return (union_code, society_code), None

    pair_results = await asyncio.gather(
        *(_lookup_pair(union_code, society_code) for union_code, society_code in unique_pairs)
    )
    pair_to_technicians: dict[tuple[str, str], Optional[list]] = dict(pair_results)

    groups: list[dict] = []
    for data, union_code, society_code in eligible_rows:
        technicians = pair_to_technicians.get((union_code, society_code))
        lookup_failed = technicians is None
        groups.append(
            {
                "farmerName": data.get("farmerName"),
                "farmerCode": data.get("farmerCode"),
                "societyName": data.get("societyName"),
                "societyCode": society_code,
                "unionCode": union_code,
                # Preserve fetch failure distinctly from a true empty list.
                "technicians": (
                    None
                    if lookup_failed
                    else [technician.model_dump() for technician in technicians]
                ),
                "techniciansLookupFailed": lookup_failed,
            }
        )
    return groups
