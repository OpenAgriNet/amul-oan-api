"""Redis copy of Amul Pashudhan's verified cattle listings (GetVerifiedAnimalDetails).

One snapshot for every pod: the last good provider payload with stable listing
numbers and its fetch time. A tool request refetches it when it is missing or
older than CATTLE_LISTINGS_REFRESH_SECONDS. The background refresher
(app.tasks.cattle_listings_refresher) refetches on the same interval, but only
while a request came in within CATTLE_LISTINGS_IDLE_SECONDS, so an unused tool
stops calling the provider. A SET NX lock lets one pod fetch at a time; it is
left to expire, so a failing provider is retried at most once per lock TTL.

A failed fetch or the provider's failure sentinel (HTTP 200 with year-one dates
and empty arrays) never replaces the snapshot: the last good copy is served
until its Redis TTL (CATTLE_LISTINGS_MAX_STALE_SECONDS) runs out. The snapshot
holds seller names and mobiles, so it is never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict

from pydantic import ValidationError

from agents.tools.cattle_trade_backend import get_verified_animal_details_api
from agents.tools.models.cattle_trade import (
    CattleListingModel,
    CattleListingsSnapshotModel,
    VerifiedAnimalDetailsModel,
)
from app.config import get_config_value, settings
from app.core.cache import build_cache_key, redis_client
from helpers.utils import get_logger

logger = get_logger(__name__)

NAMESPACE = "cattle-listings"
SNAPSHOT_KEY = build_cache_key("snapshot", namespace=NAMESPACE)
LAST_REQUEST_KEY = build_cache_key("last-request", namespace=NAMESPACE)
LOCK_KEY = build_cache_key("refresh-lock", namespace=NAMESPACE)
# Also the minimum gap between provider calls after a failure.
LOCK_TTL_SECONDS = 60
# A pod without a snapshot waits this long for another pod's fetch.
WAIT_FOR_PEER_SECONDS = 3.0


def is_failure_sentinel(details: VerifiedAnimalDetailsModel) -> bool:
    return details.from_date.year == 1 or details.to_date.year == 1


def _identity(animal) -> str:
    raw = "|".join(
        (value or "").strip().casefold()
        for value in (
            animal.farmer_mobile_number,
            animal.cattle_type,
            animal.breed,
            animal.district_name,
            animal.village_name,
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_snapshot(
    details: VerifiedAnimalDetailsModel,
    previous: CattleListingsSnapshotModel | None,
    fetched_at: float,
) -> CattleListingsSnapshotModel:
    """Number the provider's animals, keeping the numbers of animals already cached.

    New animals get the next unused number, so a number the farmer saw keeps
    pointing at the same animal across hourly refreshes, and a sold animal's
    number is never reused for another one.
    """
    rows = [
        (group.breed_name, animal)
        for groups in (details.cows, details.buffaloes)
        for group in groups
        for animal in group.animals
    ]
    # Deterministic order: by place, species, breed, then the remaining fields.
    rows.sort(
        key=lambda row: tuple(
            str(value or "").casefold()
            for value in (
                row[1].district_name,
                row[1].village_name,
                row[1].cattle_type,
                row[1].breed or row[0],
                row[1].age,
                row[1].lactation,
                row[1].cattle_price,
                row[1].milk_per_day,
                row[1].farmer_mobile_number,
            )
        )
    )
    known = {listing.key: listing.number for listing in previous.listings} if previous else {}
    next_number = previous.next_number if previous else 1
    seen: dict[str, int] = defaultdict(int)
    listings = []
    for breed_group, animal in rows:
        identity = _identity(animal)
        seen[identity] += 1
        key = f"{identity}#{seen[identity]}"
        number = known.get(key)
        if number is None:
            number = next_number
            next_number += 1
        listings.append(
            CattleListingModel(
                **animal.model_dump(),
                number=number,
                key=key,
                breed_group=breed_group,
            )
        )
    listings.sort(key=lambda listing: listing.number)
    return CattleListingsSnapshotModel(
        fetched_at=fetched_at,
        from_date=details.from_date,
        to_date=details.to_date,
        total_cows=details.total_cows,
        total_buffaloes=details.total_buffaloes,
        next_number=next_number,
        listings=listings,
    )


async def read_snapshot() -> CattleListingsSnapshotModel | None:
    try:
        raw = await redis_client.get(SNAPSHOT_KEY)
    except Exception as exc:
        logger.warning("Cattle listings snapshot read failed: %s", type(exc).__name__)
        return None
    if not raw:
        return None
    try:
        return CattleListingsSnapshotModel.model_validate_json(raw)
    except ValidationError as exc:
        logger.error("Cattle listings snapshot in Redis is unreadable (%d errors)", exc.error_count())
        return None


async def refresh_snapshot(now: float | None = None) -> CattleListingsSnapshotModel | None:
    """Fetch and store a new snapshot if this pod gets the lock; None otherwise."""
    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return None
    try:
        acquired = await redis_client.set(LOCK_KEY, "1", ex=LOCK_TTL_SECONDS, nx=True)
    except Exception as exc:
        logger.warning("Cattle listings refresh lock failed: %s", type(exc).__name__)
        return None
    if not acquired:
        return None
    details = await get_verified_animal_details_api(token)
    if details is None:
        return None
    if is_failure_sentinel(details):
        logger.warning("Cattle listings provider returned its failure sentinel; keeping the cached copy")
        return None
    previous = await read_snapshot()
    snapshot = build_snapshot(details, previous, now if now is not None else time.time())
    try:
        await redis_client.set(
            SNAPSHOT_KEY,
            snapshot.model_dump_json(),
            ex=settings.cattle_listings_max_stale_seconds,
        )
    except Exception as exc:
        logger.warning("Cattle listings snapshot write failed: %s", type(exc).__name__)
    logger.info(
        "Cattle listings refreshed: %d animals, next number %d",
        len(snapshot.listings),
        snapshot.next_number,
    )
    return snapshot


def snapshot_age_seconds(snapshot: CattleListingsSnapshotModel, now: float | None = None) -> float:
    return (now if now is not None else time.time()) - snapshot.fetched_at


async def get_snapshot(now: float | None = None) -> CattleListingsSnapshotModel | None:
    """The listings for a tool request: cached, or fetched now when missing or old."""
    now = now if now is not None else time.time()
    try:
        await redis_client.set(LAST_REQUEST_KEY, str(now), ex=settings.cattle_listings_idle_seconds)
    except Exception as exc:
        logger.warning("Cattle listings request stamp failed: %s", type(exc).__name__)
    snapshot = await read_snapshot()
    if snapshot is not None and snapshot_age_seconds(snapshot, now) < settings.cattle_listings_refresh_seconds:
        return snapshot
    fresh = await refresh_snapshot(now)
    if fresh is not None:
        return fresh
    if snapshot is not None:
        return snapshot
    # Another pod may be fetching right now.
    deadline = time.monotonic() + WAIT_FOR_PEER_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        snapshot = await read_snapshot()
        if snapshot is not None:
            return snapshot
    return None


async def refresh_if_in_use(now: float | None = None) -> bool:
    """Background tick: refresh an old snapshot only while the tool is in use."""
    now = now if now is not None else time.time()
    if not await redis_client.exists(LAST_REQUEST_KEY):
        return False
    snapshot = await read_snapshot()
    if snapshot is not None and snapshot_age_seconds(snapshot, now) < settings.cattle_listings_refresh_seconds:
        return False
    return await refresh_snapshot(now) is not None
