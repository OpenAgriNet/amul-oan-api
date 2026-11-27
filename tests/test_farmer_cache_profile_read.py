"""The app's profile read (/user/me): the cached farmer record as it is.

The read never waits on the farmer API for a cached record. A found record is
refreshed, in the background, only once it nears the end of its Redis retention,
whatever else made it stale. A not_found record is re-checked in the background
once its interval has passed. With nothing cached, the record is fetched.

Self-contained: asyncio.run + mocks, no Redis.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

import agents.tools.farmer_cache as fc
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord

PHONE = "9876543210"


def _cached(status: str, age_hours: float, stale_reason: Optional[str] = None) -> FarmerDataEnvelope:
    """A record as get_cached_farmer_data returns it: freshness computed, and
    marked stale for ``stale_reason`` when one is given."""
    farmers = [FarmerRecord(farmerName="Ramesh", farmerCode="F1")] if status == "found" else []
    envelope = FarmerDataEnvelope.from_records(farmers, source="cache", lookup_status=status)
    envelope.fetchedAt = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    envelope.stale, envelope.staleReason, envelope.refreshAfter = fc._compute_freshness(envelope)
    if stale_reason:
        envelope.stale, envelope.staleReason = True, stale_reason
    return envelope


def _read(cached: Optional[FarmerDataEnvelope], phone: str = PHONE):
    """Run the profile read on ``cached``; returns (result, enqueue, bounded fetch)."""
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=cached)), \
         patch.object(fc, "enqueue_farmer_refresh", new=AsyncMock()) as enqueue, \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock()) as bounded:
        result = asyncio.run(fc.get_farmer_data_for_profile(phone))
    return result, enqueue, bounded


@pytest.mark.parametrize(
    "age_hours, stale_reason",
    [
        (1, None),
        (13, None),  # past the refresh interval
        (30, None),  # past max-serve-stale at the code defaults: other readers wait on a fetch
        (1, "ai_technician_lookup_failed"),
        (1, "animals_expired"),
        (1, "missing_animals"),
    ],
)
def test_found_record_is_served_without_a_refresh(age_hours, stale_reason):
    cached = _cached("found", age_hours, stale_reason)
    result, enqueue, bounded = _read(cached)
    assert result is cached
    enqueue.assert_not_called()
    bounded.assert_not_called()


def test_a_found_record_near_the_end_of_its_retention_is_refreshed_in_the_background():
    due_hours = fc._PROFILE_REFRESH_AFTER / 3600

    result, enqueue, bounded = _read(_cached("found", due_hours - 1))
    enqueue.assert_not_called()

    cached = _cached("found", due_hours + 1, "ai_technician_lookup_failed")
    result, enqueue, bounded = _read(cached)
    assert result is cached
    enqueue.assert_awaited_once_with(PHONE)
    bounded.assert_not_called()


def test_old_not_found_record_is_rechecked_in_the_background():
    cached = _cached("not_found", fc.FARMER_NEGATIVE_REFRESH_INTERVAL / 3600 + 1)
    result, enqueue, bounded = _read(cached)
    assert result is cached
    enqueue.assert_awaited_once_with(PHONE)
    bounded.assert_not_called()


def test_recent_not_found_record_is_not_rechecked():
    cached = _cached("not_found", fc.FARMER_NEGATIVE_REFRESH_INTERVAL / 3600 - 1)
    result, enqueue, bounded = _read(cached)
    assert result is cached
    enqueue.assert_not_called()
    bounded.assert_not_called()


def test_legacy_unknown_record_reads_as_none_and_is_queued():
    result, enqueue, bounded = _read(_cached("unknown", 0))
    assert result is None
    enqueue.assert_awaited_once_with(PHONE)
    bounded.assert_not_called()


def test_nothing_cached_fetches_the_record():
    fetched = _cached("found", 0)
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=None)), \
         patch.object(fc, "_enqueue_backoff_active", new=AsyncMock(return_value=False)), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock(return_value=fetched)) as bounded:
        result = asyncio.run(fc.get_farmer_data_for_profile("+91 98765 43210"))
    assert result is fetched
    bounded.assert_awaited_once_with(PHONE)
