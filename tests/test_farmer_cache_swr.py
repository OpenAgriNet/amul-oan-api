"""Unit tests for the stale-while-revalidate farmer cache (Inc 4): freshness
intervals, the Redis refresh queue, the bounded cold fetch, and the don't-
downgrade / lock semantics.

Self-contained: uses asyncio.run + mocks, so it needs neither a live Redis nor
pytest-asyncio.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import agents.services.farmer_cache as fc
from agents.tools.farmer import FarmerFetchOutcome

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cold_import_has_no_circular_import():
    """Regression: `uvicorn main:app` imports the refresh worker before the
    routers, so farmer_cache must be importable COLD (first touch) without
    hitting the farmer_cache <-> agents.tools cycle. Run in a subprocess to get
    a truly fresh interpreter, mirroring the app's startup import order."""
    code = (
        "import app.tasks.farmer_refresh_worker;"
        "import agents.services.farmer_cache;"
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"cold import failed:\n{result.stderr}"
    assert "OK" in result.stdout


class _Env:
    """Duck-typed stand-in for FarmerDataEnvelope (only fields _compute_freshness reads)."""

    def __init__(self, lookup_status, age_hours):
        self.lookupStatus = lookup_status
        self.fetchedAt = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
        self.stale, self.staleReason, self.refreshAfter = fc._compute_freshness(self)
        self.farmers = [{}] if lookup_status == "found" else []

    def model_dump(self):
        return {"lookupStatus": self.lookupStatus, "fetchedAt": self.fetchedAt}


def test_freshness_intervals_found_vs_not_found():
    # found: soft expiry at 12h
    assert fc._compute_freshness(_Env("found", 6))[0] is False
    assert fc._compute_freshness(_Env("found", 13))[0] is True
    # not_found: shorter soft expiry at 2h
    assert fc._compute_freshness(_Env("not_found", 1))[0] is False
    assert fc._compute_freshness(_Env("not_found", 3))[0] is True
    # unknown: always treated as non-authoritative / immediately stale
    assert fc._compute_freshness(_Env("unknown", 0))[0] is True
    assert fc._compute_freshness(_Env("unknown", 3))[0] is True


def test_compute_backoff_exponential_with_cap():
    assert fc._compute_backoff_seconds(1) == fc.FARMER_REFRESH_RETRY_BASE_SECONDS
    assert fc._compute_backoff_seconds(2) == fc.FARMER_REFRESH_RETRY_BASE_SECONDS * 2
    assert fc._compute_backoff_seconds(99) == fc.FARMER_REFRESH_RETRY_MAX_SECONDS


def test_interval_constants():
    # Defaults (env unset) match the documented 12h / 2h / 7d.
    assert fc.FARMER_REFRESH_INTERVAL == 60 * 60 * 12
    assert fc.FARMER_NEGATIVE_REFRESH_INTERVAL == 60 * 60 * 2
    assert fc.FARMER_CACHE_TTL == 60 * 60 * 24 * 7


def test_enqueue_pushes_phone_to_redis_set():
    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=None)
    with patch.object(fc, "redis_client", fake_redis):
        asyncio.run(fc.enqueue_farmer_refresh("9999999999"))
    fake_redis.sadd.assert_awaited_once_with(fc.FARMER_REFRESH_QUEUE_KEY, "9999999999")
    fake_redis.set.assert_awaited()


def test_enqueue_ignores_empty_phone():
    fake_redis = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis):
        asyncio.run(fc.enqueue_farmer_refresh(""))
    fake_redis.get.assert_not_called()
    fake_redis.sadd.assert_not_called()


def test_enqueue_skips_duplicate_within_backoff_window():
    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=None)
    backoff = AsyncMock(side_effect=[False, True])
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "_enqueue_backoff_active", new=backoff):
        asyncio.run(fc.enqueue_farmer_refresh("9999999999"))
        asyncio.run(fc.enqueue_farmer_refresh("9999999999"))
    fake_redis.sadd.assert_awaited_once_with(fc.FARMER_REFRESH_QUEUE_KEY, "9999999999")


def test_enqueue_failed_sadd_does_not_write_backoff_state():
    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.sadd = AsyncMock(side_effect=RuntimeError("queue down"))
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "_record_refresh_enqueue_backoff", new=AsyncMock()) as record:
        asyncio.run(fc.enqueue_farmer_refresh("9999999999"))
    record.assert_not_called()


def test_drain_pops_batch_and_refreshes_each():
    fake_redis = AsyncMock()
    fake_redis.spop.return_value = ["111", "222", "333"]
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "refresh_farmer_data", new=AsyncMock()) as refresh:
        processed = asyncio.run(fc.drain_farmer_refresh_queue_once(batch=10))
    assert processed == 3
    fake_redis.spop.assert_awaited_once_with(fc.FARMER_REFRESH_QUEUE_KEY, 10)
    assert refresh.await_count == 3


def test_drain_empty_queue_returns_zero():
    fake_redis = AsyncMock()
    fake_redis.spop.return_value = None
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "refresh_farmer_data", new=AsyncMock()) as refresh:
        processed = asyncio.run(fc.drain_farmer_refresh_queue_once())
    assert processed == 0
    refresh.assert_not_called()


def test_bounded_fetch_returns_envelope_on_success():
    sentinel = object()
    with patch.object(fc, "refresh_farmer_data", new=AsyncMock(return_value=sentinel)):
        result = asyncio.run(fc.refresh_farmer_data_bounded("111", timeout=1.0))
    assert result is sentinel


def test_bounded_fetch_times_out_and_enqueues():
    async def _slow(_phone):
        await asyncio.sleep(5)

    enqueue = AsyncMock()
    with patch.object(fc, "refresh_farmer_data", new=_slow), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue):
        result = asyncio.run(fc.refresh_farmer_data_bounded("111", timeout=0.05))
    assert result is None
    enqueue.assert_awaited_once_with("111")


def test_exceeds_max_serve_stale():
    assert fc.exceeds_max_serve_stale(_Env("found", 1)) is False
    assert fc.exceeds_max_serve_stale(_Env("found", 1000)) is True
    assert fc.exceeds_max_serve_stale(None) is True


def test_refresh_keeps_found_on_transient_not_found():
    """A transient empty upstream response must not wipe a still-fresh 'found' record."""
    existing = _Env("found", 1)
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)   # lock acquired
    fake_redis.delete = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(
             fc,
             "fetch_farmer_info_with_outcome",
             new=AsyncMock(return_value=(None, FarmerFetchOutcome.ERROR)),
         ), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=existing)), \
         patch.object(fc, "set_cached_farmer_data", new=AsyncMock()) as set_cache, \
         patch.object(fc, "_record_refresh_failure", new=AsyncMock()) as record_failure:
        result = asyncio.run(fc.refresh_farmer_data("9999999999"))
    assert result is existing
    set_cache.assert_not_called()  # did NOT downgrade to not_found
    record_failure.assert_awaited()


def test_refresh_keeps_old_found_past_ceiling_on_transient_not_found():
    """Even past max-serve-stale, a transient empty must not overwrite a 'found'
    record (genuine removal is left to the hard TTL)."""
    existing = _Env("found", 1000)  # well past the 24h ceiling
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)
    fake_redis.delete = AsyncMock()
    fake_redis.ttl = AsyncMock(return_value=100000)
    fake_cache = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "cache", fake_cache), \
         patch.object(
             fc,
             "fetch_farmer_info_with_outcome",
             new=AsyncMock(return_value=(None, FarmerFetchOutcome.ERROR)),
         ), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=existing)), \
         patch.object(fc, "set_cached_farmer_data", new=AsyncMock()) as set_cache, \
         patch.object(fc, "_record_refresh_failure", new=AsyncMock()):
        result = asyncio.run(fc.refresh_farmer_data("9999999999"))
    assert result is existing
    set_cache.assert_not_called()


def test_refresh_lock_busy_awaits_inflight_value():
    """A refresh that loses the NX-lock race returns the in-flight result, not
    None (None would defeat max-serve-stale and drop queued refreshes)."""
    sentinel = _Env("found", 0)
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=None)   # lock busy
    fake_redis.exists = AsyncMock(return_value=0)   # in-flight refresh already finished
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=sentinel)):
        result = asyncio.run(fc.refresh_farmer_data("111"))
    assert result is sentinel


def test_bounded_timeout_releases_lock_on_cancel():
    """When the bounded fetch times out and cancels the in-flight refresh, the
    NX lock is released in refresh_farmer_data's finally."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)   # lock acquired
    fake_redis.delete = AsyncMock()

    async def _slow(_phone):
        await asyncio.sleep(5)
        return None, FarmerFetchOutcome.ERROR

    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "fetch_farmer_info_with_outcome", new=_slow), \
         patch.object(fc, "enqueue_farmer_refresh", new=AsyncMock()), \
         patch.object(fc, "_record_refresh_failure", new=AsyncMock()):
        result = asyncio.run(fc.refresh_farmer_data_bounded("111", timeout=0.05))
    assert result is None
    fake_redis.delete.assert_awaited()  # lock released on cancellation


def test_lock_busy_never_clears_reenqueues_not_stale():
    """If the in-flight holder outlives the wait, return None and RE-ENQUEUE —
    do not serve the stale record or silently drop the phone."""
    old = _Env("found", 1000)
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=None)   # lock busy
    fake_redis.exists = AsyncMock(return_value=1)   # never clears within the wait
    enqueue = AsyncMock()
    with patch.object(fc, "FARMER_COLD_FETCH_TIMEOUT", 0.3), \
         patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=old)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue):
        result = asyncio.run(fc.refresh_farmer_data("111"))
    assert result is None                       # NOT the ancient cached record
    enqueue.assert_awaited_once_with("111")     # re-queued, not dropped


def test_lock_busy_held_then_released_returns_fresh():
    """If the holder finishes within the wait, return its fresh result and do
    not re-enqueue."""
    fresh = _Env("found", 0)
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=None)               # lock busy
    fake_redis.exists = AsyncMock(side_effect=[1, 0])           # held, then released
    enqueue = AsyncMock()
    with patch.object(fc, "FARMER_COLD_FETCH_TIMEOUT", 2.0), \
         patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=fresh)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue):
        result = asyncio.run(fc.refresh_farmer_data("111"))
    assert result is fresh
    enqueue.assert_not_called()


def test_restamp_kept_record_preserves_ttl():
    """Don't-downgrade past the ceiling restamps fetchedAt (to stop a per-turn
    block) but PRESERVES the remaining hard TTL."""
    existing = _Env("found", 1000)  # >24h
    orig_fetched = existing.fetchedAt
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)       # lock acquired
    fake_redis.delete = AsyncMock()
    fake_redis.ttl = AsyncMock(return_value=100000)     # remaining hard TTL
    fake_cache = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(fc, "cache", fake_cache), \
         patch.object(
             fc,
             "fetch_farmer_info_with_outcome",
             new=AsyncMock(return_value=(None, FarmerFetchOutcome.ERROR)),
         ), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=existing)), \
         patch.object(fc, "_record_refresh_failure", new=AsyncMock()):
        result = asyncio.run(fc.refresh_farmer_data("9999999999"))
    assert result is existing
    assert existing.fetchedAt != orig_fetched           # restamped
    fake_cache.set.assert_awaited()
    assert fake_cache.set.call_args.kwargs.get("ttl") == 100000  # hard TTL preserved, not reset


def test_get_or_fetch_fresh_hit_does_not_enqueue():
    fresh = _Env("found", 1)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=fresh)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data", new=AsyncMock()) as refresh:
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is fresh
    enqueue.assert_not_called()
    refresh.assert_not_called()


def test_get_or_fetch_soft_stale_enqueues_and_returns_cached():
    stale = _Env("found", 13)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=stale)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock()) as bounded:
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is stale
    enqueue.assert_awaited_once_with("9999999999")
    bounded.assert_not_called()


def test_get_or_fetch_max_serve_stale_blocks_then_returns_fresh():
    ancient = _Env("found", 1000)
    fresh = _Env("found", 0)
    enqueue = AsyncMock()
    bounded = AsyncMock(return_value=fresh)
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=ancient)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=bounded):
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is fresh
    bounded.assert_awaited_once_with("9999999999")
    enqueue.assert_not_called()


def test_get_or_fetch_max_serve_stale_fallback_returns_stale_and_enqueues():
    ancient = _Env("found", 1000)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=ancient)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock(return_value=None)):
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is ancient
    enqueue.assert_awaited_once_with("9999999999")


def test_get_or_fetch_cache_miss_calls_bounded_refresh():
    sentinel = _Env("found", 0)
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=None)), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock(return_value=sentinel)) as refresh:
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is sentinel
    refresh.assert_awaited_once_with("9999999999")


def test_refresh_error_without_existing_returns_none_and_does_not_cache():
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)
    fake_redis.delete = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(
             fc,
             "fetch_farmer_info_with_outcome",
             new=AsyncMock(return_value=(None, FarmerFetchOutcome.ERROR)),
         ), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=None)), \
         patch.object(fc, "set_cached_farmer_data", new=AsyncMock()) as set_cache, \
         patch.object(fc, "_record_refresh_failure", new=AsyncMock()) as record_failure:
        result = asyncio.run(fc.refresh_farmer_data("9999999999"))
    assert result is None
    set_cache.assert_not_called()
    record_failure.assert_awaited_once()


def test_refresh_authoritative_not_found_caches_not_found():
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)
    fake_redis.delete = AsyncMock()
    with patch.object(fc, "redis_client", fake_redis), \
         patch.object(
             fc,
             "fetch_farmer_info_with_outcome",
             new=AsyncMock(return_value=(None, FarmerFetchOutcome.AUTHORITATIVE_NOT_FOUND)),
         ), \
         patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=None)), \
         patch.object(fc, "set_cached_farmer_data", new=AsyncMock()) as set_cache, \
         patch.object(fc, "_clear_refresh_attempt_state", new=AsyncMock()) as clear_state:
        result = asyncio.run(fc.refresh_farmer_data("9999999999"))
    assert result is not None
    assert result.lookupStatus == "not_found"
    set_cache.assert_awaited_once()
    clear_state.assert_awaited_once()


def test_get_or_fetch_legacy_unknown_normalizes_to_none():
    unknown = _Env("unknown", 0)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=unknown)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock()) as bounded:
        result = asyncio.run(fc.get_or_fetch_farmer_data("9999999999"))
    assert result is None
    bounded.assert_not_called()
    assert enqueue.await_count >= 1


def test_cached_only_soft_stale_enqueues_without_block_fetch():
    stale = _Env("found", 13)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=stale)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock()) as bounded:
        result = asyncio.run(fc.get_farmer_data_cached_only("9999999999"))
    assert result is stale
    enqueue.assert_awaited_once_with("9999999999")
    bounded.assert_not_called()


def test_cached_only_max_serve_stale_still_does_not_block():
    ancient = _Env("found", 1000)
    enqueue = AsyncMock()
    with patch.object(fc, "get_cached_farmer_data", new=AsyncMock(return_value=ancient)), \
         patch.object(fc, "enqueue_farmer_refresh", new=enqueue), \
         patch.object(fc, "refresh_farmer_data_bounded", new=AsyncMock()) as bounded:
        result = asyncio.run(fc.get_farmer_data_cached_only("9999999999"))
    assert result is ancient
    enqueue.assert_awaited_once_with("9999999999")
    bounded.assert_not_called()


def test_get_or_fetch_normalizes_equivalent_phone_formats():
    fresh = _Env("found", 1)
    cache_get = AsyncMock(return_value=fresh)
    with patch.object(fc, "get_cached_farmer_data", new=cache_get):
        asyncio.run(fc.get_or_fetch_farmer_data("+91 9876543210"))
        asyncio.run(fc.get_or_fetch_farmer_data("9876543210"))
    assert cache_get.await_count == 2
    assert cache_get.await_args_list[0].args[0] == "9876543210"
    assert cache_get.await_args_list[1].args[0] == "9876543210"
    assert fc._normalize_cache_phone("+91 9876543210") == "9876543210"
