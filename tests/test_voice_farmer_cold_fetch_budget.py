"""Cold-fetch budget and the in-flight marker (issue #282, causes A and D)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio

import pytest

import app.voice.farmer as voice
from agents.voice.models.farmer import FarmerDataEnvelope
from agents.voice.services import farmer_cache as fc


def _envelope():
    return FarmerDataEnvelope.from_records(
        [{"farmerCode": "5058", "societyCode": "00002", "unionCode": "159"}],
        source="api",
        lookup_status="found",
    )


def test_budget_stays_inside_a_callers_patience():
    """The budget is silence on a live call, so it is capped rather than widened.

    Voice, successful 200s, 2026-09-01..09-16: 94.0% of lookups land within 2s,
    3.9% in 2-4s, only 0.86% in 4-8s, and 1.2% exceed 8s anyway. Raising the
    budget past 4s buys under 1% of lookups and doubles the wait for every slow
    caller; the in-flight marker is what recovers a cancelled fetch, on the next
    turn, at no cost to this one.
    """
    assert fc.FARMER_COLD_FETCH_TIMEOUT <= 4.0


def test_budget_is_configurable():
    from app.config import settings
    assert fc.FARMER_COLD_FETCH_TIMEOUT == settings.farmer_cold_fetch_timeout


class _Redis:
    def __init__(self):
        self.keys = {}

    async def set(self, key, value, ex=None, nx=None):
        self.keys[key] = value
        return True

    async def exists(self, key):
        return 1 if key in self.keys else 0

    async def delete(self, key):
        self.keys.pop(key, None)
        return 1

    async def sadd(self, *a, **k):
        return 1


def test_cancelled_fetch_leaves_a_marker(monkeypatch):
    redis = _Redis()
    monkeypatch.setattr(fc, "redis_client", redis)

    async def _hang(phone):
        await asyncio.sleep(10)

    monkeypatch.setattr(fc, "refresh_farmer_data", _hang)

    result = asyncio.run(fc.refresh_farmer_data_bounded("9876543210", timeout=0.05))

    assert result is None
    assert asyncio.run(fc.is_fetch_inflight("9876543210")) is True


def test_no_marker_when_fetch_succeeds(monkeypatch):
    redis = _Redis()
    monkeypatch.setattr(fc, "redis_client", redis)

    async def _ok(phone):
        return _envelope()

    monkeypatch.setattr(fc, "refresh_farmer_data", _ok)

    assert asyncio.run(fc.refresh_farmer_data_bounded("9876543210")) is not None
    assert asyncio.run(fc.is_fetch_inflight("9876543210")) is False


def test_next_turn_does_not_re_block_while_marker_is_set(monkeypatch):
    """The second turn used to pay the same budget again on a still-cold cache."""
    calls = {"n": 0}

    async def _no_cache(mobile):
        return None

    async def _inflight(mobile):
        return True

    async def _bounded(mobile):
        calls["n"] += 1
        return None

    monkeypatch.setattr(voice, "get_farmer_data_cached_only", _no_cache)
    monkeypatch.setattr(voice, "is_fetch_inflight", _inflight)
    monkeypatch.setattr(voice, "refresh_farmer_data_bounded", _bounded)

    assert asyncio.run(voice.get_or_fetch_farmer_data("9876543210")) is None
    assert calls["n"] == 0


def test_blocks_normally_when_no_marker(monkeypatch):
    calls = {"n": 0}

    async def _no_cache(mobile):
        return None

    async def _not_inflight(mobile):
        return False

    async def _bounded(mobile):
        calls["n"] += 1
        return _envelope()

    monkeypatch.setattr(voice, "get_farmer_data_cached_only", _no_cache)
    monkeypatch.setattr(voice, "is_fetch_inflight", _not_inflight)
    monkeypatch.setattr(voice, "refresh_farmer_data_bounded", _bounded)

    assert asyncio.run(voice.get_or_fetch_farmer_data("9876543210")) is not None
    assert calls["n"] == 1


def test_marker_check_failure_does_not_block_the_turn(monkeypatch):
    """Redis being unreachable must not make every caller look unresolved."""
    class _Broken:
        async def exists(self, key):
            raise RuntimeError("redis down")

    monkeypatch.setattr(fc, "redis_client", _Broken())
    assert asyncio.run(fc.is_fetch_inflight("9876543210")) is False
