"""An upstream failure is not an answer, and must never be cached as one.

Issue #282 root cause B: 108 of 356 failing voice sessions had no fetch span at
all — they were served a cached empty envelope. Twenty-eight of those phones
booked successfully at other times, which proves those envelopes should never
have been written. The cause is that `fetch_farmer_amulpashudhan` returned None
for a 204, a non-200, a JSON error, a timeout AND a genuine "no such farmer",
so `refresh_farmer_data` wrote `not_found` and pinned it for the two-hour
FARMER_NEGATIVE_REFRESH_INTERVAL.

The distinction matters to the caller, not just the cache: a cached `not_found`
tells a registered farmer their number is not registered, while an unresolved
lookup tells them to try again shortly.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio

import httpx
import pytest

from agents.voice.tools import farmer_animal_backends as backends
from agents.voice.tools.farmer_animal_backends import BackendUnavailableError


def _response(status: int, body: str = "") -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("GET", "https://api.amulpashudhan.com/x"),
    )


# ── The partner's "not found" is an absence, not an outage ──────────────────
# api.amulpashudhan.com answers "this mobile has no farmer record" with HTTP 500
# and a 36-byte body. Across the full retained window only 9 of 234,340 HTTP 500s
# were anything else.

def test_partner_not_found_500_is_an_absence():
    r = _response(500, '{"Error":"Farmer Record Not Found."}')
    assert backends._parse_farmer_response(r, "amulpashudhan") is None


def test_other_500_is_an_outage():
    r = _response(500, '{"Error":"Internal Server Error"}')
    with pytest.raises(BackendUnavailableError):
        backends._parse_farmer_response(r, "amulpashudhan")


@pytest.mark.parametrize("status", [401, 403, 429, 502, 503])
def test_error_statuses_are_outages(status):
    with pytest.raises(BackendUnavailableError):
        backends._parse_farmer_response(_response(status, "nope"), "amulpashudhan")


def test_unparseable_body_is_an_outage():
    with pytest.raises(BackendUnavailableError):
        backends._parse_farmer_response(_response(200, "<html>gateway</html>"), "herdman")


def test_204_is_an_absence():
    assert backends._parse_farmer_response(_response(204), "amulpashudhan") is None


def test_empty_200_is_an_absence():
    assert backends._parse_farmer_response(_response(200, "   "), "amulpashudhan") is None


def test_records_are_returned():
    r = _response(200, '[{"farmerCode": "5058"}]')
    assert backends._parse_farmer_response(r, "amulpashudhan") == [{"farmerCode": "5058"}]


def test_transport_failure_raises_rather_than_returning_none(monkeypatch):
    """A timeout is the clearest possible "we do not know". It used to be
    swallowed by `except (..., Exception): return None`."""
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(backends.httpx, "AsyncClient", lambda **k: _Client())
    with pytest.raises(BackendUnavailableError):
        asyncio.run(backends.fetch_farmer_amulpashudhan("9876543210", "tok"))


# ── The cache must not record an outage ─────────────────────────────────────

def _patch_cache(monkeypatch):
    """Capture writes to the farmer cache and stub the refresh lock."""
    from agents.voice.services import farmer_cache as fc

    written = {}

    async def _set(phone, data):
        written["envelope"] = data

    async def _get(phone):
        return None

    class _Redis:
        async def set(self, *a, **k):
            return True

        async def delete(self, *a, **k):
            return True

        async def exists(self, *a, **k):
            return False

        async def sadd(self, *a, **k):
            return 1

    monkeypatch.setattr(fc, "set_cached_farmer_data", _set)
    monkeypatch.setattr(fc, "get_cached_farmer_data", _get)
    monkeypatch.setattr(fc, "redis_client", _Redis())
    return fc, written


def test_upstream_failure_writes_nothing_and_returns_unresolved(monkeypatch):
    fc, written = _patch_cache(monkeypatch)

    async def _raise(phone):
        raise BackendUnavailableError("amulpashudhan", "HTTP 502")

    monkeypatch.setattr(fc, "fetch_farmer_info_raw", _raise)

    result = asyncio.run(fc.refresh_farmer_data("9876543210"))

    assert result is None, "an outage must not produce an envelope"
    assert "envelope" not in written, "an outage must not be cached as not_found"


def test_confirmed_absence_is_still_cached(monkeypatch):
    """The negative cache is worth keeping — it stops every turn from an
    unregistered caller re-hitting the partner. Only its trigger narrows."""
    fc, written = _patch_cache(monkeypatch)

    async def _empty(phone):
        return None

    monkeypatch.setattr(fc, "fetch_farmer_info_raw", _empty)

    result = asyncio.run(fc.refresh_farmer_data("9876543210"))

    assert result is not None
    assert result.lookupStatus == "not_found"
    assert written["envelope"].lookupStatus == "not_found"


def test_records_are_cached_as_found(monkeypatch):
    fc, written = _patch_cache(monkeypatch)

    async def _records(phone):
        return [{"farmerCode": "5058", "societyCode": "00002", "unionCode": "159"}]

    async def _no_technicians(records):
        return []

    monkeypatch.setattr(fc, "fetch_farmer_info_raw", _records)
    monkeypatch.setattr(fc, "_fetch_ai_technicians", _no_technicians)

    result = asyncio.run(fc.refresh_farmer_data("9876543210"))

    assert result.lookupStatus == "found"
    assert written["envelope"].lookupStatus == "found"


# ── P2: the trace must be able to tell the two apart ────────────────────────

def test_error_body_is_recorded():
    """{status_code, bytes, keys} could not distinguish a negative lookup from a
    real fault, which is what made root cause B expensive to attribute."""
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(),
        _response(500, '{"Error":"Farmer Record Not Found."}'),
        provider="amulpashudhan",
        url="https://x",
    )
    assert "Farmer Record Not Found" in captured["output"]["error_body"]


def test_error_body_redacts_phone_numbers():
    """Error responses can echo the request, and the request carries the
    caller's mobile."""
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(),
        _response(500, '{"Error":"no record for 9876543210"}'),
        provider="amulpashudhan",
        url="https://x",
    )
    body = captured["output"]["error_body"]
    assert "9876543210" not in body
    assert "[redacted]" in body


def test_successful_response_records_no_error_body():
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(), _response(200, '[{"farmerCode":"5058"}]'),
        provider="amulpashudhan", url="https://x",
    )
    assert "error_body" not in captured["output"]


def test_get_farmer_by_mobile_unpacks_the_dual_backend_result():
    """_fetch_farmer_records_dual_backend returns (records, upstream_failed).
    Both callers must unpack it: binding the 2-tuple to `records` makes
    `if not records` permanently false and then calls .get() on a list."""
    import asyncio
    from unittest.mock import AsyncMock, patch
    import agents.voice.tools.farmer as farmer_tool

    with patch.dict("os.environ", {"PASHUGPT_TOKEN": "t1", "PASHUGPT_TOKEN_3": "t3"}), \
         patch.object(
             farmer_tool, "_fetch_farmer_records_dual_backend",
             new=AsyncMock(return_value=([{"farmerName": "Ramesh", "societyName": "S"}], False)),
         ):
        assert "Ramesh" in asyncio.run(farmer_tool.get_farmer_by_mobile("9999999999"))

    with patch.dict("os.environ", {"PASHUGPT_TOKEN": "t1", "PASHUGPT_TOKEN_3": "t3"}), \
         patch.object(
             farmer_tool, "_fetch_farmer_records_dual_backend",
             new=AsyncMock(return_value=([], True)),
         ):
        out = asyncio.run(farmer_tool.get_farmer_by_mobile("9999999999"))
    assert "could not be looked up" in out and "No farmer data found" not in out
