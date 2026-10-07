"""Voice booking tools must be idempotent per session so an agent re-run (the
OSS->managed streaming fallback re-executes tool calls) cannot double-book.
Voice books through chat's Beckn confirm, so a reservation is released only when
the booking provably did not happen."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agents.tools import ai_call as chat_ai_call
from agents.tools.beckn import network as beckn_network
from agents.voice.tools import ai_call as ai_mod
from agents.voice.tools import health_call as hc_mod
from agents.voice.models.ai_call import AISpecies
from agents.voice.models.health_call import HealthCaseType
from app.core import cache as cache_mod

# create_ai_call rejects identifiers that cannot be real; 24 base64 chars is the
# shape of every real prod technician id.
TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="
SPECIES = next(iter(AISpecies))
CASE_TYPE = next(iter(HealthCaseType))


def _ctx(session_id, tool_call_id=None):
    async def _ensure_in_scope():
        return True

    return SimpleNamespace(
        deps=SimpleNamespace(session_id=session_id, ensure_in_scope=_ensure_in_scope),
        tool_call_id=tool_call_id,
    )


@pytest.fixture
def store(monkeypatch):
    """In-memory cache simulating Redis: add() is atomic SET-NX (raises if key
    exists), shared by reserve/release_reservation and the tool's set/get."""
    store = {}

    async def fake_add(key, value, ttl=None, namespace=None):
        k = (namespace, key)
        if k in store:
            raise ValueError("key exists")  # aiocache add == Redis SET NX
        store[k] = value
        return True

    async def fake_set(key, value, ttl=None, namespace=None):
        store[(namespace, key)] = value

    async def fake_get(key, namespace=None):
        return store.get((namespace, key))

    async def fake_delete(key, namespace=None):
        store.pop((namespace, key), None)

    monkeypatch.setattr(cache_mod.cache, "add", fake_add)
    monkeypatch.setattr(cache_mod.cache, "set", fake_set)
    monkeypatch.setattr(cache_mod.cache, "get", fake_get)
    monkeypatch.setattr(cache_mod.cache, "delete", fake_delete)
    # Voice holds one booking per call whatever chat's flag says.
    monkeypatch.setattr(chat_ai_call.settings, "ai_call_booking_guard_enabled", False)
    monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)
    return store


def _confirms(monkeypatch, outcomes, name="network_create_ai_call_result", delay=0.0):
    """Answer each Beckn confirm with the next outcome: a NetworkBookingResult,
    or an exception to raise. Returns the list of calls made."""
    calls = []
    outcomes = list(outcomes)

    async def fake_confirm(*args, **kwargs):
        calls.append((args, kwargs))
        if delay:
            await asyncio.sleep(delay)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(beckn_network, name, fake_confirm)
    return calls


def _booked(ticket="T1"):
    return beckn_network.NetworkBookingResult(True, ticket, f"Booked successfully. Ticket: {ticket}")


def _nack():
    return beckn_network.NetworkBookingResult(
        False, None, "Booking failed on the network: rejected", authoritative_no_booking=True
    )


def _pending():
    return beckn_network.NetworkBookingResult(
        False, None, "Booking is still pending confirmation.", authoritative_no_booking=False
    )


def _book_ai(session_id="s1", tool_call_id=None):
    return asyncio.run(
        ai_mod.create_ai_call(_ctx(session_id, tool_call_id), "159", "00002", "5058", TECH_ID, SPECIES)
    )


def _book_health(session_id="s1", remark="fever"):
    return asyncio.run(
        hc_mod.create_health_call(_ctx(session_id), "159", "00002", "5058", SPECIES, CASE_TYPE, remark)
    )


def test_ai_call_idempotent_on_rerun(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked(), _booked("T2")])

    r1 = _book_ai()
    r2 = _book_ai()

    assert len(calls) == 1
    assert "booked successfully" in r1.lower()
    assert "already" in r2.lower()
    assert store[("ai_call_booked", "s1")] == {"ticket": "T1", "species": SPECIES.value}


def test_ai_call_sends_the_callers_codes_and_tool_call_id(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked()])

    _book_ai(session_id="s1", tool_call_id="call-7")

    (args, kwargs), = calls
    assert args == ("159", "00002", "5058", TECH_ID, SPECIES.value)
    assert kwargs == {"session_id": "s1", "tool_call_id": "call-7"}


def test_health_call_sends_the_callers_codes_and_tool_call_id(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked("H1")], name="network_create_health_call_result")

    asyncio.run(hc_mod.create_health_call(
        _ctx("s1", "call-9"), "159", "00002", "5058", SPECIES, CASE_TYPE, "fever"
    ))

    (args, kwargs), = calls
    assert args == ("159", "00002", "5058", SPECIES.value, CASE_TYPE.value, "fever")
    assert kwargs == {"session_id": "s1", "tool_call_id": "call-9"}


def test_health_call_idempotent_on_rerun(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked("H1"), _booked("H2")], name="network_create_health_call_result")

    # remark differs across the re-run (model output varies) — session key still dedupes
    r1 = _book_health(remark="remark v1")
    r2 = _book_health(remark="remark v2")

    assert len(calls) == 1
    assert "booked successfully" in r1.lower()
    assert "already" in r2.lower()


def test_ai_call_concurrent_submits_book_once(monkeypatch, store):
    """Two concurrent submits for the same session (double-tap / retry) must
    result in exactly ONE booking — the atomic reservation closes the race."""
    calls = _confirms(monkeypatch, [_booked("T1"), _booked("T2")], delay=0.02)

    async def go():
        return await asyncio.gather(
            ai_mod.create_ai_call(_ctx("sX"), "159", "00002", "5058", TECH_ID, SPECIES),
            ai_mod.create_ai_call(_ctx("sX"), "159", "00002", "5058", TECH_ID, SPECIES),
        )

    r1, r2 = asyncio.run(go())
    assert len(calls) == 1
    assert any("booked successfully" in r.lower() for r in (r1, r2))
    assert any("already" in r.lower() for r in (r1, r2))


def test_ai_call_no_session_does_not_crash(monkeypatch, store):
    _confirms(monkeypatch, [_booked()])
    assert "booked successfully" in _book_ai(session_id=None).lower()


@pytest.mark.parametrize("outcome", [
    _pending(),
    httpx.ReadTimeout("read timed out"),
    RuntimeError("unexpected"),
])
def test_ai_call_unconfirmed_booking_keeps_the_reservation(monkeypatch, store, outcome):
    """The confirm may have reached the BPP, which may have booked and texted
    the farmer: a retry in the same call must not book a second visit."""
    calls = _confirms(monkeypatch, [outcome, _booked()])

    first = _book_ai()
    second = _book_ai()

    assert len(calls) == 1
    assert "booked successfully" not in first.lower()
    assert "already" in second.lower()


@pytest.mark.parametrize("outcome", [
    _nack(),
    httpx.ConnectError("connection refused"),
])
def test_ai_call_booking_that_provably_did_not_happen_can_be_retried(monkeypatch, store, outcome):
    calls = _confirms(monkeypatch, [outcome, _booked()])

    first = _book_ai()
    second = _book_ai()

    assert len(calls) == 2
    assert "booked successfully" not in first.lower()
    assert "booked successfully" in second.lower()


def test_health_call_unconfirmed_booking_keeps_the_reservation(monkeypatch, store):
    calls = _confirms(
        monkeypatch, [_pending(), _booked("H1")], name="network_create_health_call_result"
    )

    first = _book_health()
    second = _book_health()

    assert len(calls) == 1
    assert "booked successfully" not in first.lower()
    assert "already" in second.lower()


def test_health_call_nack_can_be_retried(monkeypatch, store):
    calls = _confirms(
        monkeypatch, [_nack(), _booked("H1")], name="network_create_health_call_result"
    )

    _book_health()
    second = _book_health()

    assert len(calls) == 2
    assert "booked successfully" in second.lower()
