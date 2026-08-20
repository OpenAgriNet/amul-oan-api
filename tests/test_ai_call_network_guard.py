"""AI-call booking guards: moderation block and per-session idempotency.

Booking always goes through direct PashuGPT. These tests pin:

  * the per-session idempotency guard (Redis SET NX, gated on
    AI_CALL_BOOKING_GUARD_ENABLED — prod runs it ON), so an OSS->managed
    fallback re-run cannot duplicate a booking and its SMS to a real farmer;
  * the moderation verdict block — a booking is IRREVERSIBLE, so it must not
    be written for a query moderation rejected.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agents.tools import ai_call as ai_mod
from app.models.ai_call import AISpecies

SPECIES = next(iter(AISpecies))


def _ctx(session_id, in_scope=True):
    async def _ensure():
        return in_scope

    return SimpleNamespace(deps=SimpleNamespace(session_id=session_id, ensure_in_scope=_ensure))


def _patch_cache(monkeypatch):
    """In-memory cache simulating Redis: add() is atomic SET-NX (raises if the
    key exists), shared by try_reserve/release_reservation and the tool."""
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

    monkeypatch.setattr(ai_mod.cache, "add", fake_add)
    monkeypatch.setattr(ai_mod.cache, "set", fake_set)
    monkeypatch.setattr(ai_mod.cache, "get", fake_get)
    monkeypatch.setattr(ai_mod.cache, "delete", fake_delete)
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    return store


def _patch_direct(monkeypatch):
    """Count direct PashuGPT CreateAICall writes."""
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(
            ticket_number=f"T{calls['n']}", ait_name="AIT",
            model_dump=lambda: {"ticket_number": "T"},
        )

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)
    return calls


def _guard(monkeypatch, on):
    monkeypatch.setattr(ai_mod.settings, "ai_call_booking_guard_enabled", on)


def test_guard_on_refuses_second_booking_in_a_session(monkeypatch):
    """The re-run case: prod runs the guard ON, so a second book in the same
    session short-circuits rather than calling PashuGPT again (and sending a
    second SMS)."""
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_direct(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-1"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-1"), "U", "S", "F", "tech1", SPECIES))

    assert calls["n"] == 1
    assert "booked successfully" in r1
    assert "already" in r2.lower()


def test_guard_on_books_once_under_concurrency(monkeypatch):
    """Two concurrent submits race; the atomic reservation closes the window."""
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = {"n": 0}

    async def slow_api(request, token):
        calls["n"] += 1
        await asyncio.sleep(0.02)  # the window two requests race in
        return SimpleNamespace(
            ticket_number="X", ait_name="AIT",
            model_dump=lambda: {"ticket_number": "X"},
        )

    monkeypatch.setattr(ai_mod, "create_ai_call_api", slow_api)

    async def go():
        return await asyncio.gather(
            ai_mod.create_ai_call(_ctx("s-dir-race"), "U", "S", "F", "t", SPECIES),
            ai_mod.create_ai_call(_ctx("s-dir-race"), "U", "S", "F", "t", SPECIES),
        )

    r1, r2 = asyncio.run(go())
    assert calls["n"] == 1
    assert any("already" in r.lower() for r in (r1, r2))


def test_guard_on_failed_booking_releases_the_reservation(monkeypatch):
    """A failed PashuGPT write means nothing was booked, so the farmer must be
    able to retry inside the TTL instead of being locked out."""
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def failing_api(request, token):
        attempts["n"] += 1
        return None

    monkeypatch.setattr(ai_mod, "create_ai_call_api", failing_api)
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-fail"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-fail"), "U", "S", "F", "t", SPECIES))

    assert "failed" in r1.lower()
    assert attempts["n"] == 2, "the retry never reached PashuGPT — reservation was not released"
    assert "already" not in r2.lower()


def test_guard_off_allows_a_second_booking(monkeypatch):
    """main's default: two cows in heat is a real case."""
    _guard(monkeypatch, False)
    _patch_cache(monkeypatch)
    calls = _patch_direct(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-2"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-2"), "U", "S", "F", "tech1", SPECIES))

    assert calls["n"] == 2
    assert "booked successfully" in r1
    assert "booked successfully" in r2
    assert "already" not in r2.lower()


def test_moderation_rejected_writes_no_booking(monkeypatch):
    """A booking is IRREVERSIBLE: a rejected query must not reach PashuGPT."""
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_direct(monkeypatch)

    r = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-mod", in_scope=False), "U", "S", "F", "t", SPECIES))

    assert calls["n"] == 0
    assert "dairy farming" in r


def test_moderation_rejected_with_guard_off_also_writes_no_booking(monkeypatch):
    """The moderation block is not the idempotency guard — it applies whatever
    the guard flag says."""
    _guard(monkeypatch, False)
    _patch_cache(monkeypatch)
    calls = _patch_direct(monkeypatch)

    r = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-mod2", in_scope=False), "U", "S", "F", "t", SPECIES))

    assert calls["n"] == 0
    assert "dairy farming" in r


def test_redis_error_fail_open_does_not_delete_an_earlier_booking_marker(monkeypatch):
    """`reserve` fails OPEN: when Redis is down it lets the booking through
    WITHOUT holding the key. The old code recorded that as "reserved", so the
    next failure deleted a key it never wrote — here, the marker written by this
    session's EARLIER SUCCESSFUL booking — leaving the session unguarded for the
    rest of the TTL.
    """
    _guard(monkeypatch, True)
    store = _patch_cache(monkeypatch)
    direct = _patch_direct(monkeypatch)

    # 1. a real booking succeeds and marks the session
    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-open"), "U", "S", "F", "t", SPECIES))
    assert "booked successfully" in r1
    marker = store[(ai_mod.AI_CALL_CACHE_NAMESPACE, "s-open")]
    assert marker["ticket"] == "T1"

    # 2. Redis starts erroring on SET NX (not "key exists" — genuinely down)
    async def broken_add(key, value, ttl=None, namespace=None):
        raise RuntimeError("Redis connection reset")

    monkeypatch.setattr(ai_mod.cache, "add", broken_add)

    # 3. a second booking attempt proceeds unguarded (fail-open is UNCHANGED)
    #    and then fails upstream
    async def failing_api(request, token):
        direct["n"] += 1
        return None

    monkeypatch.setattr(ai_mod, "create_ai_call_api", failing_api)
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-open"), "U", "S", "F", "t", SPECIES))
    assert "failed" in r2.lower()
    assert direct["n"] == 2, "fail-open was lost: the booking did not reach the API"

    # 4. the earlier booking's marker must still be there
    assert store.get((ai_mod.AI_CALL_CACHE_NAMESPACE, "s-open")) == marker, (
        "a fail-open 'reservation' we never held was released, wiping the "
        "earlier successful booking's marker"
    )

    # 5. and because it is still there, the guard still refuses a re-run once
    #    Redis recovers — the protection survived the blip
    async def real_add(key, value, ttl=None, namespace=None):
        k = (namespace, key)
        if k in store:
            raise ValueError("key exists")
        store[k] = value
        return True

    monkeypatch.setattr(ai_mod.cache, "add", real_add)
    r3 = asyncio.run(ai_mod.create_ai_call(_ctx("s-open"), "U", "S", "F", "t", SPECIES))
    assert "already" in r3.lower()
    assert direct["n"] == 2, "the guard was voided: a third write reached the API"


def test_direct_path_timeout_propagates(monkeypatch):
    """Timeout handling on the direct path is live production behaviour: it
    still propagates out of the tool."""
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)

    async def timeout(request, token):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ai_mod, "create_ai_call_api", timeout)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-rt"), "U", "S", "F", "t", SPECIES))


def test_health_call_guard_stays_unconditional(monkeypatch):
    """health_call.py keeps an unconditional guard for a different contract —
    it must not get unified with the AI-call flag by this refactor."""
    import inspect

    from agents.tools import health_call as hc_mod

    src = inspect.getsource(hc_mod)
    assert "ai_call_booking_guard_enabled" not in src
    assert "enable_network" not in src
