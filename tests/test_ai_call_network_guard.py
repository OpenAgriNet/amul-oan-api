"""The Beckn network route must be as protected as the direct route.

`settings.enable_network` decides HOW an AI-call booking is executed, never
WHETHER it is protected. Before this, the network branch returned before both
protections the direct path applies:

  * the per-session idempotency guard (Redis SET NX, gated on
    AI_CALL_BOOKING_GUARD_ENABLED — prod runs it ON), so an OSS->managed
    fallback re-run could duplicate a booking and its SMS to a real farmer;
  * the moderation verdict block — a booking is IRREVERSIBLE, so it must not
    be written for a query moderation rejected.

These tests pin both, for the flag on AND off, plus the unchanged direct path.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

from agents.tools import ai_call as ai_mod
from agents.tools import beckn_network as bn_mod
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


def _patch_network(monkeypatch, ok=True):
    """Count confirms sent to the booking BPP. The count is the assertion that
    matters: every confirm is a real technician visit and a real SMS."""
    calls = {"n": 0}

    async def fake_confirm(union_code, society_code, farmer_code, user_id, species):
        calls["n"] += 1
        if not ok:
            return bn_mod.NetworkBookingResult(
                ok=False, ticket=None,
                message="Artificial insemination call booking failed on the network: society not serviced",
            )
        return bn_mod.NetworkBookingResult(
            ok=True, ticket=f"AICALL-{calls['n']}",
            message=f"Artificial insemination call booked successfully via the Beckn network. Ticket: AICALL-{calls['n']}",
        )

    # create_ai_call imports this lazily from the module, so patch it there.
    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", fake_confirm)
    return calls


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


def _network(monkeypatch, on):
    monkeypatch.setattr(ai_mod.settings, "enable_network", on)


def _guard(monkeypatch, on):
    monkeypatch.setattr(ai_mod.settings, "ai_call_booking_guard_enabled", on)


# --- guard ON, network ON -------------------------------------------------

def test_guard_on_network_on_refuses_second_booking_in_a_session(monkeypatch):
    """The re-run case: prod runs the guard ON, so the network route must
    short-circuit rather than send a second confirm (and a second SMS)."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-1"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-1"), "U", "S", "F", "tech1", SPECIES))

    assert calls["n"] == 1, "the booking BPP was confirmed more than once in one session"
    assert "booked successfully" in r1
    assert "already" in r2.lower()


def test_guard_on_network_on_books_once_under_concurrency(monkeypatch):
    """Two concurrent submits race; the atomic reservation closes the window."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = {"n": 0}

    async def slow_confirm(*a, **k):
        calls["n"] += 1
        await asyncio.sleep(0.02)  # the window two requests race in
        return bn_mod.NetworkBookingResult(ok=True, ticket="X", message="booked successfully via network")

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", slow_confirm)

    async def go():
        return await asyncio.gather(
            ai_mod.create_ai_call(_ctx("s-net-race"), "U", "S", "F", "t", SPECIES),
            ai_mod.create_ai_call(_ctx("s-net-race"), "U", "S", "F", "t", SPECIES),
        )

    r1, r2 = asyncio.run(go())
    assert calls["n"] == 1
    assert any("already" in r.lower() for r in (r1, r2))


def test_guard_on_network_on_failed_booking_releases_the_reservation(monkeypatch):
    """A NACK means nothing was booked, so the farmer must be able to retry
    inside the TTL instead of being locked out for 30 minutes."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch, ok=False)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-nack"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-nack"), "U", "S", "F", "t", SPECIES))

    assert "failed" in r1.lower()
    assert calls["n"] == 2, "the retry never reached the network — reservation was not released"
    assert "already" not in r2.lower()


def test_guard_on_network_on_transport_error_releases_the_reservation(monkeypatch):
    """Same for a transport failure: no booking happened, so no lock-out."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def boom(*a, **k):
        attempts["n"] += 1
        raise RuntimeError("connection refused")

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", boom)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-boom"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-boom"), "U", "S", "F", "t", SPECIES))

    assert "failed" in r1.lower()          # surfaced, not raised into the agent
    assert attempts["n"] == 2
    assert "already" not in r2.lower()


# --- guard OFF, network ON ------------------------------------------------

def test_guard_off_network_on_allows_a_second_booking(monkeypatch):
    """main's default: two cows in heat is a real case. The flag is a product
    trade-off; routing through Beckn must not silently change it."""
    _network(monkeypatch, True)
    _guard(monkeypatch, False)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-2"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-2"), "U", "S", "F", "tech1", SPECIES))

    assert calls["n"] == 2
    assert "booked successfully" in r1
    assert "booked successfully" in r2
    assert "already" not in r2.lower()


# --- moderation -----------------------------------------------------------

def test_moderation_rejected_network_on_writes_no_booking(monkeypatch):
    """A booking is IRREVERSIBLE: a rejected query must not reach the BPP."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch)

    r = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-mod", in_scope=False), "U", "S", "F", "t", SPECIES))

    assert calls["n"] == 0, "a moderation-rejected query was booked on the network"
    assert "dairy farming" in r


def test_moderation_rejected_network_on_with_guard_off_also_writes_no_booking(monkeypatch):
    """The moderation block is not the idempotency guard — it applies whatever
    the guard flag says."""
    _network(monkeypatch, True)
    _guard(monkeypatch, False)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch)

    r = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-mod2", in_scope=False), "U", "S", "F", "t", SPECIES))

    assert calls["n"] == 0
    assert "dairy farming" in r


def test_moderation_rejected_network_off_writes_no_booking(monkeypatch):
    """Direct path, unchanged."""
    _network(monkeypatch, False)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_direct(monkeypatch)

    r = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-mod", in_scope=False), "U", "S", "F", "t", SPECIES))

    assert calls["n"] == 0
    assert "dairy farming" in r


# --- flag off: the direct path is untouched -------------------------------

def test_network_off_still_uses_the_direct_path_and_its_guard(monkeypatch):
    """No-op for current production: flag off books directly, guard ON refuses
    the re-run, and the network is never contacted."""
    _network(monkeypatch, False)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    direct = _patch_direct(monkeypatch)
    net = _patch_network(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-1"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-1"), "U", "S", "F", "tech1", SPECIES))

    assert net["n"] == 0, "the network was contacted with enable_network=false"
    assert direct["n"] == 1
    assert "booked successfully" in r1
    assert "already" in r2.lower()


def test_network_off_guard_off_allows_a_second_direct_booking(monkeypatch):
    """The default main behaviour, still exactly as before."""
    _network(monkeypatch, False)
    _guard(monkeypatch, False)
    _patch_cache(monkeypatch)
    direct = _patch_direct(monkeypatch)
    net = _patch_network(monkeypatch)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-2"), "U", "S", "F", "tech1", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-2"), "U", "S", "F", "tech1", SPECIES))

    assert net["n"] == 0
    assert direct["n"] == 2
    assert "booked successfully" in r1 and "booked successfully" in r2


def test_health_call_guard_stays_unconditional(monkeypatch):
    """health_call.py keeps an unconditional guard for a different contract —
    it must not get unified with the AI-call flag by this refactor."""
    import inspect

    from agents.tools import health_call as hc_mod

    src = inspect.getsource(hc_mod)
    assert "ai_call_booking_guard_enabled" not in src
    assert "enable_network" not in src
