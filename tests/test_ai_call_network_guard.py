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

import httpx
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
            # Shape of a real NACK: the BPP states it did not accept the order,
            # so nothing was booked and nothing was texted.
            return bn_mod.NetworkBookingResult(
                ok=False, ticket=None,
                message="Artificial insemination call booking failed on the network: society not serviced",
                authoritative_no_booking=True,
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
    """An AUTHORITATIVE NACK means nothing was booked, so the farmer must be
    able to retry inside the TTL instead of being locked out for 30 minutes.
    This is the only failure shape that earns a release."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    calls = _patch_network(monkeypatch, ok=False)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-nack"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-nack"), "U", "S", "F", "t", SPECIES))

    assert "failed" in r1.lower()
    assert calls["n"] == 2, "the retry never reached the network — reservation was not released"
    assert "already" not in r2.lower()


def test_guard_on_network_on_pre_send_transport_error_releases_the_reservation(monkeypatch):
    """A connection that was never established is provably pre-send: the confirm
    never reached the BPP, so nothing was booked and there must be no lock-out."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def boom(*a, **k):
        attempts["n"] += 1
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", boom)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-boom"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-boom"), "U", "S", "F", "t", SPECIES))

    assert "failed" in r1.lower()          # surfaced, not raised into the agent
    assert attempts["n"] == 2, "the retry never reached the network — reservation was not released"
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


# --- a reservation you do not own is not yours to release -----------------

def test_redis_error_fail_open_does_not_delete_an_earlier_booking_marker(monkeypatch):
    """`reserve` fails OPEN: when Redis is down it lets the booking through
    WITHOUT holding the key. The old code recorded that as "reserved", so the
    next failure deleted a key it never wrote — here, the marker written by this
    session's EARLIER SUCCESSFUL booking — leaving the session unguarded for the
    rest of the TTL.

    Direct path on purpose: enable_network=false is what production runs.
    """
    _network(monkeypatch, False)
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


# --- ambiguous failures must not release (network path only) --------------

def test_read_timeout_holds_the_reservation_and_refuses_an_immediate_retry(monkeypatch):
    """A read timeout happens AFTER the confirm was sent. The BPP may already
    have called PashuGPT and texted the farmer, so a retry must be refused: we
    prefer a possible ~30-min lockout over a possible second visit + second SMS."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def timeout(*a, **k):
        attempts["n"] += 1
        raise httpx.ReadTimeout("timed out reading response")

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", timeout)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-rt"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-rt"), "U", "S", "F", "t", SPECIES))

    assert "could not be confirmed" in r1.lower(), "the farmer was told a flat 'failed'"
    assert "booked successfully" not in r1.lower()
    assert attempts["n"] == 1, "the reservation was released, so a retry re-sent the confirm"
    assert "already" in r2.lower()


def test_gateway_5xx_holds_the_reservation(monkeypatch):
    """A mid-chain 502/504 is equally ambiguous — the booking BPP may have
    completed and the gateway lost the response."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def bad_gateway(*a, **k):
        attempts["n"] += 1
        request = httpx.Request("POST", "http://bpp.invalid/confirm")
        raise httpx.HTTPStatusError(
            "502", request=request, response=httpx.Response(502, request=request)
        )

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", bad_gateway)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-502"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-502"), "U", "S", "F", "t", SPECIES))

    assert "could not be confirmed" in r1.lower()
    assert attempts["n"] == 1
    assert "already" in r2.lower()


def test_connect_timeout_is_pre_send_and_releases(monkeypatch):
    """The one timeout that IS provably pre-send: no connection was ever
    established, so nothing was booked and the farmer may retry at once."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    attempts = {"n": 0}

    async def connect_timeout(*a, **k):
        attempts["n"] += 1
        raise httpx.ConnectTimeout("timed out connecting")

    monkeypatch.setattr(bn_mod, "network_create_ai_call_result", connect_timeout)

    asyncio.run(ai_mod.create_ai_call(_ctx("s-net-ct"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-ct"), "U", "S", "F", "t", SPECIES))

    assert attempts["n"] == 2, "a provably pre-send failure kept the farmer locked out"
    assert "already" not in r2.lower()


def test_direct_path_timeout_behaviour_is_unchanged(monkeypatch):
    """The pre-send/ambiguous split is NETWORK-ONLY by design. The direct path's
    handling of a raised timeout is live production behaviour and out of scope:
    it still propagates out of the tool exactly as on main."""
    _network(monkeypatch, False)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)

    async def timeout(request, token):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ai_mod, "create_ai_call_api", timeout)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(ai_mod.create_ai_call(_ctx("s-dir-rt"), "U", "S", "F", "t", SPECIES))


# --- a 200 with an unparseable body is not a booking ----------------------

def test_unparseable_200_is_not_success_and_holds_the_reservation(monkeypatch):
    """End-to-end through the real Beckn client: `200 {}` must not tell the
    farmer "booked successfully. Ticket: None". It is a failure, and — being
    non-authoritative — it holds the reservation just like a read timeout."""
    _network(monkeypatch, True)
    _guard(monkeypatch, True)
    _patch_cache(monkeypatch)
    posts = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            posts["n"] += 1
            return _Resp()

    monkeypatch.setattr(bn_mod.httpx, "AsyncClient", _Client)

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-empty"), "U", "S", "F", "t", SPECIES))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s-net-empty"), "U", "S", "F", "t", SPECIES))

    assert "booked successfully" not in r1.lower()
    assert "ticket: none" not in r1.lower()
    assert "could not be confirmed" in r1.lower()
    assert posts["n"] == 1, "the reservation was released on an unconfirmed booking"
    assert "already" in r2.lower()


def test_health_call_guard_stays_unconditional(monkeypatch):
    """health_call.py keeps an unconditional guard for a different contract —
    it must not get unified with the AI-call flag by this refactor."""
    import inspect

    from agents.tools import health_call as hc_mod

    src = inspect.getsource(hc_mod)
    assert "ai_call_booking_guard_enabled" not in src
    assert "enable_network" not in src
