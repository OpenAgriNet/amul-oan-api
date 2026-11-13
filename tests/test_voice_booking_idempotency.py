"""Voice booking tools must be idempotent per session so an agent re-run (the
OSS->managed streaming fallback re-executes tool calls) cannot double-book.
Voice books through chat's Beckn confirm, so a reservation is released only when
the booking provably did not happen."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from agents.tools import ai_call as chat_ai_call
from agents.tools.beckn import network as beckn_network
from agents.tools.beckn import operations
from agents.voice.tools import ai_call as ai_mod
from agents.voice.models.ai_call import AISpecies
from app.core import cache as cache_mod
from tests.test_beckn_operations import MemoryRedis, _configure_amul

# create_ai_call rejects identifiers that cannot be real; 24 base64 chars is the
# shape of every real prod technician id.
TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="
SPECIES = next(iter(AISpecies))


def _ctx(session_id, tool_call_id=None, superseded=False):
    async def _ensure_in_scope():
        return True

    async def _superseded():
        return superseded

    return SimpleNamespace(
        deps=SimpleNamespace(session_id=session_id, ensure_in_scope=_ensure_in_scope, superseded=_superseded),
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


def _book_ai(session_id="s1", tool_call_id=None, species=SPECIES):
    return asyncio.run(
        ai_mod.create_ai_call(_ctx(session_id, tool_call_id), "159", "00002", "5058", TECH_ID, species)
    )


def test_ai_call_idempotent_on_rerun(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked(), _booked("T2")])

    r1 = _book_ai()
    r2 = _book_ai()

    assert len(calls) == 1
    assert "booked successfully" in r1.lower()
    # The repeat reads back the booking it hit, not a failure (AMUL-106).
    assert r2 == "Artificial insemination call is already booked. Ticket: T1."
    assert store[("ai_call_booked", "s1")] == {
        "ticket": "T1", "species": SPECIES.value, "account": "159/00002/5058"
    }


NOTHING_NEW_BOOKED = (
    "No new booking was made: only one artificial insemination booking can be made "
    "per session. This session's booking is for the cow (ticket T1); "
    "for any other animal, please try again later or contact your society."
)


def test_ai_call_rerun_for_the_other_species_is_told_nothing_was_booked(monkeypatch, store):
    """One booking per session: a buffalo asked for after the cow was booked must
    not hear the cow's ticket as its own booking."""
    calls = _confirms(monkeypatch, [_booked(), _booked("T2")])

    _book_ai(species=AISpecies.COW)
    r2 = _book_ai(species=AISpecies.BUFFALO)

    assert len(calls) == 1
    assert r2 == NOTHING_NEW_BOOKED


def test_ai_call_rerun_for_another_farmer_record_is_told_nothing_was_booked(monkeypatch, store):
    """A caller with records in two villages: a cow on the second record is a
    different visit, not a retry of the first booking."""
    calls = _confirms(monkeypatch, [_booked(), _booked("T2")])

    _book_ai(species=AISpecies.COW)
    r2 = asyncio.run(
        ai_mod.create_ai_call(_ctx("s1"), "159", "00777", "9999", TECH_ID, AISpecies.COW)
    )

    assert len(calls) == 1
    assert r2 == NOTHING_NEW_BOOKED


def test_ai_call_sends_the_callers_codes_and_tool_call_id(monkeypatch, store):
    calls = _confirms(monkeypatch, [_booked()])

    _book_ai(session_id="s1", tool_call_id="call-7")

    (args, kwargs), = calls
    assert args == ("159", "00002", "5058", TECH_ID, SPECIES.value)
    assert kwargs == {"session_id": "s1", "tool_call_id": "call-7"}


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


# ── a repeat that lands while the booking is still being made (#566) ────────


def _quick_wait(monkeypatch, seconds=1.0):
    monkeypatch.setattr(chat_ai_call, "_INFLIGHT_WAIT_SECONDS", seconds)
    monkeypatch.setattr(chat_ai_call, "_INFLIGHT_POLL_SECONDS", 0.01)


def _overlapping():
    """Raya sends one "yes" as two fragments: the second runs the booking tool
    while the first one's booking is still in flight."""
    async def go():
        first = asyncio.create_task(
            ai_mod.create_ai_call(_ctx("sR"), "159", "00002", "5058", TECH_ID, SPECIES)
        )
        await asyncio.sleep(0.02)
        second = await ai_mod.create_ai_call(_ctx("sR"), "159", "00002", "5058", TECH_ID, SPECIES)
        return await first, second

    return asyncio.run(go())


def test_a_repeat_that_lands_while_the_booking_is_in_flight_gets_the_ticket(monkeypatch, store):
    """The first fragment's turn books but is superseded, so its reply never
    reaches the caller. The second must read the ticket back, not say the
    booking may have gone through."""
    _quick_wait(monkeypatch)
    calls = _confirms(monkeypatch, [_booked("T1"), _booked("T2")], delay=0.2)

    first, second = _overlapping()

    assert len(calls) == 1
    assert "booked successfully" in first.lower()
    assert second == "Artificial insemination call is already booked. Ticket: T1."


@pytest.mark.parametrize("outcome", [
    _pending(),
    httpx.ReadTimeout("read timed out"),
])
def test_a_repeat_waiting_on_an_unconfirmed_booking_answers_once_it_settles(monkeypatch, store, outcome):
    """The booking was neither confirmed nor refused: the waiting repeat says it
    may have gone through as soon as that is known, not after the full wait,
    and the session stays held."""
    _quick_wait(monkeypatch, seconds=5.0)
    calls = _confirms(monkeypatch, [outcome, _booked()], delay=0.2)

    started = time.monotonic()
    first, second = _overlapping()
    elapsed = time.monotonic() - started

    assert len(calls) == 1
    assert first == chat_ai_call.UNCONFIRMED_MESSAGE
    assert second == chat_ai_call.ALREADY_REQUESTED_MESSAGE
    assert elapsed < 2.0
    assert store[("ai_call_booked", "sR")] == {
        "ticket": None, "species": SPECIES.value, "account": "159/00002/5058"
    }


@pytest.mark.parametrize("outcome", [
    _nack(),
    httpx.ConnectError("connection refused"),
])
def test_a_repeat_waiting_on_a_booking_that_provably_failed_is_told_nothing_was_booked(
    monkeypatch, store, outcome
):
    _quick_wait(monkeypatch)
    calls = _confirms(monkeypatch, [outcome, _booked()], delay=0.2)

    first, second = _overlapping()

    assert len(calls) == 1
    assert "booked successfully" not in first.lower()
    assert second == chat_ai_call.EARLIER_ATTEMPT_FAILED_MESSAGE
    assert ("ai_call_booked", "sR") not in store


def test_a_repeat_stops_waiting_when_the_wait_runs_out(monkeypatch, store):
    """A booking still running after the wait: the repeat says it may have gone
    through, and the visit is still booked only once."""
    _quick_wait(monkeypatch, seconds=0.1)
    calls = _confirms(monkeypatch, [_booked("T1"), _booked("T2")], delay=0.5)

    first, second = _overlapping()

    assert len(calls) == 1
    assert "booked successfully" in first.lower()
    assert second == chat_ai_call.ALREADY_REQUESTED_MESSAGE


# ── a request that a newer one has taken over (#566) ────────────────────────


def test_a_superseded_request_does_not_book_and_leaves_the_call_free(monkeypatch, store):
    """The caller kept talking after "yes": the older request's reply will not
    be heard, so it books nothing, and the newer request can still book."""
    calls = _confirms(monkeypatch, [_booked("T1")])

    told = asyncio.run(
        ai_mod.create_ai_call(_ctx("s1", superseded=True), "159", "00002", "5058", TECH_ID, SPECIES)
    )

    assert told == ai_mod.SUPERSEDED_MESSAGE
    assert calls == []
    assert ("ai_call_booked", "s1") not in store
    assert "booked successfully" in _book_ai().lower()


# ── a booking whose reply never reached the caller (#566) ───────────────────


def test_the_next_turn_is_told_a_booking_the_caller_never_heard(monkeypatch, store):
    """The request that booked was taken over, so nothing of its turn is in the
    call's history: the next turn is given the ticket to tell."""
    _confirms(monkeypatch, [_booked("T1")])
    _book_ai(species=AISpecies.BUFFALO)

    note = asyncio.run(chat_ai_call.unheard_booking_note("s1", []))

    assert note == chat_ai_call.UNHEARD_BOOKING_NOTE.format(species="buffalo", ticket="T1")


def test_a_booking_the_history_already_has_is_not_told_again(monkeypatch, store):
    _confirms(monkeypatch, [_booked("T1")])
    told = _book_ai()
    history = [ModelRequest(parts=[ToolReturnPart(tool_name="create_ai_call", content=told)])]

    assert asyncio.run(chat_ai_call.unheard_booking_note("s1", history)) is None


def test_a_ticket_inside_a_longer_number_is_not_one_told(monkeypatch, store):
    _confirms(monkeypatch, [_booked("1234")])
    _book_ai()
    history = [ModelRequest(parts=[UserPromptPart(content='**User:** "my number is 9812345670, farmer code 51234"')])]

    assert asyncio.run(chat_ai_call.unheard_booking_note("s1", history)) is not None


@pytest.mark.parametrize("marker", [
    None,
    True,
    {"ticket": None, "species": "cow", "account": "159/00002/5058"},
], ids=["no_booking", "in_flight", "unconfirmed"])
def test_there_is_nothing_to_tell_without_a_ticket(store, marker):
    if marker is not None:
        store[("ai_call_booked", "s1")] = marker

    assert asyncio.run(chat_ai_call.unheard_booking_note("s1", [])) is None


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


# ── through the real Beckn confirm ──────────────────────────────────────────


def _nack_body(code):
    return {"message": {"ack": {"status": "NACK"}}, "error": {"code": code, "message": code.lower()}}


_BOOKED = ("ack", {"message": {"order": {"id": "T9", "state": "ACTIVE"}}})


def _on_confirm_error(code, message):
    return ("ack", {"error": {"code": code, "message": message}})


class _Bridge:
    """The transaction bridge as the BAP sees it. Each confirm gets the next
    answer: an exception to raise, an (HTTP status, body) pair, or ("ack", the
    on_confirm the BPP sends back, or None for none in time)."""

    def __init__(self, store, answers):
        self.store = store
        self.answers = list(answers)
        self.posted = []

    async def post(self, url, json=None, headers=None):
        self.posted.append(json)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        request = httpx.Request("POST", url)
        status, body = answer
        if status == "ack":
            if body is not None:
                context = {**json["context"], "action": "on_confirm"}
                assert (await self.store.record_callback({"context": context, **body})).accepted
            return httpx.Response(200, json={"message": {"ack": {"status": "ACK"}}}, request=request)
        return httpx.Response(status, json=body, request=request)


class _Redis(MemoryRedis):
    """The operation store's Redis, failing once where a test asks it to."""

    fail_on = None

    async def set(self, key, value, ex=None, nx=False):
        if self.fail_on == "create":
            self.fail_on = None
            raise ConnectionError("redis went away")
        return await super().set(key, value, ex=ex, nx=nx)

    async def hset(self, key, mapping=None):
        if self.fail_on == "mark_sent" and "sent_at" in (mapping or {}):
            self.fail_on = None
            raise ConnectionError("redis went away")
        return await super().hset(key, mapping=mapping)


@pytest.fixture
def network(monkeypatch, store):
    _configure_amul(monkeypatch, operations)
    monkeypatch.setattr(operations.settings, "beckn_callback_wait_seconds", 0.05)
    monkeypatch.setattr(operations.settings, "beckn_callback_poll_interval_seconds", 0.01)
    monkeypatch.setattr(operations.settings, "beckn_forward_retry_delay_seconds", 0.0)
    redis = _Redis()
    beckn_store = operations.BecknOperationStore(redis, ttl_seconds=3600)

    def answer(*answers):
        bridge = _Bridge(beckn_store, answers)
        client = operations.BecknOperationClient(beckn_store, http_client=bridge)
        monkeypatch.setattr(operations, "_operation_client", client)
        return bridge

    return SimpleNamespace(answer=answer, redis=redis)


def _unconfigured(monkeypatch, network):
    monkeypatch.setattr(operations.settings, "beckn_bap_caller_url", "")
    return lambda: monkeypatch.setattr(operations.settings, "beckn_bap_caller_url", "http://onix/bap/caller")


def _store_fails(step):
    def breaks(monkeypatch, network):
        network.redis.fail_on = step
        return lambda: None
    return breaks


@pytest.mark.parametrize("breaks, first", [
    (_unconfigured, []),
    (_store_fails("create"), []),
    (_store_fails("mark_sent"), []),
    (None, [httpx.ConnectError("refused"), httpx.ConnectError("refused")]),
    (None, [(401, _nack_body("UNAUTHORIZED_TRANSACTION"))]),
    (None, [(404, _nack_body("UNSUPPORTED_TRANSACTION"))]),
    (None, [_on_confirm_error("BOOKING_REJECTED", "Invalid Union Code")]),
], ids=["not_configured", "not_recorded", "not_marked_sent", "connect_refused", "bad_bridge_token", "unsupported_action", "provider_rejected"])
def test_a_confirm_that_provably_did_not_book_frees_the_call_to_book_again(monkeypatch, network, breaks, first):
    bridge = network.answer(*first, _BOOKED)
    undo = breaks(monkeypatch, network) if breaks else None

    failed = _book_ai()
    if undo:
        undo()
    retried = _book_ai()

    assert failed.startswith("Artificial insemination call booking failed.")
    assert "booked successfully" in retried.lower()
    assert len(bridge.posted) == len(first) + 1


@pytest.mark.parametrize("answer", [
    httpx.ReadTimeout("read timed out"),
    (502, _nack_body("ONIX_CALLER_UNAVAILABLE")),
    (503, _nack_body("BPP_STORAGE_UNAVAILABLE")),
    ("ack", None),
    ("ack", {"error": {"code": "BOOKING_UNCONFIRMED", "message": "do not submit a duplicate"}}),
], ids=["read_timeout", "bridge_502", "bpp_503", "no_callback_in_time", "bpp_could_not_confirm"])
def test_a_confirm_that_may_have_booked_holds_the_call_and_says_it_may_have(network, answer):
    """The visit may be booked and the farmer texted: neither the first answer nor
    a retry in the same call may say it failed or that a booking is active."""
    bridge = network.answer(answer, _BOOKED)

    first = _book_ai()
    retried = _book_ai()

    assert len(bridge.posted) == 1
    assert first == chat_ai_call.UNCONFIRMED_MESSAGE
    assert retried == chat_ai_call.ALREADY_REQUESTED_MESSAGE


# ── chat's route: the same confirm, guard from AI_CALL_BOOKING_GUARD_ENABLED ──


def _book_chat(session_id="s1"):
    from agents.tools.models.ai_call import AISpecies as ChatSpecies

    return asyncio.run(chat_ai_call._book_via_network(
        "159", "00002", "5058", TECH_ID, ChatSpecies.COW, session_id, {},
    ))


@pytest.mark.parametrize("reason, told", [
    ("Invalid Union Code", "The booking system did not accept the request (Invalid Union Code), so no visit was booked."),
    ("The provider rejected the booking (HTTP 409)", "The booking system did not accept the request, so no visit was booked."),
    ("See https://example.invalid/e/1", "The booking system did not accept the request, so no visit was booked."),
], ids=["plain_reason", "status_code_dropped", "url_dropped"])
def test_chat_a_provider_rejection_says_it_failed_and_frees_the_session(monkeypatch, network, store, reason, told):
    """PashuGPT answered no: nothing was booked, so the farmer may book again now."""
    monkeypatch.setattr(chat_ai_call.settings, "ai_call_booking_guard_enabled", True)
    bridge = network.answer(_on_confirm_error("BOOKING_REJECTED", reason), _BOOKED)

    failed = _book_chat()
    held_after_failure = ("ai_call_booked", "s1") in store
    retried = _book_chat()

    assert failed.startswith("Artificial insemination call booking failed.\n\n" + told)
    assert not held_after_failure
    assert "booked successfully" in retried.lower()
    assert len(bridge.posted) == 2


def test_chat_any_other_provider_error_still_holds_the_session(monkeypatch, network, store):
    monkeypatch.setattr(chat_ai_call.settings, "ai_call_booking_guard_enabled", True)
    bridge = network.answer(_on_confirm_error("BOOKING_UNCONFIRMED", "do not submit a duplicate"), _BOOKED)

    first = _book_chat()
    retried = _book_chat()

    assert first == chat_ai_call.UNCONFIRMED_MESSAGE
    assert retried == chat_ai_call.ALREADY_REQUESTED_MESSAGE
    assert len(bridge.posted) == 1
