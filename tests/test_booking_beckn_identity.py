from types import SimpleNamespace
import asyncio
import json
from uuid import uuid4

import pytest

from agents.tools.beckn import amul as beckn_amul
from agents.tools import ai_call
from agents.tools.beckn import network as beckn_network
from agents.tools.models.ai_call import AISpecies, BookingOutcome
from app.agui.models import ChatRunRequest
from app.agui.service import advance, new_record, MilkStore, UIError, events

# create_ai_call rejects identifiers that cannot be real; every real prod
# technician id is 24 base64 chars ending "==".
TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="



async def _in_scope():
    return True


def _ctx():
    return SimpleNamespace(
        deps=SimpleNamespace(
            session_id=None,
            mobile="9000000000",
            farmer_unions=["kaira"],
            farmer_profile_status="found",
            ensure_in_scope=_in_scope,
        ),
        tool_call_id="tool-1",
    )


def _callback_mode(monkeypatch, module):
    _ = monkeypatch, module


@pytest.mark.asyncio
async def test_ai_confirm_uses_canonical_owned_account_and_discovered_technician(monkeypatch):
    _callback_mode(monkeypatch, ai_call)
    monkeypatch.setattr(ai_call.settings, "ai_call_booking_guard_enabled", False)
    account = beckn_amul.AuthenticatedFarmerAccount("CANON-U", "CANON-S", "CANON-F")

    async def resolve(mobile, **kwargs):
        assert mobile == "9000000000"
        assert kwargs["union_code"] == "MODEL-U"
        return account

    async def technicians(**kwargs):
        assert kwargs["union_code"] == "CANON-U"
        assert kwargs["force_refresh"] is True
        return [beckn_amul.AITechnicianRecord(userId=TECH_ID, fullName="Technician")]

    captured = {}
    async def confirm(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return beckn_network.NetworkBookingResult(True, "TICKET", "booked successfully")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_amul, "search_ai_technicians", technicians)
    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", confirm)

    result = await ai_call.create_ai_call(
        _ctx(), "MODEL-U", "MODEL-S", "MODEL-F", TECH_ID, AISpecies.COW
    )

    assert "booked successfully" in result
    assert captured["args"][:5] == ("CANON-U", "CANON-S", "CANON-F", TECH_ID, "cow")


@pytest.mark.asyncio
async def test_ai_confirm_is_not_sent_for_unowned_account(monkeypatch):
    _callback_mode(monkeypatch, ai_call)
    monkeypatch.setattr(ai_call.settings, "ai_call_booking_guard_enabled", False)

    async def resolve(*args, **kwargs):
        return None

    async def confirm(*args, **kwargs):
        raise AssertionError("confirm must not be sent for an unowned account")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", confirm)

    result = await ai_call.create_ai_call(
        _ctx(), "MODEL-U", "MODEL-S", "MODEL-F", TECH_ID, AISpecies.COW
    )

    assert "does not belong" in result


# AG-UI booking uses the same identity/booking adapter, with explicit controls.
def _request(component=None, *, command=None):
    props = {"version": 1, "target_lang": "gu"}
    if component:
        props["ui_action"] = dict(component_id=component["id"], revision=component["revision"],
            kind=component["kind"], value={"confirmed": True} if component["kind"] == "confirm" else
            {"option_id": component["options"][0]["id"]})
    if command:
        props["command"] = command
    return ChatRunRequest.model_validate(dict(threadId="booking-thread", runId=uuid4().hex,
        state={}, tools=[], context=[], messages=[] if component or command else
        [dict(id=uuid4().hex, role="user", content="Book an AI visit")], forwardedProps=props))


def _farmer(code="F"):
    return SimpleNamespace(union_code="U", society_code="S", farmer_code=code, union_name="kaira",
        display_farmer_name="Farmer", display_society_name="Society")


class _BookingProvider:
    def __init__(self):
        self.profiles = [_farmer()]
        self.available = [beckn_amul.AITechnicianRecord(userId=TECH_ID, fullName="Technician", mobileNumber="9000000001")]
        self.calls = []
        self.result = BookingOutcome(status="confirmed", message="Booked. Ticket: T1", ticket="T1")

    async def farmers(self, *args):
        return self.profiles

    async def technicians(self, *args):
        return self.available

    async def book(self, booking, session_id):
        self.calls.append(booking)
        return self.result


async def _advance(record, request, provider, **kwargs):
    return await advance(record, request, "9000000000", None, booking_provider=provider, **kwargs)


async def _confirmation(provider, species="cow"):
    record, response = await _advance(new_record(), _request(), provider, booking_intent={"species": species})
    technician = response["state"]["components"][-1]
    record, response = await _advance(record, _request(technician), provider)
    if species is None:
        record, response = await _advance(record, _request(response["state"]["components"][-1]), provider)
    return record, response["state"]["components"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("species", [None, "cow", "buffalo"])
async def test_agui_requires_technician_and_only_asks_species_when_missing(species):
    provider = _BookingProvider()
    record, response = await _advance(new_record(), _request(), provider, booking_intent={"species": species})
    technician = response["state"]["components"][-1]
    assert technician["purpose"] == "ai_technician"  # Even a sole technician requires a pick.
    assert technician["options"][0]["label"] == "Technician"
    assert TECH_ID not in json.dumps(response["state"])
    record, response = await _advance(record, _request(technician), provider)
    next_control = response["state"]["components"][-1]
    if species is None:
        assert next_control["purpose"] == "ai_species"
        assert [o["id"] for o in next_control["options"]] == ["cow", "buffalo"]
        record, response = await _advance(record, _request(next_control), provider)
        next_control = response["state"]["components"][-1]
    assert next_control["kind"] == "confirm"
    assert next_control["summary"]["species"] == (species or "cow")
    assert not provider.calls
    assert not any(c["purpose"] == "milk_account" for c in response["state"]["components"])
    assert [event.type for event in events(_request(), response)][-1] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_agui_reuses_owned_selection_but_stops_on_ambiguous_or_removed_account():
    provider = _BookingProvider()
    provider.profiles.append(_farmer("F2"))
    initial = new_record()
    _, response = await _advance(initial, _request(), provider, booking_intent={"species": None})
    assert response["error"]["error"]["code"] == "BOOKING_CONTEXT_UNAVAILABLE"
    initial["options"]["owned"] = dict(identity=["U", "S", "F2"], label="Second account")
    initial["ui"]["selected_milk_account"] = dict(option_id="owned", label="Second account")
    record, response = await _advance(initial, _request(), provider, booking_intent={"species": "buffalo"})
    assert record["booking"]["identity"] == ["U", "S", "F2"]
    provider.profiles = [_farmer()]
    _, response = await _advance(initial, _request(), provider, booking_intent={"species": None})
    assert response["error"]["error"]["code"] == "BOOKING_CONTEXT_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["ownership", "union", "technicians", "label"])
async def test_agui_revalidates_before_confirmation_and_never_sends_changed_details(change):
    provider = _BookingProvider()
    record, control = await _confirmation(provider)
    if change == "ownership":
        provider.profiles = [_farmer("other")]
    elif change == "union":
        provider.profiles[0].union_name = "kutch"
    elif change == "technicians":
        provider.available = []
    else:
        provider.available[0].fullName = "Updated name"
    updated, response = await _advance(record, _request(control), provider)
    assert not response.get("booking_send")
    assert not provider.calls
    assert updated["ui"]["components"][-1]["status"] != "submitted"
    if change == "label":
        assert updated["ui"]["components"][-1]["summary"]["technician"]["label"] == "Updated name"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["technician", "species"])
async def test_agui_edit_expires_confirmation_and_requires_another_review(field):
    provider = _BookingProvider()
    record, old = await _confirmation(provider)
    command = dict(kind="edit_ai_booking", workflow_id=record["booking"]["id"],
                   state_revision=record["ui"]["state_revision"], field=field)
    record, response = await _advance(record, _request(command=command), provider)
    assert record["ui"]["components"][-1]["purpose"] == "ai_" + field
    with pytest.raises(UIError) as error:
        await _advance(record, _request(old), provider)
    assert error.value.status == 409
    assert not provider.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["option", "revision", "kind"])
async def test_agui_rejects_forged_or_stale_picker_submissions(invalid):
    provider = _BookingProvider()
    record, response = await _advance(new_record(), _request(), provider, booking_intent={"species": None})
    request = _request(response["state"]["components"][-1]).model_dump()
    action = request["forwardedProps"]["ui_action"]
    if invalid == "option":
        action["value"]["option_id"] = "not-a-displayed-option"
    elif invalid == "revision":
        action["revision"] += 1
    else:
        action.update(kind="confirm", value={"confirmed": True})
    with pytest.raises(UIError) as error:
        await _advance(record, ChatRunRequest.model_validate(request), provider)
    assert error.value.status == (409 if invalid == "revision" else 422)
    assert record["ui"]["components"][-1]["status"] == "pending"
    assert not provider.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("via_command", [False, True])
async def test_agui_cancel_does_not_book(via_command):
    provider = _BookingProvider()
    record, control = await _confirmation(provider)
    request = _request(control)
    if via_command:
        request = _request(command=dict(kind="cancel_ai_booking", workflow_id=record["booking"]["id"],
                                       state_revision=record["ui"]["state_revision"]))
    else:
        request.forwardedProps.ui_action.value.confirmed = False
    record, response = await _advance(record, request, provider)
    assert record["booking"]["stage"] == "cancelled"
    assert not provider.calls and not response.get("booking_send")


@pytest.mark.asyncio
async def test_agui_post_confirms_once_and_streams_the_provider_ticket(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from fakeredis.aioredis import FakeRedis
    from app.routers import agui

    provider, redis = _BookingProvider(), FakeRedis(decode_responses=True)
    store = MilkStore(redis, "test:", 3600, booking_provider=provider)
    record, control = await _confirmation(provider)
    await redis.set(store.key("owner", "farmer", "booking-thread"), json.dumps(record))
    for setting, value in (("environment", "development"), ("chat_agui_preview_enabled", True), ("chat_agui_enabled", True)):
        monkeypatch.setattr(agui.settings, setting, value)
    app = FastAPI()
    app.include_router(agui.router, prefix="/api")
    app.dependency_overrides[agui.get_milk_store] = lambda: store
    app.dependency_overrides[agui.get_current_user] = lambda: dict(sub="owner", phone="9000000000")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            request = _request(control).model_dump(mode="json")
            response = await client.post("/api/chat/", json=request)
            assert response.status_code == 200
            assert response.headers["x-chat-format"] == "agui-v1"
            emitted = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            assert emitted[-1]["type"] == "RUN_FINISHED"
            snapshot = next(e["snapshot"]["amul_ui"] for e in emitted if e["type"] == "STATE_SNAPSHOT")
            assert snapshot["components"][-1]["outcome"]["ticket"] == "T1"
            assert (await client.post("/api/chat/", json=request)).status_code == 409
            assert len(provider.calls) == 1
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_agui_ordinary_messages_and_booking_do_not_expire_milk_controls():
    provider = _BookingProvider()
    initial = new_record()
    initial["ui"]["components"].append(dict(id="milk", workflow_id="milk-flow", message_id="m", revision=1,
        title="Dates", status="pending", kind="date_range", purpose="milk_period", selected=None,
        constraints=dict(date_format="YYYY-MM-DD", max_difference_days=31)))
    record, control = await _confirmation(provider)
    record["ui"]["components"].insert(0, initial["ui"]["components"][0])
    record, _ = await _advance(record, _request(), provider)
    assert record["ui"]["components"][-1]["status"] == "pending"
    record, _ = await _advance(record, _request(), provider, booking_intent={"species": None})
    assert record["ui"]["components"][0]["status"] == "pending"
    assert next(c for c in record["ui"]["components"] if c["id"] == control["id"])["status"] == "expired"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["confirmed", "rejected", "not_sent", "unconfirmed"])
async def test_agui_store_consumes_confirmation_before_send_and_rejects_replays(outcome):
    from fakeredis.aioredis import FakeRedis

    provider, redis = _BookingProvider(), FakeRedis(decode_responses=True)
    store = MilkStore(redis, "test:", 3600, booking_provider=provider)
    key = store.key("owner", "farmer", "booking-thread")
    record, control = await _confirmation(provider)
    await redis.set(key, json.dumps(record))
    request = _request(control)

    async def book(booking, session_id):
        claimed = json.loads(await redis.get(key))
        assert claimed["booking"]["stage"] == "submitting"
        assert request.runId in claimed["runs"]
        assert claimed["ui"]["components"][-1]["selected"] == {"confirmed": True}
        provider.calls.append(booking)
        return BookingOutcome(status=outcome, message=outcome, ticket="T1" if outcome == "confirmed" else None)

    provider.book = book
    try:
        response = await store.apply(request, "owner", "farmer", "9000000000")
        assert response["state"]["components"][-1]["outcome"]["status"] == outcome
        for replay in (request, _request(control)):
            with pytest.raises(UIError):
                await store.apply(replay, "owner", "farmer", "9000000000")
        assert len(provider.calls) == 1
    finally:
        await redis.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["claim", "cancel_after_claim", "cancel_during_send", "disconnect_after_send", "lease", "result_commit", "concurrent", "unexpected"])
async def test_agui_store_never_resends_after_storage_failure_cancellation_or_lease_loss(failure, monkeypatch):
    from fakeredis.aioredis import FakeRedis

    provider, redis = _BookingProvider(), FakeRedis(decode_responses=True)
    store = MilkStore(redis, "test:", 3600, booking_provider=provider)
    key = store.key("owner", "farmer", "booking-thread")
    record, control = await _confirmation(provider)
    await redis.set(key, json.dumps(record))
    request = _request(control)
    original_pipeline = redis.pipeline
    commits = 0

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        execute = pipe.execute

        async def execute_with_failure():
            nonlocal commits
            commits += 1
            if (failure == "claim" and commits == 1) or (failure == "result_commit" and commits == 2):
                raise ConnectionError("Redis unavailable")
            result = await execute()
            if failure == "lease" and commits == 1:
                await redis.delete(key + ":lock")
            return result

        pipe.execute = execute_with_failure
        return pipe

    monkeypatch.setattr(redis, "pipeline", pipeline)

    async def disconnected():
        return (failure == "cancel_after_claim" and commits == 1) or (failure == "disconnect_after_send" and bool(provider.calls))

    async def book(booking, session_id):
        provider.calls.append(booking)
        if failure == "cancel_during_send":
            raise asyncio.CancelledError()
        if failure == "unexpected":
            raise RuntimeError("Unknown outcome after send")
        if failure == "concurrent":
            with pytest.raises(UIError) as error:
                await store.apply(_request(control), "owner", "farmer", "9000000000")
            assert error.value.body["error"]["code"] == "RUN_IN_PROGRESS"
        return provider.result

    provider.book = book
    try:
        if failure in ("concurrent", "disconnect_after_send", "unexpected"):
            response = await store.apply(request, "owner", "farmer", "9000000000", disconnected=disconnected)
            assert response["state"]["components"][-1]["outcome"]["status"] == ("unconfirmed" if failure == "unexpected" else "confirmed")
        else:
            with pytest.raises((UIError, asyncio.CancelledError)):
                await store.apply(request, "owner", "farmer", "9000000000", disconnected=disconnected)
        assert len(provider.calls) == (1 if failure in ("concurrent", "result_commit", "cancel_during_send", "disconnect_after_send", "unexpected") else 0)
        if failure != "claim":
            saved = json.loads(await redis.get(key))
            assert saved["ui"]["components"][-1]["status"] == "submitted"
            with pytest.raises(UIError):
                await store.apply(_request(control), "owner", "farmer", "9000000000")
        else:
            assert json.loads(await redis.get(key))["ui"]["components"][-1]["status"] == "pending"
    finally:
        await redis.aclose()
