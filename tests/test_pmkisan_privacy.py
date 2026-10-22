"""PM-KISAN status: what reaches Bharat Vistaar, and what never reaches anything else.

The farmer's Aadhaar and OTP must reach BV unchanged (gujarati digits included),
while the model, logs, traces and history only ever see placeholders.
"""

import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

import main  # noqa: F401 - registers the uvicorn access-log filter
from agents.tools import session_pmkisan, vistaar_pmkisan
from agents.tools.beckn.operations import (
    BecknActionResult,
    BecknOperation,
    OperationState,
)
from app.config import settings
from app.core.cache import build_cache_key
from app.models.requests import ChatRequest
from app.routers import chat as chat_router
from app.routers import telemetry as telemetry_router

# Synthetic, checksum-valid Aadhaar; written in Gujarati digits by the farmer.
AADHAAR = "234567890124"
AADHAAR_GU = "૨૩૪૫ ૬૭૮૯ ૦૧૨૪"
EAR_TAG = "102345678901"
MOBILE = "9876543210"


class _FakeRedis:
    """One keyspace behind both the aiocache `cache` and the raw `redis_client`."""

    def __init__(self):
        self.data = {}

    async def get(self, key, namespace=None):
        return copy.deepcopy(self.data.get(build_cache_key(key, namespace) if namespace else key))

    async def set(self, key, value, ttl=None, namespace=None):
        self.data[build_cache_key(key, namespace)] = copy.deepcopy(value)

    async def delete(self, key, namespace=None):
        self.data.pop(build_cache_key(key, namespace), None)

    async def eval(self, script, numkeys, key, window):
        """The INCR-and-expire counter script."""
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]


def _operation(state=OperationState.SUCCEEDED):
    return BecknOperation(
        operation_id="op", transaction_id="txn", message_id="msg", action="x",
        expected_callback="on_x", domain="schemes:vistaar", bap_id="bap", bpp_id="bpp",
        session_id="s1", tool_call_id=None, request_hash="h", idempotency_key="k",
        state=state,
    )


def _result(payload, state=OperationState.SUCCEEDED):
    return BecknActionResult(operation=_operation(state), payload=payload)


def _on_init(code, short_desc):
    return {"message": {"order": {"items": [{"id": "NA", "tags": [
        {"descriptor": {"code": code, "short_desc": short_desc}},
    ]}]}}}


OTP_SENT = _on_init("otp_status", "Request for OTP is sent.")
COMPLETED = {"message": {"order": {"id": "5823", "state": "COMPLETED", "fulfillments": [{
    "customer": {"person": {"name": "Test Farmer"}, "contact": {"phone": "09876543210"}},
    "state": {"descriptor": {"code": "completed", "long_desc": (
        "=== PM KISAN BENEFICIARY STATUS ===\n"
        "Registration Number - GJ123456789\n"
        "Last Installment Status - 22nd Installment payment done"
    )}},
}]}}}


class _FakeClient:
    def __init__(self, init_payload=OTP_SENT, status_payload=COMPLETED):
        self.init_payload = init_payload
        self.status_payload = status_payload
        self.calls = []

    async def init_pm_kisan_status(self, **kwargs):
        self.calls.append(("init", kwargs))
        return _result(self.init_payload)

    async def submit_pm_kisan_otp(self, **kwargs):
        self.calls.append(("status", kwargs))
        if self.status_payload is None:
            return _result(None, OperationState.TIMED_OUT_PENDING)
        return _result(self.status_payload)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "vistaar_pmkisan_enabled", True)
    monkeypatch.setattr(settings, "doctor_persona_enabled", False)
    store = _FakeRedis()
    monkeypatch.setattr(session_pmkisan, "cache", store)
    monkeypatch.setattr(session_pmkisan, "redis_client", store)
    fake = _FakeClient()
    monkeypatch.setattr(vistaar_pmkisan, "get_beckn_operation_client", lambda: fake)
    return fake


def _ctx(mobile=MOBILE):
    return SimpleNamespace(deps=SimpleNamespace(session_id="s1", mobile=mobile), tool_call_id=None)


async def _chat(monkeypatch, query):
    """What the turn receives after the chat router, as the farmer typed `query`."""
    seen = {}

    async def _stream(**kwargs):
        seen.update(kwargs)
        yield "ok"

    monkeypatch.setattr(chat_router, "stream_chat_messages", _stream)
    monkeypatch.setattr(chat_router, "_get_message_history", AsyncMock(return_value=[]))
    await chat_router.chat_endpoint(
        BackgroundTasks(),
        request=ChatRequest(query=query, session_id="s1", stream=False),
        user_info={"phone": MOBILE},
    )
    return seen["query"]


@pytest.mark.asyncio
async def test_aadhaar_and_otp_reach_bv_but_nothing_else(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    turn1 = await _chat(monkeypatch, f"મારો આધાર {AADHAAR_GU} છે, ગાય {EAR_TAG} ની PM-KISAN સ્થિતિ")
    assert turn1 == f"મારો આધાર [AADHAAR_0124] છે, ગાય {EAR_TAG} ની PM-KISAN સ્થિતિ"

    assert (await vistaar_pmkisan.initiate_pm_kisan_status_check(
        _ctx(), identifier="[AADHAAR_0124]")).startswith("OTP_SENT")
    turn2 = await _chat(monkeypatch, "OTP ૫૮૨૩ છે")
    assert turn2 == "OTP [OTP] છે"
    answer = await vistaar_pmkisan.check_pm_kisan_status_with_otp(_ctx())

    assert [(action, kw.get("identifier"), kw.get("otp")) for action, kw in client.calls] == [
        ("init", AADHAAR, None), ("status", AADHAAR, "5823"),
    ]
    assert "*******6789" in answer and "GJ123456789" not in answer
    assert "******3210" in answer and MOBILE not in answer
    for secret in (AADHAAR, AADHAAR_GU, "5823", "૫૮૨૩"):
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("pending", "typed", "seen"), [
    (False, "મારી ગાય 2025 માં વિયાણી", "મારી ગાય 2025 માં વિયાણી"),
    (True, "OTP 58 23", "OTP [OTP]"),
    (True, "દૂધ 2025-10-01 થી", "દૂધ 2025-10-01 થી"),
    (True, f"આધાર {AADHAAR[:4]}  {AADHAAR[4:8]}.{AADHAAR[8:]}", "આધાર [AADHAAR_0124]"),
    (False, f"5 {AADHAAR}", "5 [AADHAAR_0124]"),
    # Amounts and dates: the first three amounts would pass the Aadhaar checksum.
    (False, "2345 6789 0124 1500", "2345 6789 0124 1500"),
    (False, "દૂધ 2025-01-04 2025-01-11", "દૂધ 2025-01-04 2025-01-11"),
    # 919876543200 passes the Aadhaar checksum, but this is a mobile.
    (False, "+91 98765 43200", "+91 98765 43200"),
])
async def test_what_counts_as_an_otp_or_an_aadhaar(client, monkeypatch, pending, typed, seen):
    if pending:
        await session_pmkisan.set_pm_kisan_session("s1", MOBILE, {"transaction_id": "txn"})
    assert await _chat(monkeypatch, typed) == seen


@pytest.mark.asyncio
async def test_new_otp_goes_out_after_a_verification_that_never_answered(client, monkeypatch):
    await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx(), identifier="GJ123456789")
    client.status_payload = None
    await _chat(monkeypatch, "5823")
    assert "still processing" in await vistaar_pmkisan.check_pm_kisan_status_with_otp(_ctx())
    store = SimpleNamespace(get=AsyncMock(return_value=_operation(OperationState.ACKED_WAITING_CALLBACK)))
    monkeypatch.setattr(vistaar_pmkisan, "get_beckn_operation_store", lambda: store)
    client.status_payload = COMPLETED
    await _chat(monkeypatch, "5823")
    assert (await vistaar_pmkisan.check_pm_kisan_status_with_otp(_ctx())).startswith("PM-KISAN STATUS FOUND")
    assert [action for action, _ in client.calls] == ["init", "status", "status"]


@pytest.mark.asyncio
async def test_parallel_otps_cannot_exceed_the_attempt_limit(client, monkeypatch):
    await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx(), identifier="GJ123456789")
    client.status_payload = None  # no answer arrives, as when the farmer hangs up
    store = SimpleNamespace(get=AsyncMock(return_value=_operation(OperationState.ACKED_WAITING_CALLBACK)))
    monkeypatch.setattr(vistaar_pmkisan, "get_beckn_operation_store", lambda: store)
    answers = []
    for otp in ("1111", "2222", "3333", "4444"):
        await _chat(monkeypatch, otp)
        answers.append(await vistaar_pmkisan.check_pm_kisan_status_with_otp(_ctx()))
    assert answers[-1].startswith("OTP_RETRY_LIMIT")
    assert [action for action, _ in client.calls] == ["init", "status", "status", "status"]


@pytest.mark.asyncio
async def test_records_outside_the_turn_are_masked(client, monkeypatch):
    await session_pmkisan.set_pm_kisan_session("s1", MOBILE, {"transaction_id": "txn"})
    queued = []
    monkeypatch.setattr(
        telemetry_router, "enqueue_canonical_telemetry_event",
        lambda canonical: queued.append(canonical) or ("enqueued", None),
    )


    async def _report(question, session_id="s1"):
        body = json.dumps({"sid": session_id, "target": {"type": "Question", "questionsDetails": {
            "questionText": question, "sessionId": session_id}}}).encode()
        request = SimpleNamespace(headers={}, body=AsyncMock(return_value=body))
        await telemetry_router.ingest_telemetry(request, {"phone": MOBILE})

    # The UI reports the question before it sends the chat request.
    await _report("OTP ૫૮૨૩")
    await _chat(monkeypatch, "OTP ૫૮૨૩")
    await session_pmkisan.clear_pm_kisan_session("s1", MOBILE)  # the check succeeded
    await _report("OTP ૫૮૨૩")
    await _report(f"આધાર {AADHAAR_GU}", "s2")
    await _report("મારી ગાય 2025 માં વિયાણી", "s2")
    assert [c.payload["target"]["questionsDetails"]["questionText"] for c in queued] == [
        "OTP [OTP]", "OTP [OTP]", "આધાર [AADHAAR_0124]", "મારી ગાય 2025 માં વિયાણી",
    ]

    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                               ("ip", "GET", "/api/chat/?query=OTP%205823&session_id=s1", "1.1", 200), None)
    logging.getLogger("uvicorn.access").filter(record)
    assert record.getMessage() == 'ip - "GET /api/chat/?query=<redacted>&session_id=s1 HTTP/1.1" 200'


@pytest.mark.asyncio
@pytest.mark.parametrize(("short_desc", "outcome"), [
    ("No Record Found for this (00000000000) Aadhar/Ben_id/Mobile.", "NO_RECORD"),
    ("Try again", "SERVICE_ERROR"),
    ("This mobile number taged with multiple records.", "MULTIPLE_RECORDS"),
    ("Some new BV reason for 9876543210", 'NOT_ACCEPTED. PM-KISAN did not send an OTP for this identifier: "Some new BV reason for ******3210"'),
])
async def test_bv_init_refusal_gets_its_own_outcome(client, short_desc, outcome):
    client.init_payload = _on_init("otp_error", short_desc)
    answer = await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx(), identifier="GJ123456789")
    assert answer.startswith(outcome)


@pytest.mark.asyncio
async def test_fourth_otp_sms_in_an_hour_is_refused(client):
    answers = [
        await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx(), identifier="GJ123456789")
        for _ in range(4)
    ]
    assert [answer.split(".")[0] for answer in answers] == ["OTP_SENT"] * 3 + ["TOO_MANY_OTP_REQUESTS"]
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_account_mobile_is_used_only_after_the_farmer_agrees(client):
    asked = await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx())
    assert asked.startswith("CONFIRM_MOBILE") and "******3210" in asked
    assert client.calls == []

    await vistaar_pmkisan.initiate_pm_kisan_status_check(_ctx(), use_account_mobile=True)
    assert client.calls[0][1]["identifier"] == MOBILE


@pytest.mark.parametrize(("environment", "allowed"), [("production", False), ("development", True)])
def test_production_refuses_the_bv_sandbox(monkeypatch, environment, allowed):
    from agents.tools.beckn.operations import BecknOperationClient

    sandbox = "bpp-network-playground-sandbox-vistaar.da.gov.in"
    for name, value in {
        "environment": environment, "vistaar_bpp_id": sandbox, "vistaar_bpp_uri": f"https://{sandbox}",
        "beckn_city_code": "std:080", "beckn_bap_caller_url": "x", "beckn_transaction_bridge_token": "x",
        "beckn_bap_id": "x", "beckn_bap_uri": "x", "beckn_callback_token": "x",
    }.items():
        monkeypatch.setattr(settings, name, value)
    if allowed:
        BecknOperationClient._validate_pmkisan_configuration()
    else:
        with pytest.raises(RuntimeError):
            BecknOperationClient._validate_pmkisan_configuration()
