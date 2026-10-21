"""PM-KISAN status: what reaches Bharat Vistaar, and what never reaches anything else.

The farmer's Aadhaar and OTP must reach BV unchanged (gujarati digits included),
while the model, logs, traces and history only ever see placeholders.
"""

import copy
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from agents.tools import session_pmkisan, vistaar_pmkisan
from agents.tools.beckn.operations import (
    BecknActionResult,
    BecknOperation,
    OperationState,
)
from app.config import settings
from app.models.requests import ChatRequest
from app.routers import chat as chat_router

# Synthetic, checksum-valid Aadhaar; written in Gujarati digits by the farmer.
AADHAAR = "234567890124"
AADHAAR_GU = "૨૩૪૫ ૬૭૮૯ ૦૧૨૪"
EAR_TAG = "102345678901"
MOBILE = "9876543210"


class _FakeCache:
    def __init__(self):
        self.data = {}

    async def get(self, key, namespace=None):
        return copy.deepcopy(self.data.get((namespace, key)))

    async def set(self, key, value, ttl=None, namespace=None):
        self.data[(namespace, key)] = copy.deepcopy(value)

    async def delete(self, key, namespace=None):
        self.data.pop((namespace, key), None)


class _FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return None
        self.data[key] = int(value)
        return True

    async def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]


def _result(payload):
    operation = BecknOperation(
        operation_id="op", transaction_id="txn", message_id="msg", action="x",
        expected_callback="on_x", domain="schemes:vistaar", bap_id="bap", bpp_id="bpp",
        session_id="s1", tool_call_id=None, request_hash="h", idempotency_key="k",
        state=OperationState.SUCCEEDED,
    )
    return BecknActionResult(operation=operation, payload=payload)


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
        return _result(self.status_payload)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "vistaar_pmkisan_enabled", True)
    monkeypatch.setattr(settings, "doctor_persona_enabled", False)
    monkeypatch.setattr(session_pmkisan, "cache", _FakeCache())
    monkeypatch.setattr(session_pmkisan, "redis_client", _FakeRedis())
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
async def test_four_digits_stay_when_no_check_is_pending(client, monkeypatch):
    assert await _chat(monkeypatch, "મારી ગાય 2025 માં વિયાણી") == "મારી ગાય 2025 માં વિયાણી"


@pytest.mark.asyncio
@pytest.mark.parametrize(("short_desc", "outcome"), [
    ("No Record Found for this (00000000000) Aadhar/Ben_id/Mobile.", "NO_RECORD"),
    ("Try again", "SERVICE_ERROR"),
])
async def test_bv_outage_is_not_reported_as_a_wrong_id(client, short_desc, outcome):
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
