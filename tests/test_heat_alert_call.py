"""Heat-alert outbound call: it rings once, only while the alert is still worth
it, and never at night. Raya is a fake ``httpx.AsyncClient``; the DB claim and
writeback are faked. No Postgres or network."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.config import settings
from app.services import heat_alert_call, outbound_call

EVENT = SimpleNamespace(
    id="e1", alert_id="111", farmer_contact="+91 90000 00001", pashuaadhar_no="PA-1", tag_no="T-1"
)


class _FakeRaya:
    def __init__(self):
        self.posts, self.replies = [], []

    def __call__(self, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers, json):
        self.posts.append({"headers": headers, "json": json})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(reply[0], json=reply[1], request=httpx.Request("POST", url))


@pytest.fixture
def wired(monkeypatch):
    raya, records = _FakeRaya(), []

    async def claim(_event_id):
        return EVENT

    async def record(_event_id, **values):
        records.append(values)

    monkeypatch.setattr(settings, "raya_call_api_key", "raya_test")
    monkeypatch.setattr(settings, "heat_alert_call_agent_id", "agent-1")
    monkeypatch.setattr(outbound_call.httpx, "AsyncClient", raya)
    monkeypatch.setattr(heat_alert_call, "_claim", claim)
    monkeypatch.setattr(heat_alert_call, "_record", record)
    return raya, records


def test_rate_limited_once_then_placed(wired):
    raya, records = wired
    raya.replies = [(429, {"retry_after": 0.01}), (200, {"uuid": "c-1"})]

    asyncio.run(heat_alert_call.call_farmer("e1"))

    assert records == [{"outbound_call_uuid": "c-1"}]
    assert len(raya.posts) == 2
    assert raya.posts[0]["headers"] == {"X-API-Key": "raya_test"}
    body = raya.posts[0]["json"]
    assert body["agent_id"] == "agent-1" and body["to_number"] == "9000000001"
    assert body["agent_args"]["alert_id"] == "111"
    assert "PA-1" in body["agent_args"]["script"] and "T-1" in body["agent_args"]["script"]


def test_a_timeout_is_not_retried_and_is_recorded(wired):
    raya, records = wired
    raya.replies = [httpx.ReadTimeout("slow")]

    asyncio.run(heat_alert_call.call_farmer("e1"))

    assert len(raya.posts) == 1
    assert records == [{"call_error": "ReadTimeout"}]


def test_an_unclaimable_alert_is_not_dialled(wired, monkeypatch):
    raya, records = wired

    async def nothing_to_claim(_event_id):
        return None

    monkeypatch.setattr(heat_alert_call, "_claim", nothing_to_claim)
    asyncio.run(heat_alert_call.call_farmer("e1"))

    assert raya.posts == [] and records == []


@pytest.mark.parametrize(
    "due_utc, expected_utc",
    [
        ("2026-10-26T17:30:00", "2026-10-27T00:30:00"),  # 23:00 IST -> 06:00 IST next day
        ("2026-10-26T21:30:00", "2026-10-27T00:30:00"),  # 03:00 IST -> 06:00 IST
        ("2026-10-26T08:30:00", "2026-10-26T08:30:00"),  # 14:00 IST unchanged
    ],
)
def test_calls_due_at_night_move_to_six(due_utc, expected_utc):
    due = datetime.fromisoformat(due_utc).replace(tzinfo=timezone.utc)
    assert outbound_call.after_quiet_hours(due) == datetime.fromisoformat(expected_utc).replace(
        tzinfo=timezone.utc
    )
