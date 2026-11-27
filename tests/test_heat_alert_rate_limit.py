"""A call Raya refuses for its rate limit is tried again later, not dropped until a pod restarts."""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.services import heat_alert_call as hac
from app.services.outbound_call import OutboundCallError


def _run(monkeypatch, error, retries=0):
    scheduled, recorded = [], []

    async def claim(event_id):
        return object()

    async def none(*a, **k):
        return None

    async def dial(event, species=""):
        raise error

    async def record(event_id, **values):
        recorded.append(values)

    monkeypatch.setattr(hac, "_claim", claim)
    monkeypatch.setattr(hac, "_reason_not_to_call", none)
    monkeypatch.setattr(hac, "_account", none)
    monkeypatch.setattr(hac, "_species", lambda *a: none())
    monkeypatch.setattr(hac, "_dial", dial)
    monkeypatch.setattr(hac, "_record", record)
    monkeypatch.setattr(hac, "schedule", lambda *a: scheduled.append(a))
    asyncio.run(hac.call_farmer(uuid.uuid4(), retries))
    return scheduled, recorded


def test_rate_limited_call_is_released_and_tried_again_in_two_to_four_minutes(monkeypatch):
    before = datetime.now(timezone.utc)
    scheduled, recorded = _run(monkeypatch, OutboundCallError("rate_limited", retry_after=20))
    assert recorded == [{"called": False, "call_error": "rate_limited"}]
    [(_, due_at, retries)] = scheduled
    assert retries == 1
    assert before + timedelta(minutes=2) <= due_at <= datetime.now(timezone.utc) + timedelta(minutes=4)


def test_retries_stop_at_the_cap(monkeypatch):
    scheduled, _ = _run(monkeypatch, OutboundCallError("rate_limited", retry_after=20), hac._RATE_LIMIT_RETRIES)
    assert scheduled == []


def test_other_refusals_are_not_retried(monkeypatch):
    for error in (OutboundCallError("http_400"), OutboundCallError("ReadTimeout", ambiguous=True)):
        scheduled, _ = _run(monkeypatch, error)
        assert scheduled == []
