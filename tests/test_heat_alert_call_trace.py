"""Each attempt to ring a farmer for a heat alert writes one heat_alert_call trace:
placed (its session the call id its turns carry), skipped or error, and why."""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.services import heat_alert_call as hac
from app.services.outbound_call import OutboundCallError

EVENT_ID = uuid.uuid4()


def _event(**row):
    values = dict(id=EVENT_ID, alert_id="alert-1", called=False, call_booked=False, ai_window_end_at=None)
    return SimpleNamespace(**{**values, **row})


def _attempt(monkeypatch, *, claimed=True, row=None, skip=None, dial=None):
    """call_farmer with the database and the dial stubbed; the traces it opened."""
    traces = []

    @contextmanager
    def root(kind, metadata, **fields):
        traces.append({"kind": kind, **metadata, **fields})
        yield None

    async def claim(event_id):
        return _event() if claimed else None

    @asynccontextmanager
    async def session():
        async def get(model, event_id):
            return row

        yield SimpleNamespace(get=get)

    async def reason(event):
        return skip

    async def none(*a, **k):
        return None

    async def dialled(event, species=""):
        if isinstance(dial, Exception):
            raise dial
        return dial

    monkeypatch.setattr(hac, "non_turn_root", root)
    monkeypatch.setattr(hac, "surface_environment", lambda surface: f"{surface}-test")
    monkeypatch.setattr(hac, "_claim", claim)
    monkeypatch.setattr(hac, "get_webhook_session", session)
    monkeypatch.setattr(hac, "_reason_not_to_call", reason)
    monkeypatch.setattr(hac, "_account", none)
    monkeypatch.setattr(hac, "_species", lambda *a: none())
    monkeypatch.setattr(hac, "_dial", dialled)
    monkeypatch.setattr(hac, "_record", none)
    monkeypatch.setattr(hac, "schedule", lambda *a: None)
    asyncio.run(hac.call_farmer(EVENT_ID))
    return traces


def _trace(result, reason=None, session_id=None):
    trace = {
        "kind": "background_job",
        "amul.job": "heat_alert_call",
        "amul.pipeline": "telephony",
        "amul.direction": "outbound",
        "amul.flow": "heat_alert",
        "amul.heat_alert_event_id": str(EVENT_ID),
        "amul.heat_alert_id": "alert-1",
        "amul.call_result": result,
        "session_id": session_id,
        "environment": "voice-test",
    }
    if reason:
        trace["amul.call_reason"] = reason
    return trace


def _sent(traces):
    stamps = {"amul.schema_version", "amul.service", "amul.release", "amul.kind"}
    return [{k: v for k, v in t.items() if k not in stamps} for t in traces]


def test_a_placed_call_is_traced_with_the_call_id_as_its_session(monkeypatch):
    traces = _attempt(monkeypatch, dial="raya-call-uuid")

    assert _sent(traces) == [_trace("placed", session_id="raya-call-uuid")]
    assert traces[0]["amul.kind"] == "background_job"


def test_a_skipped_call_is_traced_with_why(monkeypatch):
    assert _sent(_attempt(monkeypatch, skip="not_in_only_numbers")) == [_trace("skipped", "not_in_only_numbers")]
    # Not claimed: past the AI window, or booked; a row another attempt took is that attempt's.
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    assert _sent(_attempt(monkeypatch, claimed=False, row=_event(ai_window_end_at=past))) == [_trace("skipped", "window_passed")]
    assert _sent(_attempt(monkeypatch, claimed=False, row=_event(call_booked=True))) == [_trace("skipped", "already_booked")]
    assert _attempt(monkeypatch, claimed=False, row=_event(called=True)) == []


def test_a_refused_call_is_traced_as_an_error_with_its_kind(monkeypatch):
    traces = _attempt(monkeypatch, dial=OutboundCallError("rate_limited", retry_after=20))

    assert _sent(traces) == [_trace("error", "rate_limited")]
