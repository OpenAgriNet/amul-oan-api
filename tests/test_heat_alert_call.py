"""Heat-alert call: dial spacing, retries and recovery against a real Postgres.

The cases are the ways a farmer would be rung wrongly or not at all: the 06:00
batch tripping Raya's rate limit (2026-10-30: 2 of 5 refused, 11-02: 5 of 8),
an unanswered call never tried again, a call at night, and a rollout ringing a
row whose call was already settled.

Raya is a fake with its rate limit (one dial per gap); timings are scaled to
fractions of a second. Needs TEST_WEBHOOK_DB_URL (postgresql+asyncpg://...), a
throwaway database: every test drops and recreates the table.
"""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from sqlalchemy import select

from app.config import settings
from app.core import webhook_db
from app.models.webhook import HeatAlertWebhookEvent
from app.services import heat_alert_call, outbound_call

DB_URL = os.getenv("TEST_WEBHOOK_DB_URL")
pytestmark = pytest.mark.asyncio
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations" / "webhook"
GAP = 0.3


class FakeRaya:
    """One dial per ``limit`` seconds per key, as Raya; outcomes per number."""

    def __init__(self, limit=GAP * 0.9, outcome="Completed", refuse=0):
        self.limit, self.outcome, self.refuse = limit, outcome, refuse
        self.dials, self.placed = [], {}
        self.last = None

    async def initiate_call(self, agent_id, to_number, agent_args=None):
        # The real quiet-hours guard runs first, as in initiate_call.
        if outbound_call.in_quiet_hours(outbound_call.datetime.now(timezone.utc)):
            raise outbound_call.OutboundCallError("quiet_hours")
        now = asyncio.get_running_loop().time()
        self.dials.append(now)
        if self.refuse or (self.last is not None and now - self.last < self.limit):
            self.refuse = max(0, self.refuse - 1)
            raise outbound_call.OutboundCallError("rate_limited", retry_after=0.05)
        self.last = now
        call_uuid = str(uuid.uuid4())
        self.placed[call_uuid] = to_number
        return call_uuid

    async def call_outcome(self, call_uuid):
        return self.outcome(call_uuid) if callable(self.outcome) else self.outcome

    def calls_to(self, number):
        return sum(1 for n in self.placed.values() if n == number)


@pytest_asyncio.fixture
async def db(monkeypatch):
    if not DB_URL:
        pytest.skip("TEST_WEBHOOK_DB_URL not set")
    monkeypatch.setattr(settings, "webhook_db_url", DB_URL)
    monkeypatch.setattr(settings, "raya_heat_alert_agent_id", "agent-1")
    monkeypatch.setattr(settings, "raya_outbound_call_api_key", "key-1")
    monkeypatch.setattr(settings, "heat_alert_call_only_numbers", "")
    monkeypatch.setattr(settings, "raya_outbound_call_gap_seconds", GAP)
    monkeypatch.setattr(settings, "heat_alert_call_retries", 2)
    monkeypatch.setattr(settings, "heat_alert_call_retry_after_seconds", 1)
    monkeypatch.setattr(heat_alert_call, "_REFUSED_RETRY", timedelta(seconds=1))
    monkeypatch.setattr(heat_alert_call, "_OUTCOME_FIRST_CHECK", timedelta(seconds=0.2))
    monkeypatch.setattr(heat_alert_call, "_OUTCOME_RECHECK", timedelta(seconds=0.2))
    monkeypatch.setattr(heat_alert_call, "_DIAL_LOCK_POLL_S", 0.02)
    monkeypatch.setattr(heat_alert_call, "agent_args", lambda event, species="": {"alert_id": event.alert_id})
    monkeypatch.setattr(heat_alert_call, "_species", _no_species)
    monkeypatch.setattr(heat_alert_call, "_account", _no_account)

    async def no_ticket(event):
        return False

    monkeypatch.setattr(heat_alert_call, "_has_pending_ticket", no_ticket)
    monkeypatch.setattr(webhook_db, "_engine", None)
    monkeypatch.setattr(webhook_db, "_sessionmaker", None)
    await run_sql("DROP TABLE IF EXISTS heat_alert_webhook_events")
    await run_sql((MIGRATIONS / "001_init.sql").read_text())
    yield
    await outbound_call.cancel_scheduled()
    await webhook_db._engine.dispose()


async def run_sql(script):
    """A migration file as psql applies it (several statements, comments)."""
    async with webhook_db.get_webhook_session() as session:
        raw = await (await session.connection()).get_raw_connection()
        await raw.driver_connection.execute(script)
        await session.commit()


async def _no_species(event, account):
    return ""


async def _no_account(event):
    return None


def raya(monkeypatch, **kw):
    fake = FakeRaya(**kw)
    monkeypatch.setattr(outbound_call, "initiate_call", fake.initiate_call)
    monkeypatch.setattr(outbound_call, "call_outcome", fake.call_outcome)
    return fake


async def add(n=1, *, window=timedelta(hours=12), **values):
    now = datetime.now(timezone.utc)
    ids = []
    async with webhook_db.get_webhook_session() as session:
        for i in range(n):
            row = HeatAlertWebhookEvent(
                alert_id=str(uuid.uuid4()),
                device_id="D",
                farmer_contact=values.pop("farmer_contact", f"90000000{i:02d}"),
                payload_raw={},
                call_due_at=now,
                ai_window_end_at=now + window,
                **values,
            )
            session.add(row)
            await session.flush()
            ids.append(row.id)
        await session.commit()
    return ids


async def rows():
    async with webhook_db.get_webhook_session() as session:
        return {r.id: r for r in (await session.execute(select(HeatAlertWebhookEvent))).scalars()}


async def settle(seconds):
    """Let scheduled retries and outcome reads run."""
    await asyncio.sleep(seconds)


async def test_the_0600_batch_is_dialled_one_at_a_time_and_nobody_is_refused(db, monkeypatch):
    fake = raya(monkeypatch)
    ids = await add(8)
    # Eight rows due at once, as after quiet hours; each call_farmer is what a replica's timer runs.
    await asyncio.gather(*(heat_alert_call.call_farmer(i) for i in ids))
    state = await rows()
    assert len(fake.placed) == 8
    assert all(r.outbound_call_uuid and r.call_error is None for r in state.values())
    gaps = [b - a for a, b in zip(fake.dials, fake.dials[1:])]
    assert len(fake.dials) == 8 and min(gaps) >= GAP * 0.95


async def test_a_rate_limited_dial_is_tried_again_with_backoff(db, monkeypatch):
    fake = raya(monkeypatch, refuse=2)
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    row = (await rows())[event_id]
    assert len(fake.dials) == 3 and row.outbound_call_uuid and row.call_attempts == 1


async def test_a_dial_refused_throughout_is_rung_later_not_dropped(db, monkeypatch):
    fake = raya(monkeypatch, refuse=heat_alert_call._RATE_LIMIT_TRIES)
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    row = (await rows())[event_id]
    assert row.called is False and row.call_error == "rate_limited" and row.call_due_at > datetime.now(timezone.utc)
    await settle(1.5)
    assert len(fake.placed) == 1 and (await rows())[event_id].outbound_call_uuid


async def test_an_unanswered_call_is_rung_again_until_the_retries_run_out(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Failure")
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    await settle(5)
    row = (await rows())[event_id]
    assert fake.calls_to("9000000000") == 1 + settings.heat_alert_call_retries
    assert row.call_attempts == 3 and row.called is True


async def test_an_answered_call_is_not_rung_again(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Completed")
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    await settle(2)
    row = (await rows())[event_id]
    assert len(fake.placed) == 1 and row.call_outcome == "Completed"


async def test_no_retry_once_the_ai_window_would_have_closed(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Failure")
    (event_id,) = await add(window=timedelta(seconds=1))
    await heat_alert_call.call_farmer(event_id)
    await settle(2)
    assert len(fake.placed) == 1 and (await rows())[event_id].call_outcome == "Failure"


async def test_a_booked_alert_is_not_rung_again(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Failure")
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    await heat_alert_call.mark_booked(str(event_id))
    await settle(2)
    assert len(fake.placed) == 1


class _At(datetime):
    """``datetime.now`` frozen at a time of day in IST."""

    at = None

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz)


def _ist(hh, mm):
    return datetime(2026, 11, 2, hh, mm, tzinfo=outbound_call.IST)


@pytest.mark.parametrize("hh,mm", [(22, 0), (23, 30), (2, 0), (5, 59)])
async def test_initiate_call_never_dials_in_quiet_hours(monkeypatch, hh, mm):
    monkeypatch.setattr(settings, "raya_outbound_call_api_key", "key-1")
    monkeypatch.setattr(_At, "at", _ist(hh, mm))
    monkeypatch.setattr(outbound_call, "datetime", _At)

    class NoHTTP:
        def __init__(self, *a, **k):
            raise AssertionError("dialled in quiet hours")

    monkeypatch.setattr(outbound_call.httpx, "AsyncClient", NoHTTP)
    with pytest.raises(outbound_call.OutboundCallError, match="quiet_hours"):
        await outbound_call.initiate_call("agent-1", "9000000000")


async def test_a_dial_that_waited_past_2200_moves_to_0600(db, monkeypatch):
    fake = raya(monkeypatch)
    monkeypatch.setattr(_At, "at", _ist(22, 1))
    monkeypatch.setattr(outbound_call, "datetime", _At)
    monkeypatch.setattr(heat_alert_call, "datetime", _At)
    monkeypatch.setattr(heat_alert_call, "schedule", lambda event_id, due_at: None)
    (event_id,) = await add(window=timedelta(days=2))
    await heat_alert_call.call_farmer(event_id)
    row = (await rows())[event_id]
    assert not fake.placed and row.called is False
    assert row.call_due_at.astimezone(outbound_call.IST).time().hour == 6


async def test_a_retry_due_in_quiet_hours_moves_to_0600():
    late = outbound_call.after_quiet_hours(_ist(21, 50) + timedelta(minutes=30))
    assert (late.hour, late.minute, late.day) == (6, 0, 3)


async def test_recover_rings_only_rows_still_owed_a_call(db, monkeypatch):
    # Only the call recovery finds unsettled goes unanswered.
    fake = raya(monkeypatch, outcome=lambda call_uuid: "Failure" if call_uuid == "u-2" else "Completed")
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    due = await add(farmer_contact="9100000001")
    awaiting = await add(farmer_contact="9100000002", called=True, outbound_call_uuid="u-2", call_attempts=1)
    await add(farmer_contact="9100000003", called=True, call_error="invalid_number")
    await add(farmer_contact="9100000004", called=True, call_error="pending_ticket")
    await add(farmer_contact="9100000005", called=True, outbound_call_uuid="u-5", call_attempts=1, call_outcome="Completed")
    await add(farmer_contact="9100000006", called=True, outbound_call_uuid="u-6", call_attempts=3)
    await add(farmer_contact="9100000007", call_booked=True)
    await add(farmer_contact="9100000008", window=past - datetime.now(timezone.utc))
    fake.placed["u-2"] = "9100000002"
    assert await heat_alert_call.recover() == 1
    await settle(2)
    assert set(fake.placed.values()) - {"9100000002"} == {"9100000001"}  # the due row
    assert fake.calls_to("9100000002") == 2  # the unanswered row, rung again
    state = await rows()
    assert state[due[0]].outbound_call_uuid and state[awaiting[0]].call_attempts == 2


async def test_migration_006_settles_old_rows_so_the_rollout_rings_nobody(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Failure")
    # Rows as the old code left them: a refused dial released its claim; a placed call had no outcome.
    await add(farmer_contact="9200000001", call_error="rate_limited")
    await add(farmer_contact="9200000002", called=True, outbound_call_uuid="u-old")
    await run_sql((MIGRATIONS / "006_call_retries.sql").read_text())
    assert await heat_alert_call.recover() == 0
    await settle(1)
    assert not fake.placed


async def test_a_number_refused_for_good_is_not_rung_again_after_a_restart(db, monkeypatch):
    fake = raya(monkeypatch)

    async def bad_number(*a, **k):
        raise outbound_call.OutboundCallError("http_400")

    monkeypatch.setattr(outbound_call, "initiate_call", bad_number)
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    monkeypatch.setattr(outbound_call, "initiate_call", fake.initiate_call)
    assert await heat_alert_call.recover() == 0
    await settle(0.5)
    assert not fake.placed and (await rows())[event_id].call_error == "http_400"


async def test_two_replicas_reading_one_unanswered_call_ring_it_again_once(db, monkeypatch):
    fake = raya(monkeypatch, outcome="Failure")
    monkeypatch.setattr(settings, "heat_alert_call_retries", 1)
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    call_uuid = (await rows())[event_id].outbound_call_uuid
    await asyncio.gather(*(heat_alert_call._check_outcome(event_id, call_uuid, 1) for _ in range(2)))
    await settle(2)
    assert len(fake.placed) == 2


async def test_a_late_second_read_of_an_outcome_leaves_the_retry_alone(db, monkeypatch):
    raya(monkeypatch, outcome="Failure")
    monkeypatch.setattr(heat_alert_call, "schedule", lambda event_id, due_at: None)
    monkeypatch.setattr(heat_alert_call, "_follow_up", lambda *a, **k: None)
    (event_id,) = await add()
    await heat_alert_call.call_farmer(event_id)
    call_uuid = (await rows())[event_id].outbound_call_uuid
    await heat_alert_call._check_outcome(event_id, call_uuid, 1)
    assert await heat_alert_call._claim(event_id) is not None  # the retry is being dialled
    await heat_alert_call._check_outcome(event_id, call_uuid, 1)  # another replica, late
    assert (await rows())[event_id].called is True
