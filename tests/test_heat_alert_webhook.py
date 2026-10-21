"""Partner heat-alert webhook: the failure modes that would cost a farmer an alert
or send them one twice.

The real ``get_webhook_session`` runs against a fake DB that keeps Postgres's
transaction semantics (uncommitted rows vanish on rollback or close) and keys
``ON CONFLICT`` on the columns the statement names. The WhatsApp API is a fake
``httpx.AsyncClient``. No Postgres or network.
"""
import asyncio
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import settings
from app.core import webhook_db
from app.services import whatsapp_heat_alert
from app.tasks import webhook_cleanup_worker as cleanup
from main import app

TOKEN = "test-webhook-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PAYLOAD = {
    "alertID": "111",
    "deviceid": "DEV-1",
    "tag no": "T-1",
    "Pashuaadhar No": "PA-1",
    "alert_type": "Amber",
    "notification_type": "HEAT",
    "AI Window start_time": "2026-10-17T09:40:39.000Z",
    "AI Window end_time": "2026-10-17T21:40:39.000Z",
    "timestamp": "2026-10-17T09:40:39.000Z",
    "farmer_contact": "9000000001",
    "booking_url": "https://booking.example.test/heat/abc",
}


class _FakeDB:
    def __init__(self):
        self.rows = {}
        self.deletes = []

    def session(self):
        return _FakeSession(self)


class _FakeSession:
    def __init__(self, db):
        self.db = db
        self.pending = {}

    async def execute(self, stmt):
        compiled = stmt.compile(dialect=postgresql.dialect())
        sql, params = str(compiled), compiled.params
        if sql.startswith("INSERT"):
            columns = re.search(r"ON CONFLICT \(([^)]*)\) DO NOTHING", sql).group(1).split(", ")
            key = tuple(params[c] for c in columns)
            if key in self.db.rows or key in self.pending:
                return _Result(None)
            row = (uuid.uuid4(), params["received_at"])
            self.pending[key] = row
            return _Result(row)
        if sql.startswith("SELECT"):
            wanted = (params["alert_id_1"], params["farmer_contact_1"])
            return _Result(next((row for key, row in self.db.rows.items() if key == wanted), None))
        self.db.deletes.append(compiled)
        return _Result(None, rowcount=0)

    async def commit(self):
        self.db.rows.update(self.pending)
        self.pending.clear()

    async def rollback(self):
        self.pending.clear()

    async def close(self):
        self.pending.clear()


class _Result:
    def __init__(self, row, rowcount=1):
        self._row = row
        self.rowcount = rowcount

    def first(self):
        return self._row


class _FakeWhatsApp:
    """Stands in for httpx.AsyncClient; answers each POST from a queue."""

    def __init__(self):
        self.posts = []
        self.replies = []

    def __call__(self, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers, json):
        self.posts.append({"url": url, "headers": headers, "json": json})
        status, body = self.replies.pop(0) if self.replies else (200, {"success": True, "messageUuid": "m-1"})
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))


@pytest.fixture
def wired(monkeypatch):
    db, whatsapp = _FakeDB(), _FakeWhatsApp()
    monkeypatch.setattr(settings, "webhook_shared_token", TOKEN)
    monkeypatch.setattr(settings, "webhook_db_url", "postgresql+asyncpg://fake/db")
    monkeypatch.setattr(settings, "sarlaben_whatsapp_api_key", "test-api-key")
    monkeypatch.setattr(webhook_db, "_get_sessionmaker", lambda: db.session)
    monkeypatch.setattr(whatsapp_heat_alert.httpx, "AsyncClient", whatsapp)
    return TestClient(app), db, whatsapp


def test_a_partner_retry_is_accepted_without_a_second_whatsapp(wired):
    client, db, whatsapp = wired

    first = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)
    retry = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)

    assert first.status_code == retry.status_code == 202
    assert retry.json()["id"] == first.json()["id"]
    assert len(whatsapp.posts) == 1
    assert whatsapp.posts[0]["url"] == "https://comm.amul.in/sarlaben/api/whatsapp/heat-alert"
    assert whatsapp.posts[0]["json"] == {
        "mobile": "9000000001",
        "animalNo": "PA-1",
        "tagNo": "T-1",
        "bookingLink": "https://booking.example.test/heat/abc",
    }

    # The idempotency key is the (alert, farmer) pair: another farmer on the same alert is sent.
    other_farmer = client.post("/api/webhooks/alerts", json={**PAYLOAD, "farmer_contact": "9000000002"}, headers=AUTH)
    assert other_farmer.status_code == 202
    assert [p["json"]["mobile"] for p in whatsapp.posts] == ["9000000001", "9000000002"]
    assert len(db.rows) == 2


def test_auth_fails_closed(wired, monkeypatch):
    client, db, whatsapp = wired

    assert client.post("/api/webhooks/alerts", json=PAYLOAD).status_code == 401
    assert client.post("/api/webhooks/alerts", json=PAYLOAD, headers={"Authorization": "Bearer nope"}).status_code == 401

    monkeypatch.setattr(settings, "webhook_shared_token", None)
    unset = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)
    assert unset.status_code == 503
    assert unset.json() == {"detail": "Webhook auth is not configured"}
    assert db.rows == {} and whatsapp.posts == []


def test_a_failed_whatsapp_send_rolls_back_so_the_retry_sends(wired):
    client, db, whatsapp = wired
    whatsapp.replies.append((500, {"error": "down"}))

    failed = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)

    assert failed.status_code == 502
    assert db.rows == {}

    retry = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)
    assert retry.status_code == 202
    assert len(whatsapp.posts) == 2
    assert len(db.rows) == 1


def test_without_webhook_db_url_the_route_refuses_and_no_worker_starts(wired, monkeypatch):
    client, _db, whatsapp = wired
    monkeypatch.setattr(settings, "webhook_db_url", None)

    resp = client.post("/api/webhooks/alerts", json=PAYLOAD, headers=AUTH)
    asyncio.run(cleanup.start_webhook_cleanup_worker())

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Webhook persistence DB is not configured"}
    assert whatsapp.posts == []
    assert cleanup._worker_task is None


def test_cleanup_deletes_only_rows_past_retention(wired, monkeypatch):
    _client, db, _whatsapp = wired
    monkeypatch.setattr(settings, "webhook_retention_hours", 24)

    asyncio.run(cleanup._cleanup_once())

    (compiled,) = db.deletes
    assert "WHERE heat_alert_webhook_events.received_at < %(received_at_1)s" in str(compiled)
    cutoff = compiled.params["received_at_1"]
    assert abs(cutoff - (datetime.now(timezone.utc) - timedelta(hours=24))) < timedelta(seconds=5)
