"""The dashboard's read-only queries over the telemetry database.

ClickHouse is stood in for by a client that records each query and answers with
fixed rows, so these pin the SQL the rules in docs/TELEMETRY_PIPELINE.md ask for
(FINAL, one environment, parameters rather than inlined values) and what the
endpoints return.
"""
import os
import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import telemetry_query as router_module
from app.services import telemetry_query

KEY = "dashboard-key"


class _ClickHouse:
    """Answers each channel's table with the rows given for it."""

    def __init__(self, voice=(), chat=(), fail=None):
        self.rows = {"voice": list(voice), "chat": list(chat)}
        self.fail = fail
        self.queries = []
        self.closed = False

    def query(self, query, parameters=None):
        if self.fail:
            raise self.fail
        self.queries.append((query, dict(parameters or {})))
        channel = "voice" if "telemetry.voice_turns" in query else "chat"
        return SimpleNamespace(result_rows=self.rows[channel])

    def close(self):
        self.closed = True


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "telemetry_query_api_key", KEY)
    monkeypatch.setattr(settings, "telemetry_dashboard_password", "secret")
    monkeypatch.setattr(settings, "telemetry_query_voice_environment", "voice-production")
    monkeypatch.setattr(settings, "telemetry_query_chat_environment", "chat-production")


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router_module.router, prefix="/api")
    return TestClient(app)


def _use(monkeypatch, clickhouse):
    monkeypatch.setattr(telemetry_query, "dashboard_client", lambda: clickhouse)
    return clickhouse


# ── access ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key, password", [(None, "secret"), ("", "secret"), (KEY, None)])
def test_unconfigured_queries_answer_503(monkeypatch, api, key, password):
    monkeypatch.setattr(settings, "telemetry_query_api_key", key)
    monkeypatch.setattr(settings, "telemetry_dashboard_password", password)
    clickhouse = _use(monkeypatch, _ClickHouse())

    response = api.get("/api/telemetry/stats", headers={"X-API-Key": KEY})

    assert response.status_code == 503
    assert clickhouse.queries == []


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_a_missing_or_wrong_key_is_refused(monkeypatch, api, configured, headers):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in ("/api/telemetry/stats", "/api/telemetry/daily"):
        assert api.get(path, headers=headers).status_code == 401
    assert clickhouse.queries == []


# ── /stats ──────────────────────────────────────────────────────────────────


def test_stats_counts_each_channel_and_adds_them_up(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(voice=[(120, 40, 30)], chat=[(80, 25, 20)]))

    response = api.get(
        "/api/telemetry/stats", params={"from": "2026-09-01", "to": "2026-09-30"}, headers={"X-API-Key": KEY}
    )

    assert response.status_code == 200
    assert response.json() == {
        "from": "2026-09-01",
        "to": "2026-09-30",
        "voice": {"questions": 120, "sessions": 40, "users": 30},
        "chat": {"questions": 80, "sessions": 25, "users": 20},
        "total": {"questions": 200, "sessions": 65, "users": 50},
    }
    assert clickhouse.closed


def test_stats_reads_each_table_final_for_its_production_environment(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(voice=[(0, 0, 0)], chat=[(0, 0, 0)]))

    api.get("/api/telemetry/stats", params={"from": "2026-09-01", "to": "2026-09-30"}, headers={"X-API-Key": KEY})

    (voice_sql, voice_params), (chat_sql, chat_params) = clickhouse.queries
    assert "FROM telemetry.voice_turns FINAL" in voice_sql
    assert "FROM telemetry.chat_turns FINAL" in chat_sql
    assert voice_params == {"environment": "voice-production", "first_day": date(2026, 9, 1), "last_day": date(2026, 9, 30)}
    assert chat_params["environment"] == "chat-production"
    for sql, _ in clickhouse.queries:
        assert "{environment:String}" in sql
        assert "{first_day:Date}" in sql and "{last_day:Date}" in sql
        assert "uniq(session_id)" in sql and "uniq(user_id_hash)" in sql
        # Values travel as parameters, never inside the SQL.
        assert "production" not in sql and "2026" not in sql


def test_an_empty_answer_counts_zero(monkeypatch, api, configured):
    _use(monkeypatch, _ClickHouse(voice=[], chat=[]))

    body = api.get("/api/telemetry/stats", headers={"X-API-Key": KEY}).json()

    assert body["total"] == {"questions": 0, "sessions": 0, "users": 0}


def test_with_no_range_it_counts_everything_up_to_today(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(voice=[(1, 1, 1)], chat=[(1, 1, 1)]))

    body = api.get("/api/telemetry/stats", headers={"X-API-Key": KEY}).json()

    today = datetime.now(timezone.utc).date()
    assert (body["from"], body["to"]) == ("1970-01-01", today.isoformat())
    assert clickhouse.queries[0][1]["last_day"] == today


def test_other_environments_can_be_read(monkeypatch, api, configured):
    monkeypatch.setattr(settings, "telemetry_query_voice_environment", "voice-development")
    monkeypatch.setattr(settings, "telemetry_query_chat_environment", "chat-development")
    clickhouse = _use(monkeypatch, _ClickHouse(voice=[(0, 0, 0)], chat=[(0, 0, 0)]))

    api.get("/api/telemetry/stats", headers={"X-API-Key": KEY})

    assert [params["environment"] for _, params in clickhouse.queries] == ["voice-development", "chat-development"]


@pytest.mark.parametrize("params, status", [
    ({"from": "2026-09-30", "to": "2026-09-01"}, 400),
    ({"from": "30-09-2026"}, 422),
])
def test_a_bad_range_is_refused_before_any_query(monkeypatch, api, configured, params, status):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert api.get("/api/telemetry/stats", params=params, headers={"X-API-Key": KEY}).status_code == status
    assert clickhouse.queries == []


def test_a_database_failure_is_502_and_the_client_is_closed(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(fail=ConnectionError("clickhouse down")))

    response = api.get("/api/telemetry/stats", headers={"X-API-Key": KEY})

    assert response.status_code == 502
    assert response.json() == {"detail": "Telemetry database unavailable"}
    assert clickhouse.closed


def test_an_unreachable_database_is_502(monkeypatch, api, configured):
    def _unreachable():
        raise ConnectionError("no route to clickhouse")

    monkeypatch.setattr(telemetry_query, "dashboard_client", _unreachable)

    assert api.get("/api/telemetry/daily", headers={"X-API-Key": KEY}).status_code == 502


# ── /daily ──────────────────────────────────────────────────────────────────


def test_daily_gives_each_channels_days_in_order(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        voice=[(date(2026, 9, 1), 10, 4, 3), (date(2026, 9, 2), 12, 5, 4)],
        chat=[(date(2026, 9, 2), 7, 3, 2)],
    ))

    response = api.get(
        "/api/telemetry/daily", params={"from": "2026-09-01", "to": "2026-09-02"}, headers={"X-API-Key": KEY}
    )

    assert response.json() == {
        "from": "2026-09-01",
        "to": "2026-09-02",
        "voice": [
            {"day": "2026-09-01", "questions": 10, "sessions": 4, "users": 3},
            {"day": "2026-09-02", "questions": 12, "sessions": 5, "users": 4},
        ],
        "chat": [{"day": "2026-09-02", "questions": 7, "sessions": 3, "users": 2}],
    }
    for sql, _ in clickhouse.queries:
        assert "FINAL" in sql
        assert "GROUP BY day" in sql and "ORDER BY day" in sql
    assert clickhouse.closed


# ── the client ──────────────────────────────────────────────────────────────


def test_the_client_reads_the_telemetry_database_as_telemetry_dashboard(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "clickhouse_connect", SimpleNamespace(get_client=lambda **kw: calls.append(kw)))
    monkeypatch.setattr(settings, "telemetry_clickhouse_host", "clickhouse.internal")
    monkeypatch.setattr(settings, "telemetry_clickhouse_port", 8124)
    monkeypatch.setattr(settings, "telemetry_dashboard_password", "secret")

    telemetry_query.dashboard_client()

    assert calls == [{
        "host": "clickhouse.internal",
        "port": 8124,
        "username": "telemetry_dashboard",
        "password": "secret",
        "database": "telemetry",
    }]


def test_the_app_serves_both_queries():
    import main

    paths = {route.path for route in main.app.routes}
    assert {"/api/telemetry/stats", "/api/telemetry/daily"} <= paths
