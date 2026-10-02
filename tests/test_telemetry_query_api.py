"""The dashboard's read-only queries over the telemetry database.

ClickHouse is stood in for by a client that records each query and answers with
fixed rows, so these pin the SQL the rules in docs/TELEMETRY_PIPELINE.md ask for
(FINAL, one environment, parameters rather than inlined values) and what the
endpoints return.
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import telemetry_query as router_module
from app.services import telemetry_query

KEY = "dashboard-key"
PATHS = ("/api/telemetry/stats", "/api/telemetry/graph", "/api/telemetry/sessions", "/api/telemetry/outcomes")
SEPTEMBER = {"from": "2026-09-01", "to": "2026-09-30"}


def _kind(sql):
    for marker, kind in (("first_seen", "new_users"), ("AS start", "graph"), ("avgOrNull", "sessions"), ("AS outcome", "outcomes")):
        if marker in sql:
            return kind
    return "totals"


class _ClickHouse:
    """Answers each kind of query on each channel's table with the rows given
    for it, e.g. ``totals_voice=[(1, 1, 1)]``; anything else gets no rows."""

    def __init__(self, fail=None, **rows):
        self.rows = rows
        self.fail = fail
        self.queries = []
        self.closed = False

    def query(self, query, parameters=None):
        if self.fail:
            raise self.fail
        self.queries.append((query, dict(parameters or {})))
        channel = "voice" if "telemetry.voice_turns" in query else "chat"
        return SimpleNamespace(result_rows=list(self.rows.get(f"{_kind(query)}_{channel}", [])))

    def close(self):
        self.closed = True

    def sql(self, kind):
        return [sql for sql, _ in self.queries if _kind(sql) == kind]


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


def _get(api, path, **params):
    """A request for September unless it says otherwise."""
    return api.get(path, params={**SEPTEMBER, **params}, headers={"X-API-Key": KEY})


# ── access ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key, password", [(None, "secret"), ("", "secret"), (KEY, None)])
def test_unconfigured_queries_answer_503(monkeypatch, api, key, password):
    monkeypatch.setattr(settings, "telemetry_query_api_key", key)
    monkeypatch.setattr(settings, "telemetry_dashboard_password", password)
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        assert _get(api, path).status_code == 503
    assert clickhouse.queries == []


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_a_missing_or_wrong_key_is_refused(monkeypatch, api, configured, headers):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        assert api.get(path, headers=headers).status_code == 401
    assert clickhouse.queries == []


# ── every query ─────────────────────────────────────────────────────────────


def test_every_query_reads_one_table_final_for_its_production_environment(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        assert _get(api, path, **SEPTEMBER).status_code == 200

    assert {_kind(sql) for sql, _ in clickhouse.queries} == {"totals", "new_users", "graph", "sessions", "outcomes"}
    for sql, params in clickhouse.queries:
        voice = "telemetry.voice_turns" in sql
        assert voice != ("telemetry.chat_turns" in sql)
        assert f"FROM {'telemetry.voice_turns' if voice else 'telemetry.chat_turns'} FINAL" in sql
        assert params == {
            "environment": "voice-production" if voice else "chat-production",
            "first_day": date(2026, 9, 1),
            "last_day": date(2026, 9, 30),
        }
        assert "environment = {environment:String}" in sql
        assert "{first_day:Date}" in sql and "{last_day:Date}" in sql
        # Values travel as parameters, never inside the SQL.
        assert "production" not in sql and "2026" not in sql


def test_questions_leave_out_voices_non_question_turns(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS[:3]:
        _get(api, path)

    for kind in ("totals", "graph", "sessions"):
        for sql in clickhouse.sql(kind):
            assert "countIf(outcome_class IS NULL OR outcome_class != 'non_question')" in sql


def test_other_environments_can_be_read(monkeypatch, api, configured):
    monkeypatch.setattr(settings, "telemetry_query_voice_environment", "voice-development")
    monkeypatch.setattr(settings, "telemetry_query_chat_environment", "chat-development")
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/outcomes")

    assert [params["environment"] for _, params in clickhouse.queries] == ["voice-development", "chat-development"]


@pytest.mark.parametrize("params", [{}, {"from": "2026-09-01"}, {"to": "2026-09-30"}])
def test_every_query_needs_a_range(monkeypatch, api, configured, params):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        assert api.get(path, params=params, headers={"X-API-Key": KEY}).status_code == 422
    assert clickhouse.queries == []


@pytest.mark.parametrize("path, params, status", [
    ("/api/telemetry/stats", {"from": "2026-09-30", "to": "2026-09-01"}, 400),
    ("/api/telemetry/graph", {"from": "2026-09-30", "to": "2026-09-01"}, 400),
    ("/api/telemetry/stats", {"from": "30-09-2026"}, 422),
    ("/api/telemetry/graph", {"granularity": "quarter"}, 422),
    ("/api/telemetry/graph", {"granularity": "toStartOfYear(timestamp)"}, 422),
])
def test_a_bad_request_is_refused_before_any_query(monkeypatch, api, configured, path, params, status):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert _get(api, path, **params).status_code == status
    assert clickhouse.queries == []


@pytest.mark.parametrize("path", PATHS)
def test_a_database_failure_is_502_and_the_client_is_closed(monkeypatch, api, configured, path):
    clickhouse = _use(monkeypatch, _ClickHouse(fail=ConnectionError("clickhouse down")))

    response = _get(api, path)

    assert response.status_code == 502
    assert response.json() == {"detail": "Telemetry database unavailable"}
    assert clickhouse.closed


def test_an_unreachable_database_is_502(monkeypatch, api, configured):
    def _unreachable():
        raise ConnectionError("no route to clickhouse")

    monkeypatch.setattr(telemetry_query, "dashboard_client", _unreachable)

    assert _get(api, "/api/telemetry/graph").status_code == 502


# ── /stats ──────────────────────────────────────────────────────────────────


def test_stats_counts_each_channel_and_adds_them_up(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        totals_voice=[(120, 40, 30)], totals_chat=[(80, 25, 20)],
        new_users_voice=[(7,)], new_users_chat=[(5,)],
    ))

    response = _get(api, "/api/telemetry/stats", **SEPTEMBER)

    assert response.status_code == 200
    assert response.json() == {
        "from": "2026-09-01",
        "to": "2026-09-30",
        "voice": {"questions": 120, "sessions": 40, "users": 30, "new_users": 7},
        "chat": {"questions": 80, "sessions": 25, "users": 20, "new_users": 5},
        "total": {"questions": 200, "sessions": 65, "users": 50, "new_users": 12},
    }
    assert len(clickhouse.queries) == 4
    assert clickhouse.closed


def test_stats_counts_distinct_sessions_and_hashed_users(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/stats")

    for sql in clickhouse.sql("totals"):
        assert "uniq(session_id) AS sessions" in sql and "uniq(user_id_hash) AS users" in sql


def test_a_new_user_is_first_seen_in_the_range(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/stats", **SEPTEMBER)

    for sql in clickhouse.sql("new_users"):
        assert "min(timestamp) AS first_seen" in sql
        assert "user_id_hash IS NOT NULL" in sql and "GROUP BY user_id_hash" in sql
        assert "WHERE toDate(first_seen) BETWEEN {first_day:Date} AND {last_day:Date}" in sql
        # The first turn is looked for in all of history, not just the range.
        assert sql.count("{first_day:Date}") == 1


def test_an_empty_answer_counts_zero(monkeypatch, api, configured):
    _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/stats").json()

    assert body["total"] == {"questions": 0, "sessions": 0, "users": 0, "new_users": 0}


# ── /graph ──────────────────────────────────────────────────────────────────


def test_graph_gives_each_channels_days_in_order(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        graph_voice=[(date(2026, 9, 1), 10, 4, 3, 1), (date(2026, 9, 2), 12, 5, 4, 0)],
        graph_chat=[(date(2026, 9, 2), 7, 3, 2, 2)],
    ))

    response = _get(api, "/api/telemetry/graph", **{"from": "2026-09-01", "to": "2026-09-02"})

    assert response.json() == {
        "from": "2026-09-01",
        "to": "2026-09-02",
        "granularity": "day",
        "voice": [
            {"start": "2026-09-01", "questions": 10, "sessions": 4, "users": 3, "failed": 1},
            {"start": "2026-09-02", "questions": 12, "sessions": 5, "users": 4, "failed": 0},
        ],
        "chat": [{"start": "2026-09-02", "questions": 7, "sessions": 3, "users": 2, "failed": 2}],
    }
    for sql in clickhouse.sql("graph"):
        assert "SELECT toDate(timestamp) AS start" in sql
        assert "uniq(session_id) AS sessions" in sql and "uniq(user_id_hash) AS users" in sql
        assert "countIf(outcome_class = 'failed') AS failed" in sql
        assert "GROUP BY start" in sql and "ORDER BY start" in sql
    assert clickhouse.closed


@pytest.mark.parametrize("granularity, bucket", [
    ("hour", "toStartOfHour(timestamp)"),
    ("day", "toDate(timestamp)"),
    ("week", "toMonday(timestamp)"),
    ("month", "toStartOfMonth(timestamp)"),
])
def test_graph_counts_each_bucket_in_clickhouse(monkeypatch, api, configured, granularity, bucket):
    clickhouse = _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/graph", granularity=granularity).json()

    assert body["granularity"] == granularity
    for sql in clickhouse.sql("graph"):
        # Unique users per week or month, not days' users added up.
        assert f"SELECT {bucket} AS start" in sql


def test_hours_start_at_a_utc_time(monkeypatch, api, configured):
    ist = timezone(timedelta(hours=5, minutes=30))
    _use(monkeypatch, _ClickHouse(
        graph_voice=[(datetime(2026, 9, 1, 10), 1, 1, 1, 0), (datetime(2026, 9, 1, 11, tzinfo=timezone.utc), 1, 1, 1, 0)],
        graph_chat=[(datetime(2026, 9, 1, 17, 30, tzinfo=ist), 1, 1, 1, 0)],
    ))

    body = _get(api, "/api/telemetry/graph", granularity="hour").json()

    assert [bucket["start"] for bucket in body["voice"]] == ["2026-09-01T10:00:00Z", "2026-09-01T11:00:00Z"]
    assert [bucket["start"] for bucket in body["chat"]] == ["2026-09-01T12:00:00Z"]


# ── /sessions ───────────────────────────────────────────────────────────────


def test_sessions_are_averaged_per_channel(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(sessions_voice=[(1.3333333, 100.0)], sessions_chat=[(None, None)]))

    body = _get(api, "/api/telemetry/sessions", **SEPTEMBER).json()

    assert body == {
        "from": "2026-09-01",
        "to": "2026-09-30",
        "voice": {"avg_questions": 1.33, "avg_seconds": 100.0},
        "chat": {"avg_questions": None, "avg_seconds": None},
    }
    for sql in clickhouse.sql("sessions"):
        assert "SELECT avgOrNull(questions), avgOrNull(seconds)" in sql
        assert "dateDiff('second', min(timestamp), max(timestamp)) AS seconds" in sql
        assert "session_id IS NOT NULL" in sql and "GROUP BY session_id" in sql
        assert "HAVING questions > 0" in sql


# ── /outcomes ───────────────────────────────────────────────────────────────


def test_outcomes_are_counted_per_class(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        outcomes_voice=[("delivered", 90, 30, 25), ("failed", 3, 2, 2), ("non_question", 40, 20, 15), ("not_recorded", 5, 2, 0)],
        outcomes_chat=[("delivered", 70, 20, 18), ("failed", 1, 1, 1)],
    ))

    body = _get(api, "/api/telemetry/outcomes", **SEPTEMBER).json()

    assert body["voice"] == {
        "delivered": {"turns": 90, "sessions": 30, "users": 25},
        "failed": {"turns": 3, "sessions": 2, "users": 2},
        "non_question": {"turns": 40, "sessions": 20, "users": 15},
        "not_recorded": {"turns": 5, "sessions": 2, "users": 0},
    }
    assert body["chat"] == {
        "delivered": {"turns": 70, "sessions": 20, "users": 18},
        "failed": {"turns": 1, "sessions": 1, "users": 1},
    }
    for sql in clickhouse.sql("outcomes"):
        assert "ifNull(outcome_class, 'not_recorded') AS outcome" in sql
        assert "count() AS turns" in sql
        assert "GROUP BY outcome" in sql


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


def test_the_app_serves_every_query():
    import main

    paths = {route.path for route in main.app.routes}
    assert set(PATHS) <= paths
