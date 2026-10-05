"""The dashboard service's queries: overview, breakdown, latency, tools, import
health and single turns.

ClickHouse is stood in for by a client that records each query and answers
from fixed rows, keyed by what the query is. These pin the rules every query
follows (a date range, FINAL, one environment unless asked for all, filters as
parameters, chat and voice kept apart) and the numbers each endpoint returns.
"""
import math
import os
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import get_args

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import telemetry_query as router_module
from app.services import telemetry_query

KEY = "dashboard-key"
SEPTEMBER = {"from": "2026-09-01", "to": "2026-09-30"}
PATHS = {
    "/api/telemetry/overview": {},
    "/api/telemetry/breakdown": {"by": "pipeline_profile"},
    "/api/telemetry/latency": {},
    "/api/telemetry/tools": {},
    "/api/telemetry/health": {},
    "/api/telemetry/turns": {"channel": "voice"},
}

# What each query is, by a piece only its SQL has. Checked in order.
_KINDS = (
    ("ARRAY JOIN mapKeys", "stages"),
    ("AS tool", "tools"),
    ("SELECT tool_call_count", "tool_calls"),
    ("max(imported_at)", "last_import"),
    ("traces, turns, rejected", "import_days"),
    ("disposition, trace_name, reason", "not_turns"),
    ("GROUP BY disposition", "ledger"),
    ("min(timestamp)", "coverage"),
    ("LIMIT {limit:UInt32}", "turns"),
    ("toDate(timestamp) AS day", "overview_days"),
    ("AS value, count() AS turns", "breakdown"),
    ("AS value, countIf(full_turn_latency_ms", "latency_by"),
    ("SELECT countIf(full_turn_latency_ms", "latency"),
    ("SELECT count() AS turns", "overview"),
    ("SELECT count() FROM", "count"),
)

# One row of the overview's metrics: turns, questions, delivered, failed,
# refused_or_blocked, non_question, outcome_not_recorded, rated, sessions,
# known_users, anonymous_turns, anonymous_sessions, [p50, p95].
_EMPTY_METRICS = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [math.nan, math.nan])
# What ClickHouse answers for an aggregate over no rows.
_DEFAULTS = {"overview": [_EMPTY_METRICS], "latency": [(0, [math.nan] * 3)], "count": [(0,)]}


def _kind(sql):
    return next(kind for marker, kind in _KINDS if marker in sql)


def _channel(sql, parameters):
    if "trace_ledger" in sql:
        return parameters["channel"]
    return "voice" if "voice_" in sql else "chat"


class _ClickHouse:
    """Answers each kind of query on each channel with the rows given for it,
    e.g. ``overview_voice=[...]``."""

    def __init__(self, fail=None, **rows):
        self.rows = rows
        self.fail = fail
        self.queries = []
        self.closed = False

    def query(self, query, parameters=None):
        if self.fail:
            raise self.fail
        parameters = dict(parameters or {})
        self.queries.append((query, parameters))
        kind = _kind(query)
        rows = self.rows.get(f"{kind}_{_channel(query, parameters)}", _DEFAULTS.get(kind, []))
        return SimpleNamespace(result_rows=list(rows))

    def close(self):
        self.closed = True

    def of(self, kind):
        return [(sql, params) for sql, params in self.queries if _kind(sql) == kind]


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
    return api.get(path, params={**SEPTEMBER, **PATHS.get(path, {}), **params}, headers={"X-API-Key": KEY})


# ── every query ─────────────────────────────────────────────────────────────


def test_every_query_reads_final_for_its_production_environment_and_range(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        assert _get(api, path).status_code == 200, path

    assert clickhouse.queries
    for sql, params in clickhouse.queries:
        assert " FINAL" in sql
        voice = _channel(sql, params) == "voice"
        assert params["environment"] == ("voice-production" if voice else "chat-production")
        assert "environment = {environment:String}" in sql
        if _kind(sql) != "last_import":
            assert (params["first_day"], params["last_day"]) == (date(2026, 9, 1), date(2026, 9, 30))
            assert "BETWEEN {first_day:Date} AND {last_day:Date}" in sql
        # Values travel as parameters, never inside the SQL.
        assert "production" not in sql and "2026" not in sql


def test_the_last_import_is_not_limited_to_the_range(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/health")

    for sql, params in clickhouse.of("last_import"):
        assert "first_day" not in sql and "first_day" not in params


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("params", [{"from": None, "to": None}, {"from": None}, {"to": None}])
def test_every_query_needs_a_range(monkeypatch, api, configured, path, params):
    clickhouse = _use(monkeypatch, _ClickHouse())
    query = {k: v for k, v in {**SEPTEMBER, **PATHS[path], **params}.items() if v is not None}

    assert api.get(path, params=query, headers={"X-API-Key": KEY}).status_code == 422
    assert clickhouse.queries == []


@pytest.mark.parametrize("path", list(PATHS))
def test_a_reversed_range_is_refused(monkeypatch, api, configured, path):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert _get(api, path, **{"from": "2026-09-30", "to": "2026-09-01"}).status_code == 400
    assert clickhouse.queries == []


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_a_missing_or_wrong_key_is_refused(monkeypatch, api, configured, path, headers):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert api.get(path, params={**SEPTEMBER, **PATHS[path]}, headers=headers).status_code == 401
    assert clickhouse.queries == []


@pytest.mark.parametrize("path", list(PATHS))
def test_a_database_failure_is_502_and_the_client_is_closed(monkeypatch, api, configured, path):
    clickhouse = _use(monkeypatch, _ClickHouse(fail=ConnectionError("clickhouse down")))

    response = _get(api, path)

    assert response.status_code == 502
    assert clickhouse.closed


def test_every_environment_can_be_read_together(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    for path in PATHS:
        _get(api, path, environment="all")

    for sql, params in clickhouse.queries:
        assert "environment = " not in sql and "environment" not in params


def test_one_other_environment_can_be_read(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/overview", environment="voice-development")

    assert {params["environment"] for _, params in clickhouse.queries} == {"voice-development"}


def test_filters_are_parameters_on_their_own_columns(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/overview", pipeline_profile="oss", source_lang="gu")

    for sql, params in clickhouse.queries:
        assert "pipeline_profile = {match_0:String}" in sql and "source_lang = {match_1:String}" in sql
        assert (params["match_0"], params["match_1"]) == ("oss", "gu")
        assert "oss" not in sql


def test_a_filter_a_channel_does_not_record_leaves_it_out(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/overview", route="greeting_fast_path").json()

    assert body["chat"] is None and body["voice"] is not None
    assert all(_channel(sql, params) == "voice" for sql, params in clickhouse.queries)
    assert all("route = {match_0:String}" in sql for sql, _ in clickhouse.queries)


def test_signed_in_and_chats_delivery_channel_read_their_columns(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/overview", signed_in="true")
    _get(api, "/api/telemetry/overview", chat_channel="whatsapp")

    voice_sql = [sql for sql, params in clickhouse.queries if _channel(sql, params) == "voice"]
    chat_sql = [sql for sql, params in clickhouse.queries if _channel(sql, params) == "chat"]
    assert voice_sql and all("toString(signed_in) = {match_0:String}" in sql for sql in voice_sql)
    assert chat_sql and all(" AND channel = {match_0:String}" in sql for sql in chat_sql)
    assert _get(api, "/api/telemetry/overview", signed_in="yes").status_code == 422


def test_service_and_release_filter_and_group_both_channels(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/overview", release="0f4cda4", service="amul-oan-api").json()
    by_release = _get(api, "/api/telemetry/breakdown", by="release").json()

    assert body["voice"] is not None and body["chat"] is not None
    assert by_release["voice"] == [] and by_release["chat"] == []
    for sql, params in clickhouse.of("overview"):
        assert "release = {match_0:String}" in sql and "service = {match_1:String}" in sql
        assert (params["match_0"], params["match_1"]) == ("0f4cda4", "amul-oan-api")
    assert {_channel(sql, params) for sql, params in clickhouse.of("breakdown")} == {"voice", "chat"}
    assert all("SELECT release AS value" in sql for sql, _ in clickhouse.of("breakdown"))


def test_the_filters_and_dimensions_cover_what_each_channel_records():
    recorded = set(telemetry_query.DIMENSIONS["voice"]) | set(telemetry_query.DIMENSIONS["chat"])

    assert set(router_module.TurnFilters.model_fields) - {"environment"} == recorded - {"environment"}
    assert set(get_args(router_module.Dimension)) == recorded


# ── /overview ───────────────────────────────────────────────────────────────


def test_overview_gives_the_totals_and_each_day(monkeypatch, api, configured):
    _use(monkeypatch, _ClickHouse(
        overview_voice=[(5, 4, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, [750.0, 2700.0])],
        overview_days_voice=[(date(2026, 9, 1), 4, 3, 1, 1, 0, 1, 1, 2, 2, 1, 1, 1, [1000.0, 2800.0])],
        overview_chat=[(3, 3, 2, 1, 0, 0, 0, 3, 2, 1, 1, 1, [5000.0, 5900.0])],
    ))

    body = _get(api, "/api/telemetry/overview").json()

    assert body["voice"]["total"] == {
        "turns": 5, "questions": 4, "delivered": 1, "failed": 1, "refused_or_blocked": 1,
        "non_question": 1, "outcome_not_recorded": 1, "delivered_rate": 0.3333,
        "sessions": 3, "known_users": 2, "anonymous_turns": 1, "anonymous_sessions": 1,
        "latency_p50_ms": 750.0, "latency_p95_ms": 2700.0,
    }
    assert body["voice"]["days"][0]["day"] == "2026-09-01"
    assert body["voice"]["days"][0]["delivered_rate"] == 0.5
    assert body["chat"]["total"]["delivered_rate"] == 0.6667
    assert body["chat"]["days"] == []


def test_a_rate_with_nothing_to_rate_and_empty_latency_are_null(monkeypatch, api, configured):
    _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/overview").json()

    total = body["voice"]["total"]
    assert total["turns"] == 0
    assert (total["delivered_rate"], total["latency_p50_ms"], total["latency_p95_ms"]) == (None, None, None)


def test_the_rate_counts_only_questions_with_an_outcome(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    _get(api, "/api/telemetry/overview")

    for sql, _ in clickhouse.of("overview"):
        assert "countIf(field_availability['outcome'] = 'recorded' AND outcome_class != 'non_question') AS rated" in sql
        assert "uniq(user_id_hash) AS known_users" in sql
        assert "countIf(user_id_hash IS NULL) AS anonymous_turns" in sql


# ── /breakdown ──────────────────────────────────────────────────────────────


def test_breakdown_gives_each_values_numbers_and_share(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(breakdown_voice=[
        ("oss", 3, 3, 1, 1, 0, 1, 0, 2, 1, 1, 0, 0, [1000.0, 2800.0]),
        (None, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, [math.nan, math.nan]),
    ]))

    body = _get(api, "/api/telemetry/breakdown", by="pipeline_profile").json()

    assert body["by"] == "pipeline_profile"
    oss, unknown = body["voice"]
    assert (oss["value"], oss["turns"], oss["share"], oss["delivered_rate"]) == ("oss", 3, 0.75, 0.5)
    assert (unknown["value"], unknown["share"], unknown["delivered_rate"], unknown["latency_p50_ms"]) == (None, 0.25, None, None)
    assert body["chat"] == []
    for sql, _ in clickhouse.of("breakdown"):
        assert "SELECT pipeline_profile AS value" in sql and "GROUP BY value" in sql


def test_a_breakdown_by_something_a_channel_does_not_record_is_null_for_it(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    body = _get(api, "/api/telemetry/breakdown", by="persona").json()

    assert body["voice"] is None and body["chat"] == []
    assert all(_channel(sql, params) == "chat" for sql, params in clickhouse.queries)


@pytest.mark.parametrize("by", ["user_id_hash", "question_sha256", "toString(1)", ""])
def test_breakdown_only_by_a_known_dimension(monkeypatch, api, configured, by):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert _get(api, "/api/telemetry/breakdown", by=by).status_code == 422
    assert clickhouse.queries == []


# ── /latency ────────────────────────────────────────────────────────────────


def test_latency_gives_quantiles_and_each_channels_stages(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        latency_voice=[(4, [750.0, 2700.0, 2940.0])],
        stages_voice=[("agent", 2, [1650.0, 2415.0, 2483.0])],
        latency_chat=[(2, [5000.0, 5900.0, 5980.0])],
        stages_chat=[("moderation", 2, [250.0, 290.0, 298.0])],
    ))

    body = _get(api, "/api/telemetry/latency").json()

    assert body["voice"]["total"] == {"timed_turns": 4, "p50_ms": 750.0, "p95_ms": 2700.0, "p99_ms": 2940.0}
    assert body["voice"]["stages"] == [{"stage": "agent", "turns": 2, "p50_ms": 1650.0, "p95_ms": 2415.0, "p99_ms": 2483.0}]
    assert body["chat"]["stages"] == [{"stage": "moderation", "turns": 2, "p50_ms": 250.0, "p95_ms": 290.0, "p99_ms": 298.0}]
    assert "by" not in body["voice"]
    stages = {_channel(sql, params): sql for sql, params in clickhouse.of("stages")}
    assert "FROM telemetry.voice_turns FINAL" in stages["voice"]
    assert "FROM telemetry.chat_turns FINAL" in stages["chat"]


def test_latency_by_a_dimension(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(latency_by_voice=[("managed", 1, [500.0, 500.0, 500.0])]))

    body = _get(api, "/api/telemetry/latency", by="route").json()

    assert body["voice"]["by"] == [{"value": "managed", "timed_turns": 1, "p50_ms": 500.0, "p95_ms": 500.0, "p99_ms": 500.0}]
    assert body["chat"] is None
    assert all("SELECT route AS value" in sql for sql, _ in clickhouse.of("latency_by"))


# ── /tools ──────────────────────────────────────────────────────────────────


def test_tools_count_turns_per_tool_and_per_number_of_calls(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        count_chat=[(4,)],
        tools_chat=[("search_documents", 2), ("weather", 1)],
        tool_calls_chat=[(None, 1), (0, 1), (3, 2)],
    ))

    body = _get(api, "/api/telemetry/tools").json()

    assert body["voice"] is None
    assert body["chat"] == {
        "turns": 4,
        "tools": [{"tool": "search_documents", "turns": 2, "share": 0.5}, {"tool": "weather", "turns": 1, "share": 0.25}],
        "tool_calls": [
            {"tool_calls": None, "turns": 1, "share": 0.25},
            {"tool_calls": 0, "turns": 1, "share": 0.25},
            {"tool_calls": 3, "turns": 2, "share": 0.5},
        ],
    }
    # A turn that called a tool twice counts once for it.
    (sql, _), = clickhouse.of("tools")
    assert "arrayDistinct(tool_names)" in sql


def test_tools_with_no_turns_have_no_share(monkeypatch, api, configured):
    _use(monkeypatch, _ClickHouse(tools_chat=[("weather", 0)]))

    body = _get(api, "/api/telemetry/tools").json()

    assert body["chat"]["tools"] == [{"tool": "weather", "turns": 0, "share": None}]


# ── /health ─────────────────────────────────────────────────────────────────


def test_health_reports_imports_dispositions_and_coverage(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse(
        import_days_voice=[("voice-production", date(2026, 9, 1), 8, 5, 1, datetime(2026, 9, 3, 1))],
        last_import_voice=[("voice-production", date(2026, 10, 1), datetime(2026, 10, 2, 1, tzinfo=timezone.utc))],
        ledger_voice=[("activity", 1), ("rejected", 1), ("turn", 5), ("unrecognised", 1)],
        not_turns_voice=[("rejected", "agent_journey", "Unknown voice schema version", 1)],
        coverage_voice=[("voice.turn.v1", "voice.v4", 4, datetime(2026, 9, 1, 10), datetime(2026, 9, 8, 9))],
    ))

    body = _get(api, "/api/telemetry/health").json()

    voice = body["voice"]
    assert voice["days"] == [{"environment": "voice-production", "day": "2026-09-01", "traces": 8, "turns": 5,
                              "rejected": 1, "imported_at": "2026-09-03T01:00:00Z"}]
    assert voice["last_import"] == [{"environment": "voice-production", "day": "2026-10-01",
                                     "imported_at": "2026-10-02T01:00:00Z"}]
    assert voice["traces"] == {"activity": 1, "rejected": 1, "turn": 5, "unrecognised": 1}
    assert voice["not_turns"] == [{"disposition": "rejected", "trace_name": "agent_journey",
                                   "reason": "Unknown voice schema version", "traces": 1}]
    assert voice["coverage"] == [{"schema_version": "voice.turn.v1", "source_era": "voice.v4", "turns": 4,
                                  "first": "2026-09-01T10:00:00Z", "last": "2026-09-08T09:00:00Z"}]
    assert body["chat"] == {"days": [], "last_import": [], "traces": {}, "not_turns": [], "coverage": []}
    for sql, params in clickhouse.of("ledger") + clickhouse.of("not_turns"):
        assert "channel = {channel:String}" in sql and params["channel"] in ("voice", "chat")
        assert "day BETWEEN" in sql


# ── /turns ──────────────────────────────────────────────────────────────────


def test_turns_are_a_page_of_metadata_newest_first(monkeypatch, api, configured):
    columns = telemetry_query.TURN_COLUMNS["voice"]
    row = {
        "source_trace_id": "v3", "timestamp": datetime(2026, 9, 1, 10, 5), "environment": "voice-production",
        "schema_version": "voice.turn.v1", "source_era": "voice.v4", "service": "voice-oan-api",
        "release": "3b19835", "session_id": "s1", "known_user": True,
        "signed_in": True, "provider": "RAYA", "call_type": "inbound", "route": None, "pipeline_profile": "oss",
        "source_lang": "gu", "target_lang": "gu", "outcome": "error", "outcome_class": "failed",
        "full_turn_latency_ms": 3000.0, "question_chars": 12, "answer_chars": 0,
    }
    names = [column.split(" AS ")[-1] for column in columns]
    clickhouse = _use(monkeypatch, _ClickHouse(count_voice=[(5,)], turns_voice=[tuple(row[name] for name in names)]))

    body = _get(api, "/api/telemetry/turns", limit=2, offset=1).json()

    assert (body["channel"], body["limit"], body["offset"], body["total"]) == ("voice", 2, 1, 5)
    assert body["rows"] == [{**row, "timestamp": "2026-09-01T10:05:00Z"}]
    (sql, params), = clickhouse.of("turns")
    assert "ORDER BY timestamp DESC, source_trace_id" in sql
    assert (params["limit"], params["offset"]) == (2, 1)
    assert all(_channel(sql, params) == "voice" for sql, params in clickhouse.queries)


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_a_turn_never_carries_text_its_hash_or_the_callers_id(channel):
    columns = " ".join(telemetry_query.TURN_COLUMNS[channel])

    for private in ("question_sha256", "answer_sha256", "question_sanitized", "answer_sanitized", "attributes"):
        assert private not in columns
    assert "user_id_hash IS NOT NULL" in columns
    assert [c for c in telemetry_query.TURN_COLUMNS[channel] if "user_id_hash" in c] == [
        "toBool(user_id_hash IS NOT NULL) AS known_user"
    ]


def test_turns_refuse_a_filter_the_channel_does_not_record(monkeypatch, api, configured):
    clickhouse = _use(monkeypatch, _ClickHouse())

    response = _get(api, "/api/telemetry/turns", channel="chat", route="greeting_fast_path")

    assert response.status_code == 400
    assert "route" in response.json()["detail"]
    assert clickhouse.queries == []


@pytest.mark.parametrize("params", [
    {"channel": "both"}, {"limit": 0}, {"limit": 201}, {"offset": -1}, {"offset": 100_001},
])
def test_a_bad_page_is_refused_before_any_query(monkeypatch, api, configured, params):
    clickhouse = _use(monkeypatch, _ClickHouse())

    assert _get(api, "/api/telemetry/turns", **params).status_code == 422
    assert clickhouse.queries == []


def test_the_app_serves_the_dashboard_queries():
    import main

    paths = {getattr(route, "path", None) for route in main.app.routes}
    assert set(PATHS) <= paths
