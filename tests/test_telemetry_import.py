import importlib.util
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.models.telemetry_voice_analytics import CanonicalVoiceTurn
from app.services import telemetry_fetcher
from app.services.telemetry_era_registry import TelemetryEraRegistry, default_era_registry_path
from app.services.telemetry_fetcher import REDACTED_USER_ID, fetch_voice_bundles
from app.services.telemetry_import import (
    IMPORT_DAY_COLUMNS,
    REJECTION_COLUMNS,
    VOICE_TURN_COLUMNS,
    import_voice_days,
    rejection_reason,
)
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary, load_voice_mappings

REPO = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))
DAY_START = datetime(2026, 9, 24, tzinfo=timezone.utc)
DAY_END = datetime(2026, 9, 25, tzinfo=timezone.utc)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def named_results(self):
        return iter(self._rows)


class FakeClickHouse:
    """Serves rows per table and records every query and insert."""

    def __init__(self, traces=(), observations=(), scores=()):
        self.tables = {"traces": list(traces), "observations": list(observations), "scores": list(scores)}
        self.queries = []
        self.inserts = {}

    def query(self, query, parameters=None):
        self.queries.append((query, parameters))
        rows = self.tables[re.search(r"FROM (\w+)", query).group(1)]
        if "trace_ids" in parameters:
            rows = [row for row in rows if row["trace_id"] in parameters["trace_ids"]]
        return FakeResult(rows)

    def insert(self, table, data, column_names, database):
        assert database == "telemetry"
        self.inserts.setdefault(table, []).extend(dict(zip(column_names, row)) for row in data)


def trace_row(trace_id, *, name="agent_journey", when="2026-09-24T10:00:00Z", metadata=None, is_deleted=0):
    moment = datetime.fromisoformat(when.replace("Z", "+00:00"))
    return {
        "id": trace_id,
        "name": name,
        "timestamp_ms": int(moment.timestamp() * 1000),
        "session_id": "session-redacted",
        "user_id": REDACTED_USER_ID,
        "metadata": metadata or {},
        "is_deleted": is_deleted,
    }


def turn_metadata(**overrides):
    """A voice turn's metadata as ClickHouse returns it: strings, with blocks as JSON."""
    metadata = {
        "process_id": "3",
        "provider": "<redacted-provider>",
        "source_lang": "gu",
        "target_lang": "gu",
        "user_id_hash": "0" * 64,
        "session_id": "session-redacted",
        "route": "agent",
        "outcome": "success",
        "pipeline_profile": "managed",
        "total_ms": "1234.5",
        "timings_ms": json.dumps({"ttft_ms": 900.0}),
        "stage_totals_ms": json.dumps({"agent": 800.0}),
        "agent": json.dumps({"signed_in": True, "tool_call_count": 1}),
        "query": json.dumps({"chars": 12, "sha256": "a" * 64, "preview": "<redacted question>"}),
        "response": json.dumps({"chars": 30, "sha256": "b" * 64, "preview": "<redacted answer>"}),
    }
    metadata.update(overrides)
    return metadata


def test_a_bundle_carries_its_observation_names_and_scores():
    client = FakeClickHouse(
        traces=[trace_row("t1", metadata=turn_metadata())],
        observations=[
            {"id": "o1", "trace_id": "t1", "name": "stream_translation", "is_deleted": 0},
            {"id": "o2", "trace_id": "other", "name": "unit_stage", "is_deleted": 0},
        ],
        scores=[{"id": "s1", "trace_id": "t1", "name": "turn_outcome", "value": 0.0, "string_value": "answered", "is_deleted": 0}],
    )

    [bundle] = fetch_voice_bundles(client, environment="voice-development", start=DAY_START, end=DAY_END, root_names=["agent_journey"])

    assert bundle.trace["id"] == "t1"
    assert bundle.trace["timestamp"] == datetime(2026, 9, 24, 10, tzinfo=timezone.utc)
    assert bundle.trace["user_id"] == REDACTED_USER_ID
    assert bundle.trace["metadata"]["outcome"] == "success"
    assert bundle.observations == [{"name": "stream_translation"}]
    assert bundle.scores == [{"name": "turn_outcome", "value": "answered"}]


def test_deleted_rows_are_skipped():
    client = FakeClickHouse(
        traces=[trace_row("t1"), trace_row("gone", is_deleted=1)],
        observations=[{"id": "o1", "trace_id": "t1", "name": "unit_stage", "is_deleted": 1}],
    )

    bundles = list(fetch_voice_bundles(client, environment="voice-development", start=DAY_START, end=DAY_END, root_names=["agent_journey"]))

    assert [bundle.trace["id"] for bundle in bundles] == ["t1"]
    assert bundles[0].observations == []


def test_the_raw_user_id_is_never_selected_and_times_are_utc():
    client = FakeClickHouse(traces=[trace_row("t1")])

    list(
        fetch_voice_bundles(
            client,
            environment="voice-development",
            start=datetime(2026, 9, 24, 5, 30, tzinfo=IST),
            end=datetime(2026, 9, 25, 5, 30, tzinfo=IST),
            root_names=["voice_request", "agent_journey"],
        )
    )

    (traces_sql, traces_params), (observations_sql, observations_params), _ = client.queries
    assert "NULL, {redacted:String}) AS user_id" in traces_sql
    assert "LIMIT 1 BY id" in traces_sql and "LIMIT 1 BY id" in observations_sql
    assert traces_params["start"] == "2026-09-24 00:00:00.000"
    assert traces_params["end"] == "2026-09-25 00:00:00.000"
    assert traces_params["names"] == ["agent_journey", "voice_request"]
    assert observations_params["start"] == "2026-09-23 00:00:00.000"
    assert observations_params["trace_ids"] == ["t1"]


def test_children_are_fetched_per_batch(monkeypatch):
    monkeypatch.setattr(telemetry_fetcher, "_BATCH_SIZE", 1)
    client = FakeClickHouse(traces=[trace_row("t1"), trace_row("t2")])

    list(fetch_voice_bundles(client, environment="voice-development", start=DAY_START, end=DAY_END, root_names=["agent_journey"]))

    child_queries = [params["trace_ids"] for sql, params in client.queries if "FROM observations" in sql]
    assert child_queries == [["t1"], ["t2"]]


def _import(client, *, writer=True, first_day=date(2026, 9, 24), last_day=None):
    path = default_era_registry_path()
    return import_voice_days(
        client,
        client if writer else None,
        environment="voice-development",
        first_day=first_day,
        last_day=last_day or first_day,
        registry=TelemetryEraRegistry.from_yaml(path, section="voice_eras"),
        vocabulary=VoiceOutcomeVocabulary.from_yaml(path),
        mappings=load_voice_mappings(),
    )


STAMP = {"amul.schema_version": "voice.turn.v1", "service": "voice-oan-api", "release": "unknown"}


def test_turns_are_written_without_the_phone_number_or_caller_text():
    client = FakeClickHouse(traces=[trace_row("stamped", metadata=turn_metadata(**STAMP)), trace_row("unstamped", metadata=turn_metadata())])

    report = _import(client)

    rows = {row["source_trace_id"]: row for row in client.inserts["voice_turns"]}
    assert rows["stamped"]["source_era"] == "voice.turn.v1"
    assert rows["unstamped"]["source_era"] == "voice.v4"
    for row in rows.values():
        assert "user_id" not in row
        assert "<redacted question>" not in repr(row) and "<redacted answer>" not in repr(row)
        assert (row["question_chars"], row["question_sha256"]) == (12, "a" * 64)
        assert (row["answer_chars"], row["answer_sha256"]) == (30, "b" * 64)
        assert row["field_availability"]["user_id"] == "recorded"
        assert row["outcome_class"] == "delivered"
        assert row["signed_in"] is True
        assert row["full_turn_latency_ms"] == 1234.5
    assert report.turns == 2 and report.written


def test_rejected_traces_are_counted_by_reason_not_raised():
    client = FakeClickHouse(
        traces=[
            trace_row("ok", metadata=turn_metadata()),
            trace_row("future", metadata=turn_metadata(**{"amul.schema_version": "voice.turn.v9"})),
            trace_row("early", name="agent_journey", when="2026-09-24T01:00:00Z", metadata=turn_metadata(**{"amul.schema_version": "voice.turn.v9"})),
        ]
    )

    report = _import(client)

    assert report.rejected == {("agent_journey", "Unknown voice schema version 'voice.turn.v9'"): 2}
    [rejection] = client.inserts["voice_rejections"]
    assert rejection["count"] == 2 and rejection["day"] == date(2026, 9, 24)
    [day] = client.inserts["voice_import_days"]
    assert (day["traces"], day["turns"], day["rejected"]) == (3, 1, 2)
    assert day["imported_at"] == rejection["imported_at"]


def test_a_trace_outside_every_era_is_rejected_without_its_timestamp_in_the_reason():
    client = FakeClickHouse(traces=[trace_row("old", when="2026-06-01T10:00:00Z", metadata=turn_metadata())])

    report = _import(client, first_day=date(2026, 6, 1))

    assert report.rejected == {("agent_journey", "No voice adapter registered for trace name='agent_journey'"): 1}


def test_a_dry_run_writes_nothing():
    client = FakeClickHouse(traces=[trace_row("t1", metadata=turn_metadata())])

    report = _import(client, writer=False)

    assert client.inserts == {}
    assert report.turns == 1
    assert "dry run" in report.lines()[0]


def test_each_utc_day_is_read_once():
    client = FakeClickHouse()

    _import(client, first_day=date(2026, 9, 24), last_day=date(2026, 9, 25))

    starts = [params["start"] for sql, params in client.queries if "FROM traces" in sql]
    assert starts == ["2026-09-24 00:00:00.000", "2026-09-25 00:00:00.000"]
    assert [day["turns"] for day in client.inserts["voice_import_days"]] == [0, 0]


def test_every_voice_root_name_is_fetched():
    client = FakeClickHouse()

    _import(client)

    names = next(params["names"] for sql, params in client.queries if "FROM traces" in sql)
    assert {"Voice Agent run", "Voice Agent Signed In run", "voice_request", "agent_journey"} <= set(names)


def test_validation_errors_become_one_line_reasons():
    with pytest.raises(Exception) as error:
        CanonicalVoiceTurn.model_validate({"source_era": "voice.v4"})

    assert rejection_reason(error.value).startswith("invalid trace: ")


def test_the_report_shows_counts_and_field_coverage():
    client = FakeClickHouse(traces=[trace_row("t1", metadata=turn_metadata())])

    lines = _import(client, writer=False).lines()

    assert "turns        1" in lines
    assert "  1  voice.v4" in lines
    assert "  1  delivered" in lines
    assert "  100.0%  outcome" in lines


def _ddl_columns(table):
    ddl = (REPO / "telemetry" / "clickhouse" / "voice.sql").read_text(encoding="utf-8")
    body = re.search(rf"CREATE TABLE IF NOT EXISTS telemetry\.{table}\s*\((.*?)\n\)", ddl, re.S).group(1)
    return tuple(line.split()[0] for line in body.strip().splitlines())


@pytest.mark.parametrize(
    ("table", "columns"),
    [("voice_turns", VOICE_TURN_COLUMNS), ("voice_import_days", IMPORT_DAY_COLUMNS), ("voice_rejections", REJECTION_COLUMNS)],
)
def test_written_columns_match_the_tables(table, columns):
    assert _ddl_columns(table) == columns


def _script():
    spec = importlib.util.spec_from_file_location("telemetry_import_script", REPO / "scripts" / "telemetry_import.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_imports_yesterday_by_default():
    args = _script()._parse_args(["--env", "voice-production"])

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    assert (args.first_day, args.last_day, args.dry_run) == (yesterday, yesterday, False)


def test_the_script_rejects_a_backwards_range():
    with pytest.raises(SystemExit):
        _script()._parse_args(["--env", "voice-production", "--from", "2026-09-25", "--to", "2026-09-24"])


def test_the_script_reads_the_password_from_the_env_or_a_file(monkeypatch, tmp_path):
    script = _script()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("TELEMETRY_READER_PASSWORD", raising=False)
    (tmp_path / ".telemetry_reader_password").write_text("from-file\n", encoding="utf-8")

    assert script._password("reader") == "from-file"
    monkeypatch.setenv("TELEMETRY_READER_PASSWORD", "from-env")
    assert script._password("reader") == "from-env"
