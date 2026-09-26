import importlib.util
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.models.telemetry_analytics import CanonicalChatTurn
from app.models.telemetry_voice_analytics import CanonicalVoiceTurn
from app.services import telemetry_fetcher
from app.services.telemetry_era_adapters import load_chat_mappings
from app.services.telemetry_era_registry import TelemetryEraRegistry, default_era_registry_path
from app.services.telemetry_fetcher import REDACTED_USER_ID, fetch_chat_bundles, fetch_voice_bundles
from app.services.telemetry_import import (
    CHAT_NOT_STORED,
    CHAT_TURN_COLUMNS,
    IMPORT_DAY_COLUMNS,
    NOT_STORED,
    REJECTION_COLUMNS,
    VOICE_TURN_COLUMNS,
    chat_turn_row,
    import_chat_days,
    import_voice_days,
    rejection_reason,
    voice_turn_row,
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


VOICE_SQL = (REPO / "telemetry" / "clickhouse" / "voice.sql").read_text(encoding="utf-8")

# voice_turns as released with voice.canonical.v1. Dashboards read these, so each
# keeps its name and type for good. Only ever add to this list.
RELEASED_VOICE_TURN_COLUMNS = {
    "source_trace_id": "String",
    "timestamp": "DateTime64(3, 'UTC')",
    "environment": "LowCardinality(String)",
    "schema_version": "LowCardinality(String)",
    "source_era": "LowCardinality(String)",
    "source_schema_version": "LowCardinality(String)",
    "source_era_extensions": "Array(LowCardinality(String))",
    "source_trace_name": "LowCardinality(String)",
    "session_id": "Nullable(String)",
    "process_id": "Nullable(String)",
    "user_id_hash": "Nullable(String)",
    "signed_in": "Nullable(Bool)",
    "provider": "LowCardinality(Nullable(String))",
    "call_type": "LowCardinality(Nullable(String))",
    "route": "LowCardinality(Nullable(String))",
    "pipeline_profile": "LowCardinality(Nullable(String))",
    "source_lang": "LowCardinality(Nullable(String))",
    "target_lang": "LowCardinality(Nullable(String))",
    "question_chars": "Nullable(UInt32)",
    "question_sha256": "Nullable(String)",
    "answer_chars": "Nullable(UInt32)",
    "answer_sha256": "Nullable(String)",
    "outcome": "LowCardinality(Nullable(String))",
    "outcome_class": "LowCardinality(Nullable(String))",
    "full_turn_latency_ms": "Nullable(Float64)",
    "stage_totals_ms": "Map(String, Float64)",
    "timings_ms": "Map(String, Float64)",
    "observation_names": "Array(String)",
    "score_names": "Array(String)",
    "field_availability": "Map(String, LowCardinality(String))",
    "imported_at": "DateTime64(3, 'UTC')",
}


def _table_columns(sql, table):
    """Column name -> type: the CREATE for the table, then each ALTER ... ADD COLUMN, in file order."""
    text = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    body = re.search(rf"CREATE TABLE IF NOT EXISTS telemetry\.{table}\s*\((.*?)\n\)", text, re.S).group(1)
    columns = {}
    for line in body.strip().splitlines():
        name, _, kind = line.strip().rstrip(",").partition(" ")
        columns[name] = kind
    for name, kind in re.findall(rf"ALTER TABLE telemetry\.{table} ADD COLUMN IF NOT EXISTS (\w+) ([^;]+);", text):
        columns[name] = kind.strip()
    return columns


@pytest.mark.parametrize(
    ("table", "columns"),
    [("voice_turns", VOICE_TURN_COLUMNS), ("voice_import_days", IMPORT_DAY_COLUMNS), ("voice_rejections", REJECTION_COLUMNS)],
)
def test_written_columns_match_the_tables(table, columns):
    assert tuple(_table_columns(VOICE_SQL, table)) == columns, (
        f"telemetry.{table} and the importer disagree. A new column goes at the end of both: an ALTER at the "
        "bottom of telemetry/clickhouse/voice.sql, and the end of the importer's column list."
    )


def test_released_columns_keep_their_name_and_type():
    current = _table_columns(VOICE_SQL, "voice_turns")
    changed = sorted(name for name, kind in RELEASED_VOICE_TURN_COLUMNS.items() if current.get(name) != kind)

    assert not changed, (
        f"telemetry.voice_turns changed the released columns {changed}. Dashboards read them, so add a new "
        "column instead. A change that can't be avoided needs a new table and a new canonical version."
    )


def test_every_canonical_field_is_stored_or_left_out_on_purpose():
    fields = set(CanonicalVoiceTurn.model_fields)
    missing = sorted(fields - set(VOICE_TURN_COLUMNS) - set(NOT_STORED))

    assert not missing, (
        f"CanonicalVoiceTurn has {missing}, but telemetry.voice_turns doesn't store it, so dashboards can't read "
        "it. Add a column (an ALTER in voice.sql, VOICE_TURN_COLUMNS and voice_turn_row), or list it in "
        "NOT_STORED with the reason."
    )
    assert set(NOT_STORED) <= fields, f"NOT_STORED lists {sorted(set(NOT_STORED) - fields)}, which isn't a field"


def test_a_row_has_exactly_the_table_columns():
    turn = CanonicalVoiceTurn(source_era="voice.v4", source_schema_version="voice.v4.v1", source_trace_name="agent_journey", timestamp=DAY_START)

    assert set(voice_turn_row(turn, environment="voice-development", imported_at=DAY_START)) == set(VOICE_TURN_COLUMNS)


def test_new_columns_are_read_from_alter_statements():
    sql = """
CREATE TABLE IF NOT EXISTS telemetry.voice_turns
(
    source_trace_id String,
    imported_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(imported_at);
--   ALTER TABLE telemetry.voice_turns ADD COLUMN IF NOT EXISTS example String;
ALTER TABLE telemetry.voice_turns ADD COLUMN IF NOT EXISTS farmer_type LowCardinality(Nullable(String));
"""

    assert _table_columns(sql, "voice_turns") == {
        "source_trace_id": "String",
        "imported_at": "DateTime64(3, 'UTC')",
        "farmer_type": "LowCardinality(Nullable(String))",
    }


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


# Chat


def chat_row(trace_id, *, name="chat.translation", when="2026-09-20T10:00:00Z", session="session-a", metadata=None, input=None, output=None):
    """A chat root as ClickHouse returns it: metadata values as strings, input and output as JSON text."""
    moment = datetime.fromisoformat(when.replace("Z", "+00:00"))
    return {
        "id": trace_id,
        "name": name,
        "timestamp_ms": int(moment.timestamp() * 1000),
        "session_id": session,
        "user_id": REDACTED_USER_ID,
        "metadata": metadata if metadata is not None else {"pipeline": "translation", "pipeline_profile": "oss", "user_id": "9990001112"},
        "input": json.dumps(input) if input is not None else None,
        "output": json.dumps(output) if output is not None else None,
        "is_deleted": 0,
    }


def chat_observation(obs_id, trace_id, *, name, type="SPAN", start_ms=0, metadata=None, output=None):
    return {
        "id": obs_id,
        "trace_id": trace_id,
        "type": type,
        "name": name,
        "start_ms": start_ms,
        "metadata": {key: json.dumps(value) if isinstance(value, dict) else value for key, value in (metadata or {}).items()},
        "input": json.dumps({"farmer_code": "<redacted>"}) if type == "TOOL" else None,
        "output": json.dumps(output) if output is not None else None,
        "is_deleted": 0,
    }


def _import_chat(client, *, writer=True, first_day=date(2026, 9, 20)):
    return import_chat_days(
        client,
        client if writer else None,
        environment="chat-development",
        first_day=first_day,
        last_day=first_day,
        registry=TelemetryEraRegistry.from_yaml(default_era_registry_path(), section="chat_eras"),
        mappings=load_chat_mappings(),
    )


def test_chat_rows_keep_no_phone_number_text_or_tool_data():
    client = FakeClickHouse(
        traces=[
            chat_row("c8", input={"query": "<redacted question>", "channel": "web"}, output="<redacted answer>"),
            chat_row("c4", name="Amul AI Agent", when="2026-07-23T10:00:00Z", input={"action": "<redacted action>"}, output="<redacted answer>"),
        ],
        observations=[
            chat_observation("tool", "c4", name="get_farmer_bonus_amount (redacted)", type="TOOL",
                             metadata={"attributes": {"gen_ai.tool.name": "get_farmer_bonus_amount"}}, output="<redacted tool response>"),
        ],
    )

    _import_chat(client, first_day=date(2026, 9, 20))
    _import_chat(client, first_day=date(2026, 7, 23))

    rows = {row["source_trace_id"]: row for row in client.inserts["chat_turns"]}
    assert rows["c8"]["source_era"] == "chat.c8"
    assert rows["c8"]["question_chars"] == len("<redacted question>")
    assert rows["c8"]["answer_chars"] == len("<redacted answer>")
    assert rows["c4"]["tool_names"] == ["get_farmer_bonus_amount"]
    assert rows["c4"]["tool_call_count"] == 1
    for row in rows.values():
        assert "9990001112" not in repr(row) and "<redacted" not in repr(row)
        assert row["user_id_hash"] and "user_id" not in row


def test_a_c2_turn_finds_its_question_in_a_pretranslation_just_before_midnight():
    client = FakeClickHouse(
        traces=[
            chat_row("question", when="2026-05-11T23:59:40Z", session="s1",
                     metadata={"pipeline_stage": "query_pretranslation"}, input={"text": "<redacted question>"}),
            chat_row("turn", when="2026-05-12T00:00:30Z", session="s1", metadata={"pipeline": "translation"}),
        ],
        observations=[
            chat_observation("agent", "turn", name="Amul AI Agent run (redacted)",
                             metadata={"attributes": {"agent_name": "Amul AI Agent", "final_result": "<redacted answer>"}}),
        ],
    )

    report = _import_chat(client, first_day=date(2026, 5, 12))

    (sql, params), *_ = client.queries
    assert params["start"] == "2026-05-11 23:58:00.000"
    [row] = client.inserts["chat_turns"]
    assert row["source_trace_id"] == "turn"
    assert row["question_chars"] == len("<redacted question>")
    assert report.traces == 1


def test_c2_traces_without_an_agent_run_are_rejected_with_the_reason():
    client = FakeClickHouse(traces=[chat_row("noise", when="2026-05-12T10:00:00Z", metadata={"pipeline_stage": "text_translation"})])

    report = _import_chat(client, writer=False, first_day=date(2026, 5, 12))

    [(name, reason)] = report.rejected
    assert name == "chat.translation" and reason.startswith("chat.c2 requires an 'Amul AI Agent run' observation")


def test_chat_reads_full_rows_only_for_the_observations_the_adapters_use():
    client = FakeClickHouse(traces=[chat_row("c8", when="2026-09-24T10:00:00Z", input={"query": "q"})])

    list(fetch_chat_bundles(client, environment="chat-development", start=DAY_START, end=DAY_END, root_names=["chat.translation"]))

    names_sql, details_sql = [sql for sql, _ in client.queries if "FROM observations" in sql]
    assert "input" not in names_sql.split("FROM")[0]
    assert "type = 'TOOL'" in details_sql and "stream_translation" in details_sql


CHAT_SQL = (REPO / "telemetry" / "clickhouse" / "chat.sql").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("table", "columns"),
    [("chat_turns", CHAT_TURN_COLUMNS), ("chat_import_days", IMPORT_DAY_COLUMNS), ("chat_rejections", REJECTION_COLUMNS)],
)
def test_written_chat_columns_match_the_tables(table, columns):
    assert tuple(_table_columns(CHAT_SQL, table)) == columns, (
        f"telemetry.{table} and the importer disagree. A new column goes at the end of both: an ALTER at the "
        "bottom of telemetry/clickhouse/chat.sql, and the end of the importer's column list."
    )


def test_every_canonical_chat_field_is_stored_or_left_out_on_purpose():
    fields = set(CanonicalChatTurn.model_fields)
    missing = sorted(fields - set(CHAT_TURN_COLUMNS) - set(CHAT_NOT_STORED))

    assert not missing, (
        f"CanonicalChatTurn has {missing}, but telemetry.chat_turns doesn't store it. Add a column (an ALTER in "
        "chat.sql, CHAT_TURN_COLUMNS and chat_turn_row), or list it in CHAT_NOT_STORED with the reason."
    )
    assert set(CHAT_NOT_STORED) <= fields, f"CHAT_NOT_STORED lists {sorted(set(CHAT_NOT_STORED) - fields)}, which isn't a field"


def test_a_chat_row_has_exactly_the_table_columns():
    turn = CanonicalChatTurn(source_era="chat.c8", source_schema_version="chat.c8.v1", source_trace_name="chat.translation", timestamp=DAY_START)

    assert set(chat_turn_row(turn, environment="chat-development", imported_at=DAY_START)) == set(CHAT_TURN_COLUMNS)


def test_the_script_takes_a_channel():
    assert _script()._parse_args(["--channel", "chat", "--env", "chat-production"]).channel == "chat"
    assert _script()._parse_args(["--env", "voice-production"]).channel == "voice"
