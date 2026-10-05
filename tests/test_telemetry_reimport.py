"""Re-importing a day on real ClickHouse (chdb): the day ends as a first import would leave it.

Langfuse's tables and the telemetry tables from telemetry/clickhouse/voice.sql run
in one chdb session, and the import reads and writes them with its own SQL.
"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

chdb_session = pytest.importorskip("chdb.session")

from app.services.telemetry_era_registry import TelemetryEraRegistry, default_era_registry_path  # noqa: E402
from app.services.telemetry_import import CallerKey, import_voice_days  # noqa: E402
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary, load_voice_mappings  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ENV = "voice-development"

# The columns of Langfuse's tables the import reads, with Langfuse's engine.
LANGFUSE_SQL = """
CREATE TABLE traces (
    id String, name String, timestamp DateTime64(3), environment String,
    session_id Nullable(String), user_id Nullable(String), metadata Map(LowCardinality(String), String),
    is_deleted UInt8, event_ts DateTime64(3)
) ENGINE = ReplacingMergeTree(event_ts, is_deleted) ORDER BY id;
CREATE TABLE observations (
    id String, trace_id String, name String, start_time DateTime64(3), is_deleted UInt8, event_ts DateTime64(3)
) ENGINE = ReplacingMergeTree(event_ts, is_deleted) ORDER BY id;
CREATE TABLE scores (
    id String, trace_id String, name String, value Float64, string_value Nullable(String),
    timestamp DateTime64(3), is_deleted UInt8, event_ts DateTime64(3)
) ENGINE = ReplacingMergeTree(event_ts, is_deleted) ORDER BY id;
"""


class ClickHouse:
    """A chdb session standing in for the server, as both the import's reader and writer.
    Each test starts from empty tables."""

    def __init__(self, session):
        self.session = session
        self.session.query("SET output_format_json_quote_64bit_integers = 0")
        self.events = 0
        for table in ("traces", "observations", "scores"):
            self.session.query(f"DROP TABLE IF EXISTS default.{table}")
        self.session.query("DROP DATABASE IF EXISTS telemetry")
        for sql in (LANGFUSE_SQL, (REPO / "telemetry" / "clickhouse" / "voice.sql").read_text(encoding="utf-8")):
            for statement in re.sub(r"--[^\n]*", "", sql).split(";"):
                if statement.strip():
                    self.session.query(statement)

    def query(self, query, parameters=None):
        params = {name: _param(value) for name, value in (parameters or {}).items()}
        output = self.session.query(query, "JSONEachRow", params=params).bytes().decode()
        return _Result([json.loads(line) for line in output.splitlines() if line])

    def insert(self, table, data, column_names, database):
        lines = "\n".join(json.dumps(dict(zip(column_names, row)), default=_json) for row in data)
        self.session.query(f"INSERT INTO {database}.{table} ({', '.join(column_names)}) FORMAT JSONEachRow\n{lines}")

    def langfuse_trace(self, trace_id, when, *, metadata=None, is_deleted=0, environment=ENV):
        """A new version of a trace in Langfuse, newer than every one before it."""
        self.events += 1
        event_ts = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(seconds=self.events)
        row = [trace_id, "agent_journey", when, environment, "session-x", None, metadata or turn_metadata(), is_deleted, event_ts]
        columns = ["id", "name", "timestamp", "environment", "session_id", "user_id", "metadata", "is_deleted", "event_ts"]
        self.insert("traces", [row], column_names=columns, database="default")

    def turns(self, environment=ENV):
        """What a dashboard sees: (trace id, UTC day) of every turn, read with FINAL."""
        rows = self.query(
            "SELECT source_trace_id, toString(toDate(timestamp)) AS day FROM telemetry.voice_turns FINAL "
            "WHERE environment = {environment:String} ORDER BY day, source_trace_id",
            {"environment": environment},
        ).named_results()
        return [(row["source_trace_id"], row["day"]) for row in rows]

    def import_day(self, first_day, last_day=None, environment=ENV, caller_key=None):
        path = default_era_registry_path()
        return import_voice_days(
            self,
            self,
            environment=environment,
            first_day=first_day,
            last_day=last_day or first_day,
            registry=TelemetryEraRegistry.from_yaml(path, section="voice_eras"),
            vocabulary=VoiceOutcomeVocabulary.from_yaml(path),
            mappings=load_voice_mappings(),
            caller_key=caller_key or KEY,
        )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def named_results(self):
        return iter(self._rows)


def _param(value):
    if isinstance(value, (list, tuple)):
        return "[" + ",".join("'" + item.replace("\\", "\\\\").replace("'", "\\'") + "'" for item in value) + "]"
    if isinstance(value, datetime):
        return _json(value)
    return str(value)


def _json(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"can't send {value!r} to ClickHouse")


def turn_metadata(**overrides):
    metadata = {
        "amul.schema_version": "voice.turn.v1",
        "service": "voice-oan-api",
        "release": "unknown",
        "process_id": "3",
        "source_lang": "gu",
        "target_lang": "gu",
        "session_id": "session-x",
        "route": "agent",
        "outcome": "success",
        "total_ms": "1234.5",
        "query": json.dumps({"chars": 12, "sha256": "a" * 64}),
        "response": json.dumps({"chars": 30, "sha256": "b" * 64}),
    }
    metadata.update(overrides)
    return metadata


SEP_20 = date(2026, 9, 20)
KEY = CallerKey(b"k" * 32)


@pytest.fixture(scope="module")
def chdb_server(tmp_path_factory):
    # chdb runs one embedded server per process, on one data path.
    return chdb_session.Session(str(tmp_path_factory.mktemp("chdb")))


@pytest.fixture
def clickhouse(chdb_server):
    return ClickHouse(chdb_server)


def test_a_trace_deleted_in_langfuse_leaves_the_day(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00")
    clickhouse.import_day(SEP_20)

    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00", is_deleted=1)
    report = clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("A", "2026-09-20")]
    assert report.removed == 1
    assert "removed      1" in report.lines()


def test_a_turn_now_rejected_leaves_the_day(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00")
    clickhouse.import_day(SEP_20)

    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00", metadata=turn_metadata(**{"amul.schema_version": "voice.turn.v9"}))
    clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("A", "2026-09-20")]
    [day] = clickhouse.query("SELECT turns, rejected FROM telemetry.voice_import_days FINAL").named_results()
    assert (day["turns"], day["rejected"]) == (1, 1)


@pytest.mark.parametrize("order", ["new day first", "old day first"])
def test_a_turn_moved_to_the_next_day_is_kept_once(clickhouse, order):
    # Across a month, so the two rows sit in different partitions.
    clickhouse.langfuse_trace("A", "2026-09-30 23:59:00")
    clickhouse.langfuse_trace("B", "2026-09-30 10:00:00")
    clickhouse.import_day(date(2026, 9, 30))

    clickhouse.langfuse_trace("A", "2026-10-01 00:01:00")
    days = [date(2026, 10, 1), date(2026, 9, 30)]
    clickhouse.import_day(days[0] if order == "new day first" else days[1])

    assert ("A", "2026-09-30") not in clickhouse.turns()
    for day in days:
        clickhouse.import_day(day)
    assert clickhouse.turns() == [("B", "2026-09-30"), ("A", "2026-10-01")]


def test_a_turn_moved_within_its_day_keeps_one_row(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.import_day(SEP_20)

    clickhouse.langfuse_trace("A", "2026-09-20 10:00:05")
    report = clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("A", "2026-09-20")]
    [row] = clickhouse.query("SELECT toString(timestamp) AS at FROM telemetry.voice_turns FINAL").named_results()
    assert row["at"] == "2026-09-20 10:00:05.000"
    assert report.removed == 0


def test_a_removed_turn_comes_back_once_it_is_a_turn_again(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.import_day(SEP_20)
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00", metadata=turn_metadata(**{"amul.schema_version": "voice.turn.v9"}))
    clickhouse.import_day(SEP_20)

    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("A", "2026-09-20")]


def test_an_unchanged_day_reimports_to_the_same_turns(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00")
    clickhouse.import_day(SEP_20)

    report = clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("A", "2026-09-20"), ("B", "2026-09-20")]
    assert report.removed == 0


def test_a_reimport_leaves_other_days_and_environments_alone(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.langfuse_trace("C", "2026-09-21 10:00:00")
    clickhouse.langfuse_trace("P", "2026-09-20 10:00:00", environment="voice-production")
    clickhouse.import_day(SEP_20, date(2026, 9, 21))
    clickhouse.import_day(SEP_20, environment="voice-production")

    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00", is_deleted=1)
    clickhouse.import_day(SEP_20)

    assert clickhouse.turns() == [("C", "2026-09-21")]
    assert clickhouse.turns("voice-production") == [("P", "2026-09-20")]


def test_removed_turns_stay_removed_once_clickhouse_merges_the_table(clickhouse):
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00")
    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00")
    clickhouse.import_day(SEP_20)
    clickhouse.langfuse_trace("B", "2026-09-20 11:00:00", is_deleted=1)
    clickhouse.import_day(SEP_20)

    clickhouse.query("OPTIMIZE TABLE telemetry.voice_turns FINAL")

    assert clickhouse.turns() == [("A", "2026-09-20")]


def test_a_reimport_under_a_new_caller_key_moves_the_day_to_it(clickhouse):
    new_key = CallerKey(b"n" * 32)
    clickhouse.langfuse_trace("A", "2026-09-20 10:00:00", metadata=turn_metadata(user_id_hash="0" * 64))
    clickhouse.import_day(SEP_20)

    clickhouse.import_day(SEP_20, caller_key=new_key)

    [row] = clickhouse.query("SELECT user_id_hash, user_id_hash_key FROM telemetry.voice_turns FINAL").named_results()
    assert (row["user_id_hash"], row["user_id_hash_key"]) == (new_key.pseudonym("0" * 64), new_key.key_id)
