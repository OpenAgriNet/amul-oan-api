"""Read voice and chat traces from Langfuse's ClickHouse as the bundles the adapters take.

Only the columns the adapters need are read. The trace's user_id column never
leaves ClickHouse: it comes back as a placeholder, so the adapters still see
whether the trace had one. Chat reads its user id from metadata instead, and
the chat model hashes it straight away.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

REDACTED_USER_ID = "redacted"

# Child spans and scores can be written a while after the root.
_CHILD_WINDOW = timedelta(days=1)
# A c2 turn's question sits in a pretranslation trace up to 2 minutes away.
_RELATED_WINDOW = timedelta(minutes=2)
# How far a trace's timestamp can move between versions and still be read on its new day.
_MOVE_WINDOW = timedelta(days=1)
_BATCH_SIZE = 1000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Langfuse tables keep several versions of a row; the newest event_ts wins.
# A trace's timestamp can change between versions, so the trace queries pick the
# newest version from a day either side before keeping the day: otherwise an
# older version would put the trace in a day it has left.
_TRACES_SQL = """
SELECT id, name, timestamp_ms, session_id, user_id, metadata, is_deleted
FROM (
    SELECT id, name, timestamp, toUnixTimestamp64Milli(timestamp) AS timestamp_ms, session_id,
           if(ifNull(user_id, '') = '', NULL, {redacted:String}) AS user_id,
           metadata, is_deleted
    FROM traces
    WHERE environment = {environment:String}
      AND name IN {names:Array(String)}
      AND timestamp >= toDateTime64({window_start:String}, 3, 'UTC')
      AND timestamp < toDateTime64({window_end:String}, 3, 'UTC')
    ORDER BY event_ts DESC
    LIMIT 1 BY id
)
WHERE timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
"""

_CHAT_TRACES_SQL = """
SELECT id, name, timestamp_ms, session_id, user_id, metadata, input, output, is_deleted
FROM (
    SELECT id, name, timestamp, toUnixTimestamp64Milli(timestamp) AS timestamp_ms, session_id,
           if(ifNull(user_id, '') = '', NULL, {redacted:String}) AS user_id,
           metadata, input, output, is_deleted
    FROM traces
    WHERE environment = {environment:String}
      AND name IN {names:Array(String)}
      AND timestamp >= toDateTime64({window_start:String}, 3, 'UTC')
      AND timestamp < toDateTime64({window_end:String}, 3, 'UTC')
    ORDER BY event_ts DESC
    LIMIT 1 BY id
)
WHERE timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
"""

_OBSERVATIONS_SQL = """
SELECT id, trace_id, name, toUnixTimestamp64Milli(start_time) AS start_ms,
       if(end_time IS NULL, NULL, toUnixTimestamp64Milli(end_time)) AS end_ms, is_deleted
FROM observations
WHERE trace_id IN {trace_ids:Array(String)}
  AND start_time >= toDateTime64({start:String}, 3, 'UTC')
  AND start_time < toDateTime64({end:String}, 3, 'UTC')
ORDER BY event_ts DESC
LIMIT 1 BY id
"""

_SCORES_SQL = """
SELECT id, trace_id, name, value, string_value, is_deleted
FROM scores
WHERE trace_id IN {trace_ids:Array(String)}
  AND timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
ORDER BY event_ts DESC
LIMIT 1 BY id
"""

# Full rows only for the observations the chat adapters read (_find_agent_observation,
# _c2_answer and _tool_calls in telemetry_era_adapters.py). Everything else is
# read by name only, since prompts and model calls carry large inputs.
_CHAT_OBSERVATION_DETAILS_SQL = """
SELECT id, trace_id, type, name, metadata, input, output, is_deleted
FROM observations
WHERE trace_id IN {trace_ids:Array(String)}
  AND start_time >= toDateTime64({start:String}, 3, 'UTC')
  AND start_time < toDateTime64({end:String}, 3, 'UTC')
  AND (type = 'TOOL'
       OR startsWith(name, 'Amul AI Agent run')
       OR metadata['pipeline_stage'] = 'stream_translation'
       OR position(metadata['attributes'], 'Amul AI Agent') > 0)
ORDER BY event_ts DESC
LIMIT 1 BY id
"""


# Every live root trace of a day, whatever its name, so the import can account for
# each one in telemetry.trace_ledger. Only what identifies a trace: no input,
# output, user id or metadata beyond the schema stamp.
_TRACE_IDENTITIES_SQL = """
SELECT id, name, timestamp_ms, schema_version, outcome, is_deleted
FROM (
    SELECT id, ifNull(name, '') AS name, timestamp, toUnixTimestamp64Milli(timestamp) AS timestamp_ms,
           metadata['amul.schema_version'] AS schema_version, metadata['outcome'] AS outcome, is_deleted
    FROM traces
    WHERE environment = {environment:String}
      AND timestamp >= toDateTime64({window_start:String}, 3, 'UTC')
      AND timestamp < toDateTime64({window_end:String}, 3, 'UTC')
    ORDER BY event_ts DESC
    LIMIT 1 BY id
)
WHERE timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
"""

# A trace has no durable end-time field in Langfuse's trace table. The latest
# end time among its observations gives the privacy-safe root duration needed
# by the ledger, without selecting any observation input or output.
_TRACE_DURATION_OBSERVATIONS_SQL = """
SELECT id, trace_id, toUnixTimestamp64Milli(end_time) AS end_ms, is_deleted
FROM observations
WHERE trace_id IN {trace_ids:Array(String)}
  AND end_time IS NOT NULL
  AND start_time >= toDateTime64({start:String}, 3, 'UTC')
  AND start_time < toDateTime64({end:String}, 3, 'UTC')
ORDER BY event_ts DESC
LIMIT 1 BY id
"""


class ClickHouseReader(Protocol):
    def query(self, query: str, parameters: Mapping[str, Any] | None = None) -> Any: ...


@dataclass
class TraceBundle:
    trace: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    related_traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    name: str
    timestamp: datetime
    schema_version: str
    duration_ms: float | None = None
    outcome: str | None = None


def fetch_trace_identities(
    client: ClickHouseReader, *, environment: str, start: datetime, end: datetime
) -> list[TraceIdentity]:
    """Every live root trace with timestamp in [start, end), any name, newest version only."""
    parameters = {
        "environment": environment,
        "start": _sql_time(start),
        "end": _sql_time(end),
        **_move_window(start, end),
    }
    rows = _live_rows(client, _TRACE_IDENTITIES_SQL, parameters)
    end_ms_by_trace: dict[str, int] = {}
    child_window = {
        "start": _sql_time(start - _CHILD_WINDOW),
        "end": _sql_time(end + _CHILD_WINDOW),
    }
    for batch in _batches(rows, _BATCH_SIZE):
        for observation in _live_rows(
            client,
            _TRACE_DURATION_OBSERVATIONS_SQL,
            {"trace_ids": [row["id"] for row in batch], **child_window},
        ):
            end_ms = observation.get("end_ms")
            if isinstance(end_ms, int):
                trace_id = observation["trace_id"]
                end_ms_by_trace[trace_id] = max(end_ms_by_trace.get(trace_id, end_ms), end_ms)
    identities = []
    for row in rows:
        timestamp = _EPOCH + timedelta(milliseconds=row["timestamp_ms"])
        if start <= timestamp < end:
            duration_ms = end_ms_by_trace.get(row["id"])
            if duration_ms is not None and duration_ms >= row["timestamp_ms"]:
                duration_ms -= row["timestamp_ms"]
            else:
                duration_ms = None
            outcome = row.get("outcome")
            identities.append(
                TraceIdentity(
                    row["id"],
                    row.get("name") or "",
                    timestamp,
                    row.get("schema_version") or "",
                    float(duration_ms) if duration_ms is not None else None,
                    outcome if isinstance(outcome, str) and outcome else None,
                )
            )
    return identities


def fetch_voice_bundles(
    client: ClickHouseReader,
    *,
    environment: str,
    start: datetime,
    end: datetime,
    root_names: Iterable[str],
) -> Iterator[TraceBundle]:
    """Voice root traces with timestamp in [start, end), with their observation names and scores."""
    traces = _live_rows(
        client,
        _TRACES_SQL,
        {
            "redacted": REDACTED_USER_ID,
            "environment": environment,
            "names": sorted(root_names),
            "start": _sql_time(start),
            "end": _sql_time(end),
            **_move_window(start, end),
        },
    )
    child_window = {"start": _sql_time(start - _CHILD_WINDOW), "end": _sql_time(end + _CHILD_WINDOW)}
    for batch in _batches(traces, _BATCH_SIZE):
        ids = {"trace_ids": [row["id"] for row in batch], **child_window}
        observations = _by_trace(_live_rows(client, _OBSERVATIONS_SQL, ids))
        scores = _by_trace(_live_rows(client, _SCORES_SQL, ids))
        for row in batch:
            yield TraceBundle(
                trace=_trace(row),
                observations=[{"name": obs["name"]} for obs in observations.get(row["id"], [])],
                scores=[_score(score) for score in scores.get(row["id"], [])],
            )


def fetch_chat_bundles(
    client: ClickHouseReader,
    *,
    environment: str,
    start: datetime,
    end: datetime,
    root_names: Iterable[str],
) -> Iterator[TraceBundle]:
    """Chat root traces with timestamp in [start, end), with what the chat adapters read.

    Traces from just before start are read too, only as c2 pretranslation
    context for turns near the start of the window.
    """
    rows = _live_rows(
        client,
        _CHAT_TRACES_SQL,
        {
            "redacted": REDACTED_USER_ID,
            "environment": environment,
            "names": sorted(root_names),
            "start": _sql_time(start - _RELATED_WINDOW),
            "end": _sql_time(end),
            **_move_window(start - _RELATED_WINDOW, end),
        },
    )
    traces = [_chat_trace(row) for row in rows]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace["session_id"] and trace["metadata"].get("pipeline_stage") == "query_pretranslation":
            by_session[trace["session_id"]].append(trace)

    turns = [trace for trace in traces if trace["timestamp"] >= start]
    child_window = {"start": _sql_time(start - _CHILD_WINDOW), "end": _sql_time(end + _CHILD_WINDOW)}
    for batch in _batches(turns, _BATCH_SIZE):
        ids = {"trace_ids": [trace["id"] for trace in batch], **child_window}
        names = _by_trace(_live_rows(client, _OBSERVATIONS_SQL, ids))
        details = {row["id"]: row for row in _live_rows(client, _CHAT_OBSERVATION_DETAILS_SQL, ids)}
        scores = _by_trace(_live_rows(client, _SCORES_SQL, ids))
        for trace in batch:
            ordered = sorted(names.get(trace["id"], []), key=lambda obs: obs.get("start_ms") or 0)
            yield TraceBundle(
                trace=trace,
                observations=[
                    _chat_observation({**obs, **details.get(obs["id"], {})})
                    for obs in ordered
                ],
                scores=[_score(score) for score in scores.get(trace["id"], [])],
                related_traces=by_session.get(trace["session_id"], []) if trace["session_id"] else [],
            )


def _chat_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = _trace(row)
    trace["input"] = _decoded(row.get("input"))
    trace["output"] = _decoded(row.get("output"))
    return trace


def _chat_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {key: _decoded(value) for key, value in (row.get("metadata") or {}).items()}
    return {
        "id": row["id"],
        "type": row.get("type"),
        "name": row["name"],
        "metadata": metadata,
        "input": _decoded(row.get("input")),
        "output": _decoded(row.get("output")),
        "start_ms": row.get("start_ms"),
        "end_ms": row.get("end_ms"),
    }


def _decoded(value: Any) -> Any:
    """Langfuse keeps objects and strings as JSON text; give the adapters the value itself."""
    if isinstance(value, str) and value[:1] in ("{", "[", '"'):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _live_rows(client: ClickHouseReader, sql: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in client.query(sql, parameters=parameters).named_results() if not row.get("is_deleted")]


def _trace(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "timestamp": _EPOCH + timedelta(milliseconds=row["timestamp_ms"]),
        "session_id": row.get("session_id"),
        "user_id": row.get("user_id"),
        "metadata": dict(row.get("metadata") or {}),
    }


def _score(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": row["name"], "value": row.get("string_value") or row.get("value")}


def _by_trace(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trace_id"]].append(row)
    return grouped


def _batches(rows: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _move_window(start: datetime, end: datetime) -> dict[str, str]:
    return {"window_start": _sql_time(start - _MOVE_WINDOW), "window_end": _sql_time(end + _MOVE_WINDOW)}


def _sql_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
