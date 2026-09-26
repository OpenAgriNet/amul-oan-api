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
_BATCH_SIZE = 1000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Langfuse tables keep several versions of a row; the newest event_ts wins.
_TRACES_SQL = """
SELECT id, name, toUnixTimestamp64Milli(timestamp) AS timestamp_ms, session_id,
       if(ifNull(user_id, '') = '', NULL, {redacted:String}) AS user_id,
       metadata, is_deleted
FROM traces
WHERE environment = {environment:String}
  AND name IN {names:Array(String)}
  AND timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
ORDER BY event_ts DESC
LIMIT 1 BY id
"""

_CHAT_TRACES_SQL = """
SELECT id, name, toUnixTimestamp64Milli(timestamp) AS timestamp_ms, session_id,
       if(ifNull(user_id, '') = '', NULL, {redacted:String}) AS user_id,
       metadata, input, output, is_deleted
FROM traces
WHERE environment = {environment:String}
  AND name IN {names:Array(String)}
  AND timestamp >= toDateTime64({start:String}, 3, 'UTC')
  AND timestamp < toDateTime64({end:String}, 3, 'UTC')
ORDER BY event_ts DESC
LIMIT 1 BY id
"""

_OBSERVATIONS_SQL = """
SELECT id, trace_id, name, toUnixTimestamp64Milli(start_time) AS start_ms, is_deleted
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


class ClickHouseReader(Protocol):
    def query(self, query: str, parameters: Mapping[str, Any] | None = None) -> Any: ...


@dataclass
class TraceBundle:
    trace: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    related_traces: list[dict[str, Any]] = field(default_factory=list)


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
                observations=[_chat_observation(details.get(obs["id"]) or obs) for obs in ordered],
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


def _sql_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
