"""Read voice traces from Langfuse's ClickHouse as the bundles the adapters take.

Only the columns the adapters need are read. The raw user_id never leaves
ClickHouse: it comes back as a placeholder, so the adapters still see whether
the trace had one.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

REDACTED_USER_ID = "redacted"

# Child spans and scores can be written a while after the root.
_CHILD_WINDOW = timedelta(days=1)
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

_OBSERVATIONS_SQL = """
SELECT id, trace_id, name, is_deleted
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


class ClickHouseReader(Protocol):
    def query(self, query: str, parameters: Mapping[str, Any] | None = None) -> Any: ...


@dataclass
class TraceBundle:
    trace: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)


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
