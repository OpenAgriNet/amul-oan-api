"""Read-only queries over the telemetry database, for the dashboards.

They follow "Querying, for dashboards" in docs/TELEMETRY_PIPELINE.md: read as
telemetry_dashboard, always FINAL, one environment per channel, days in UTC.
A question is a turn, leaving out voice's non_question turns (a stale
re-dispatch, non-speech, a greeting...); voice before v3 recorded no outcome,
so all of its turns count. A session is a distinct session id, and a user a
distinct user_id_hash, so anonymous callers are not users. Voice and chat hash
user ids with different salts, so someone who used both counts once in each.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Protocol

from app.config import settings

# Table names and SQL pieces come from here, never from a request.
TABLES = {"voice": "telemetry.voice_turns", "chat": "telemetry.chat_turns"}

# Graph buckets, each named by the day or hour (UTC) it starts at.
BUCKETS = {
    "hour": "toStartOfHour(timestamp)",
    "day": "toDate(timestamp)",
    "week": "toMonday(timestamp)",
    "month": "toStartOfMonth(timestamp)",
}

_QUESTIONS = "countIf(outcome_class IS NULL OR outcome_class != 'non_question')"

_COUNTS = f"{_QUESTIONS} AS questions, uniq(session_id) AS sessions, uniq(user_id_hash) AS users"

_IN_RANGE = """environment = {{environment:String}}
  AND toDate(timestamp) BETWEEN {{first_day:Date}} AND {{last_day:Date}}"""

_TOTALS_SQL = f"""
SELECT {_COUNTS}
FROM {{table}} FINAL
WHERE {_IN_RANGE}
"""

# A new user's first turn on record falls in the range.
_NEW_USERS_SQL = """
SELECT count()
FROM (
    SELECT min(timestamp) AS first_seen
    FROM {table} FINAL
    WHERE environment = {{environment:String}} AND user_id_hash IS NOT NULL
    GROUP BY user_id_hash
)
WHERE toDate(first_seen) BETWEEN {{first_day:Date}} AND {{last_day:Date}}
"""

# Unique sessions and users are counted per bucket: a week's users are not
# its days' users added up.
_GRAPH_SQL = f"""
SELECT {{bucket}} AS start, {_COUNTS}, countIf(outcome_class = 'failed') AS failed
FROM {{table}} FINAL
WHERE {_IN_RANGE}
GROUP BY start
ORDER BY start
"""

# Only sessions with at least one question.
_SESSIONS_SQL = f"""
SELECT avgOrNull(questions), avgOrNull(seconds)
FROM (
    SELECT {_QUESTIONS} AS questions,
           dateDiff('second', min(timestamp), max(timestamp)) AS seconds
    FROM {{table}} FINAL
    WHERE {_IN_RANGE}
      AND session_id IS NOT NULL
    GROUP BY session_id
    HAVING questions > 0
)
"""

_OUTCOMES_SQL = f"""
SELECT ifNull(outcome_class, 'not_recorded') AS outcome, count() AS turns,
       uniq(session_id) AS sessions, uniq(user_id_hash) AS users
FROM {{table}} FINAL
WHERE {_IN_RANGE}
GROUP BY outcome
ORDER BY outcome
"""


class TelemetryQueryClient(Protocol):
    def query(self, query: str, parameters: Mapping[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class Counts:
    questions: int
    sessions: int
    users: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.questions + other.questions,
            self.sessions + other.sessions,
            self.users + other.users,
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _environments() -> dict[str, str]:
    return {
        "voice": settings.telemetry_query_voice_environment,
        "chat": settings.telemetry_query_chat_environment,
    }


def _rows(client: TelemetryQueryClient, sql: str, channel: str, first_day: date, last_day: date) -> list:
    return client.query(
        sql,
        parameters={"environment": _environments()[channel], "first_day": first_day, "last_day": last_day},
    ).result_rows


def _start(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.isoformat()


def totals(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, Counts]:
    """Questions, sessions and users per channel between two UTC days, both included."""
    out = {}
    for channel, table in TABLES.items():
        rows = _rows(client, _TOTALS_SQL.format(table=table), channel, first_day, last_day)
        questions, sessions, users = rows[0] if rows else (0, 0, 0)
        out[channel] = Counts(int(questions), int(sessions), int(users))
    return out


def new_users(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, int]:
    """Users per channel whose first turn on record is in the range."""
    out = {}
    for channel, table in TABLES.items():
        rows = _rows(client, _NEW_USERS_SQL.format(table=table), channel, first_day, last_day)
        out[channel] = int(rows[0][0]) if rows else 0
    return out


def graph(
    client: TelemetryQueryClient, *, first_day: date, last_day: date, bucket: str = "day"
) -> dict[str, list[dict[str, Any]]]:
    """Questions, sessions, users and failed turns per bucket and channel.
    Buckets without turns are left out; the first and last can be partial."""
    out = {}
    for channel, table in TABLES.items():
        rows = _rows(client, _GRAPH_SQL.format(bucket=BUCKETS[bucket], table=table), channel, first_day, last_day)
        out[channel] = [
            {
                "start": _start(start),
                **Counts(int(questions), int(sessions_), int(users)).as_dict(),
                "failed": int(failed),
            }
            for start, questions, sessions_, users, failed in rows
        ]
    return out


def sessions(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, dict[str, float | None]]:
    """Average questions per session and seconds from its first turn to its
    last, per channel. None when there were no sessions."""
    out = {}
    for channel, table in TABLES.items():
        rows = _rows(client, _SESSIONS_SQL.format(table=table), channel, first_day, last_day)
        questions, seconds = rows[0] if rows else (None, None)
        out[channel] = {
            "avg_questions": None if questions is None else round(float(questions), 2),
            "avg_seconds": None if seconds is None else round(float(seconds), 2),
        }
    return out


def outcomes(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, dict[str, dict[str, int]]]:
    """Turns, sessions and users per outcome class and channel. Classes are
    those in telemetry/eras.yaml; turns without an outcome are not_recorded."""
    out = {}
    for channel, table in TABLES.items():
        rows = _rows(client, _OUTCOMES_SQL.format(table=table), channel, first_day, last_day)
        out[channel] = {
            outcome: {"turns": int(turns), "sessions": int(sessions_), "users": int(users)}
            for outcome, turns, sessions_, users in rows
        }
    return out


def dashboard_client() -> TelemetryQueryClient:
    """A ClickHouse client for the telemetry database as telemetry_dashboard."""
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.telemetry_clickhouse_host,
        port=settings.telemetry_clickhouse_port,
        username="telemetry_dashboard",
        password=settings.telemetry_dashboard_password,
        database="telemetry",
    )
