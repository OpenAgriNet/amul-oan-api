"""Read-only queries over the telemetry database, for the dashboards.

They follow "Querying, for dashboards" in docs/TELEMETRY_PIPELINE.md: read as
telemetry_dashboard, always FINAL, one environment per channel, days in UTC.
A question is a turn, a session a distinct session id, and a user a distinct
user_id_hash, so anonymous callers are not users. Voice and chat hash user ids
with different salts, so someone who used both counts once in each.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping, Protocol

from app.config import settings

# Table names come from here, never from a request.
TABLES = {"voice": "telemetry.voice_turns", "chat": "telemetry.chat_turns"}

_COUNTS = "count() AS questions, uniq(session_id) AS sessions, uniq(user_id_hash) AS users"

_TOTALS_SQL = """
SELECT {counts}
FROM {table} FINAL
WHERE environment = {{environment:String}}
  AND toDate(timestamp) BETWEEN {{first_day:Date}} AND {{last_day:Date}}
"""

_DAILY_SQL = """
SELECT toDate(timestamp) AS day, {counts}
FROM {table} FINAL
WHERE environment = {{environment:String}}
  AND day BETWEEN {{first_day:Date}} AND {{last_day:Date}}
GROUP BY day
ORDER BY day
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


def _parameters(channel: str, first_day: date, last_day: date) -> dict[str, Any]:
    return {"environment": _environments()[channel], "first_day": first_day, "last_day": last_day}


def totals(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, Counts]:
    """Questions, sessions and users per channel between two UTC days, both included."""
    out = {}
    for channel, table in TABLES.items():
        rows = client.query(
            _TOTALS_SQL.format(counts=_COUNTS, table=table),
            parameters=_parameters(channel, first_day, last_day),
        ).result_rows
        questions, sessions, users = rows[0] if rows else (0, 0, 0)
        out[channel] = Counts(int(questions), int(sessions), int(users))
    return out


def daily(client: TelemetryQueryClient, *, first_day: date, last_day: date) -> dict[str, list[tuple[date, Counts]]]:
    """The same counts per UTC day, per channel. Days without turns are left out."""
    out = {}
    for channel, table in TABLES.items():
        rows = client.query(
            _DAILY_SQL.format(counts=_COUNTS, table=table),
            parameters=_parameters(channel, first_day, last_day),
        ).result_rows
        out[channel] = [
            (day, Counts(int(questions), int(sessions), int(users)))
            for day, questions, sessions, users in rows
        ]
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
