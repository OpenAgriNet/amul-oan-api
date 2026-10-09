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

# Exact counts: plain uniq() estimates above 65,536 values, about 0.5% off at a
# month of chat users.
_COUNTS = f"{_QUESTIONS} AS questions, uniqExact(session_id) AS sessions, uniqExact(user_id_hash) AS users"

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
       uniqExact(session_id) AS sessions, uniqExact(user_id_hash) AS users
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


# ── filters, breakdowns, latency, tools, import health and single turns ─────

# What a request may group by or match on, per channel, and the column each one
# reads. Only these names reach the SQL; the values travel as parameters.
_COMMON_DIMENSIONS = {
    "environment": "environment",
    "schema_version": "schema_version",
    "source_era": "source_era",
    "pipeline_profile": "pipeline_profile",
    "source_lang": "source_lang",
    "target_lang": "target_lang",
    "outcome": "outcome",
    "outcome_class": "outcome_class",
    # Null on turns sent before the service and release stamps.
    "service": "service",
    "release": "release",
}
DIMENSIONS = {
    "voice": {
        **_COMMON_DIMENSIONS,
        "route": "route",
        "call_type": "call_type",
        "provider": "provider",
        "signed_in": "toString(signed_in)",
    },
    "chat": {
        **_COMMON_DIMENSIONS,
        # Web or WhatsApp. "Channel" in this API is voice or chat.
        "chat_channel": "channel",
        "pipeline": "pipeline",
        "persona": "persona",
        "served_tier": "served_tier",
    },
}

#: ``environment`` value that reads every environment instead of one.
ALL_ENVIRONMENTS = "all"

# A rate counts only questions whose outcome was recorded (voice had none before v3).
_METRICS = """count() AS turns,
    countIf(outcome_class IS NULL OR outcome_class != 'non_question') AS questions,
    countIf(outcome_class = 'delivered') AS delivered,
    countIf(outcome_class = 'failed') AS failed,
    countIf(outcome_class = 'refused_or_blocked') AS refused_or_blocked,
    countIf(outcome_class = 'non_question') AS non_question,
    countIf(field_availability['outcome'] != 'recorded') AS outcome_not_recorded,
    countIf(field_availability['outcome'] = 'recorded' AND outcome_class != 'non_question') AS rated,
    uniqExact(session_id) AS sessions,
    uniqExact(user_id_hash) AS known_users,
    countIf(user_id_hash IS NULL) AS anonymous_turns,
    uniqExactIf(session_id, user_id_hash IS NULL) AS anonymous_sessions,
    quantiles(0.5, 0.95)(full_turn_latency_ms) AS latency_ms"""

_OVERVIEW_SQL = "SELECT {metrics} FROM {table} FINAL WHERE {where}"

_OVERVIEW_DAYS_SQL = """
SELECT toDate(timestamp) AS day, {metrics}
FROM {table} FINAL
WHERE {where}
GROUP BY day
ORDER BY day
"""

_BREAKDOWN_SQL = """
SELECT {column} AS value, {metrics}
FROM {table} FINAL
WHERE {where}
GROUP BY value
ORDER BY turns DESC, value
"""

_LATENCY = "countIf(full_turn_latency_ms IS NOT NULL) AS timed_turns, quantiles(0.5, 0.95, 0.99)(full_turn_latency_ms)"

_LATENCY_SQL = "SELECT {latency} FROM {table} FINAL WHERE {where}"

_LATENCY_BY_SQL = """
SELECT {column} AS value, {latency}
FROM {table} FINAL
WHERE {where}
GROUP BY value
ORDER BY value
"""

# Voice records its stage times; chat's are worked out at import from the
# trace's spans.
_STAGES_SQL = """
SELECT stage, count() AS turns, quantiles(0.5, 0.95, 0.99)(ms)
FROM {table} FINAL
ARRAY JOIN mapKeys(stage_totals_ms) AS stage, mapValues(stage_totals_ms) AS ms
WHERE {where}
GROUP BY stage
ORDER BY stage
"""

_COUNT_SQL = "SELECT count() FROM {table} FINAL WHERE {where}"

# Only chat keeps tool names. A turn that called a tool twice counts once for it.
_TOOLS_SQL = """
SELECT tool, count() AS turns
FROM telemetry.chat_turns FINAL
ARRAY JOIN arrayDistinct(tool_names) AS tool
WHERE {where}
GROUP BY tool
ORDER BY turns DESC, tool
"""

_TOOL_CALLS_SQL = """
SELECT tool_call_count, count() AS turns
FROM telemetry.chat_turns FINAL
WHERE {where}
GROUP BY tool_call_count
ORDER BY tool_call_count
"""

_IMPORT_DAYS_SQL = """
SELECT environment, day, traces, turns, rejected, imported_at
FROM telemetry.{channel}_import_days FINAL
WHERE {where}
ORDER BY environment, day
"""

# Not limited to the range: each environment's most recent import, the day it
# wrote and when, from the same row. Of two days written in the same
# millisecond, the newer day counts as the last.
_LAST_IMPORT_SQL = """
SELECT environment, argMax(day, (imported_at, day)) AS last_day, max(imported_at) AS last_imported_at
FROM telemetry.{channel}_import_days FINAL
WHERE {where}
GROUP BY environment
ORDER BY environment
"""

_LEDGER_SQL = """
SELECT disposition, count() AS traces
FROM telemetry.trace_ledger FINAL
WHERE channel = {{channel:String}} AND {where}
GROUP BY disposition
ORDER BY disposition
"""

_NOT_TURNS_SQL = """
SELECT disposition, trace_name, reason, count() AS traces
FROM telemetry.trace_ledger FINAL
WHERE channel = {{channel:String}} AND disposition IN ('rejected', 'unrecognised') AND {where}
GROUP BY disposition, trace_name, reason
ORDER BY traces DESC, disposition, trace_name, reason
"""

_COVERAGE_SQL = """
SELECT schema_version, source_era, count() AS turns, min(timestamp), max(timestamp)
FROM {table} FINAL
WHERE {where}
GROUP BY schema_version, source_era
ORDER BY schema_version, source_era
"""

# A turn's metadata for the drill-down: never its text, the hashes of its text,
# or the caller's hashed id.
TURN_COLUMNS = {
    "voice": (
        "source_trace_id", "timestamp", "environment", "schema_version", "source_era", "service", "release",
        "session_id", "toBool(user_id_hash IS NOT NULL) AS known_user", "signed_in", "provider", "call_type",
        "route", "pipeline_profile", "source_lang", "target_lang", "outcome", "outcome_class",
        "full_turn_latency_ms", "question_chars", "answer_chars",
    ),
    "chat": (
        "source_trace_id", "timestamp", "environment", "schema_version", "source_era", "service", "release",
        "session_id", "toBool(user_id_hash IS NOT NULL) AS known_user", "channel", "pipeline", "persona",
        "pipeline_profile", "source_lang", "target_lang", "outcome", "outcome_class", "served_tier",
        "full_turn_latency_ms", "tool_names", "tool_call_count", "question_chars", "answer_chars",
    ),
}

_TURNS_SQL = """
SELECT {columns}
FROM {table} FINAL
WHERE {where}
ORDER BY timestamp DESC, source_trace_id
LIMIT {{limit:UInt32}} OFFSET {{offset:UInt32}}
"""


def _environment(channel: str, environment: str | None) -> tuple[list[str], dict[str, Any]]:
    """The condition for one environment, none for all of them. No
    ``environment`` means the channel's production one."""
    environment = environment or _environments()[channel]
    if environment == ALL_ENVIRONMENTS:
        return [], {}
    return ["environment = {environment:String}"], {"environment": environment}


def _where(
    channel: str,
    *,
    first_day: date,
    last_day: date,
    environment: str | None,
    match: Mapping[str, str] | None = None,
    day: str = "toDate(timestamp)",
) -> tuple[str, dict[str, Any]] | None:
    """The WHERE clause and its parameters, or None when ``match`` names
    something the channel doesn't record."""
    columns = DIMENSIONS[channel]
    match = match or {}
    if any(name not in columns for name in match):
        return None
    conditions, parameters = _environment(channel, environment)
    conditions.insert(0, f"{day} BETWEEN {{first_day:Date}} AND {{last_day:Date}}")
    parameters.update(first_day=first_day, last_day=last_day)
    for index, (name, value) in enumerate(sorted(match.items())):
        conditions.append(f"{columns[name]} = {{match_{index}:String}}")
        parameters[f"match_{index}"] = value
    return " AND ".join(conditions), parameters


def _number(value: Any) -> float | None:
    """A float rounded for JSON, or None for NULL and NaN (an empty quantile)."""
    if value is None or value != value:
        return None
    return round(float(value), 2)


def _quantiles(values: Any, names: tuple[str, ...]) -> dict[str, float | None]:
    values = list(values or ()) or [None] * len(names)
    return {name: _number(value) for name, value in zip(names, values)}


def _metrics(row: tuple) -> dict[str, Any]:
    (turns, questions, delivered, failed, refused, non_question, not_recorded, rated,
     sessions_, known_users, anonymous_turns, anonymous_sessions, latency) = row
    return {
        "turns": int(turns),
        "questions": int(questions),
        "delivered": int(delivered),
        "failed": int(failed),
        "refused_or_blocked": int(refused),
        "non_question": int(non_question),
        "outcome_not_recorded": int(not_recorded),
        "delivered_rate": round(delivered / rated, 4) if rated else None,
        "sessions": int(sessions_),
        "known_users": int(known_users),
        "anonymous_turns": int(anonymous_turns),
        "anonymous_sessions": int(anonymous_sessions),
        **_quantiles(latency, ("latency_p50_ms", "latency_p95_ms")),
    }


def _value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return _start(value)
    return value


def overview(
    client: TelemetryQueryClient,
    *,
    first_day: date,
    last_day: date,
    environment: str | None = None,
    match: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Each channel's metrics for the whole range and for each day in it.
    None for a channel that doesn't record something ``match`` names."""
    out: dict[str, dict[str, Any] | None] = {}
    for channel, table in TABLES.items():
        clause = _where(channel, first_day=first_day, last_day=last_day, environment=environment, match=match)
        if clause is None:
            out[channel] = None
            continue
        where, parameters = clause
        (total,) = client.query(_OVERVIEW_SQL.format(metrics=_METRICS, table=table, where=where), parameters=parameters).result_rows
        days = client.query(_OVERVIEW_DAYS_SQL.format(metrics=_METRICS, table=table, where=where), parameters=parameters).result_rows
        out[channel] = {
            "total": _metrics(total),
            "days": [{"day": _start(row[0]), **_metrics(row[1:])} for row in days],
        }
    return out


def breakdown(
    client: TelemetryQueryClient,
    *,
    first_day: date,
    last_day: date,
    by: str,
    environment: str | None = None,
    match: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, Any]] | None]:
    """The overview's metrics per value of ``by``, most turns first, with each
    value's share of the turns. None for a channel that doesn't record ``by``."""
    out: dict[str, list[dict[str, Any]] | None] = {}
    for channel, table in TABLES.items():
        clause = _where(channel, first_day=first_day, last_day=last_day, environment=environment, match=match)
        if clause is None or by not in DIMENSIONS[channel]:
            out[channel] = None
            continue
        where, parameters = clause
        sql = _BREAKDOWN_SQL.format(column=DIMENSIONS[channel][by], metrics=_METRICS, table=table, where=where)
        rows = [{"value": row[0], **_metrics(row[1:])} for row in client.query(sql, parameters=parameters).result_rows]
        total = sum(row["turns"] for row in rows)
        out[channel] = [{**row, "share": round(row["turns"] / total, 4)} for row in rows]
    return out


_LATENCY_NAMES = ("p50_ms", "p95_ms", "p99_ms")


def _latency(timed_turns: Any, values: Any) -> dict[str, Any]:
    return {"timed_turns": int(timed_turns), **_quantiles(values, _LATENCY_NAMES)}


def latency(
    client: TelemetryQueryClient,
    *,
    first_day: date,
    last_day: date,
    by: str | None = None,
    environment: str | None = None,
    match: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Full-turn latency quantiles per channel, per value of ``by`` if given,
    and per stage."""
    out: dict[str, dict[str, Any] | None] = {}
    for channel, table in TABLES.items():
        clause = _where(channel, first_day=first_day, last_day=last_day, environment=environment, match=match)
        if clause is None or (by is not None and by not in DIMENSIONS[channel]):
            out[channel] = None
            continue
        where, parameters = clause
        (total,) = client.query(_LATENCY_SQL.format(latency=_LATENCY, table=table, where=where), parameters=parameters).result_rows
        result: dict[str, Any] = {"total": _latency(*total)}
        if by is not None:
            sql = _LATENCY_BY_SQL.format(column=DIMENSIONS[channel][by], latency=_LATENCY, table=table, where=where)
            result["by"] = [{"value": value, **_latency(timed, values)} for value, timed, values in client.query(sql, parameters=parameters).result_rows]
        rows = client.query(_STAGES_SQL.format(table=table, where=where), parameters=parameters).result_rows
        result["stages"] = [{"stage": stage, "turns": int(turns), **_quantiles(values, _LATENCY_NAMES)} for stage, turns, values in rows]
        out[channel] = result
    return out


def tools(
    client: TelemetryQueryClient,
    *,
    first_day: date,
    last_day: date,
    environment: str | None = None,
    match: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Turns that used each tool and turns by how many tool calls they made,
    with their share of all turns. Only chat keeps tool names, so voice is None."""
    out: dict[str, dict[str, Any] | None] = {"voice": None}
    clause = _where("chat", first_day=first_day, last_day=last_day, environment=environment, match=match)
    if clause is None:
        out["chat"] = None
        return out
    where, parameters = clause
    table = TABLES["chat"]
    ((turns,),) = client.query(_COUNT_SQL.format(table=table, where=where), parameters=parameters).result_rows
    turns = int(turns)

    def share(count: int) -> float | None:
        return round(count / turns, 4) if turns else None

    used = client.query(_TOOLS_SQL.format(where=where), parameters=parameters).result_rows
    calls = client.query(_TOOL_CALLS_SQL.format(where=where), parameters=parameters).result_rows
    out["chat"] = {
        "turns": turns,
        "tools": [{"tool": tool, "turns": int(count), "share": share(int(count))} for tool, count in used],
        "tool_calls": [
            {"tool_calls": None if n is None else int(n), "turns": int(count), "share": share(int(count))}
            for n, count in calls
        ],
    }
    return out


def health(
    client: TelemetryQueryClient,
    *,
    first_day: date,
    last_day: date,
    environment: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Per channel: each day's import, when each environment was last imported,
    what became of every root trace (turn, rejected, activity, unrecognised),
    why traces were rejected or not recognised, and which schema versions and
    eras the turns came in."""
    out: dict[str, dict[str, Any]] = {}
    for channel, table in TABLES.items():
        by_day, parameters = _where(channel, first_day=first_day, last_day=last_day, environment=environment, day="day")
        by_turn, _ = _where(channel, first_day=first_day, last_day=last_day, environment=environment)
        conditions, environment_parameters = _environment(channel, environment)
        ledger_parameters = {**parameters, "channel": channel}
        import_days = client.query(_IMPORT_DAYS_SQL.format(channel=channel, where=by_day), parameters=parameters)
        last_import = client.query(
            _LAST_IMPORT_SQL.format(channel=channel, where=" AND ".join(conditions) or "1"),
            parameters=environment_parameters,
        )
        ledger = client.query(_LEDGER_SQL.format(where=by_day), parameters=ledger_parameters)
        not_turns = client.query(_NOT_TURNS_SQL.format(where=by_day), parameters=ledger_parameters)
        coverage = client.query(_COVERAGE_SQL.format(table=table, where=by_turn), parameters=parameters)
        out[channel] = {
            "days": [
                {"environment": env, "day": _start(day), "traces": int(traces), "turns": int(count),
                 "rejected": int(rejected), "imported_at": _start(imported_at)}
                for env, day, traces, count, rejected, imported_at in import_days.result_rows
            ],
            "last_import": [
                {"environment": env, "day": _start(day), "imported_at": _start(imported_at)}
                for env, day, imported_at in last_import.result_rows
            ],
            "traces": {disposition: int(count) for disposition, count in ledger.result_rows},
            "not_turns": [
                {"disposition": disposition, "trace_name": name, "reason": reason, "traces": int(count)}
                for disposition, name, reason, count in not_turns.result_rows
            ],
            "coverage": [
                {"schema_version": version, "source_era": era, "turns": int(count),
                 "first": _start(first), "last": _start(last)}
                for version, era, count, first, last in coverage.result_rows
            ],
        }
    return out


def turn_page(
    client: TelemetryQueryClient,
    *,
    channel: str,
    first_day: date,
    last_day: date,
    limit: int,
    offset: int,
    environment: str | None = None,
    match: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """One page of a channel's turns, newest first, as metadata only. None when
    ``match`` names something the channel doesn't record."""
    clause = _where(channel, first_day=first_day, last_day=last_day, environment=environment, match=match)
    if clause is None:
        return None
    where, parameters = clause
    table = TABLES[channel]
    ((total,),) = client.query(_COUNT_SQL.format(table=table, where=where), parameters=parameters).result_rows
    names = [column.split(" AS ")[-1] for column in TURN_COLUMNS[channel]]
    sql = _TURNS_SQL.format(columns=", ".join(TURN_COLUMNS[channel]), table=table, where=where)
    rows = client.query(sql, parameters={**parameters, "limit": limit, "offset": offset}).result_rows
    return {
        "total": int(total),
        "rows": [{name: _value(value) for name, value in zip(names, row)} for row in rows],
    }


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
