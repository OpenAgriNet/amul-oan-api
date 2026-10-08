"""Import voice and chat turns into the telemetry database (telemetry/clickhouse/).

Rows never carry a phone number or anyone's words: user ids are only kept as
an HMAC under a secret caller key, and question/answer keep only their length
and sha256.

No trace is dropped without a record: every root trace of a day gets one row in
telemetry.trace_ledger saying what became of it (a turn, rejected with the
reason, a known non-turn activity, or unrecognised).
"""

import hashlib
import hmac
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from pydantic import ValidationError

from app.models.telemetry_analytics import CanonicalChatTurn
from app.models.telemetry_voice_analytics import CanonicalVoiceTurn
from app.services.telemetry_era_adapters import (
    UnsupportedTelemetryEra,
    _chat_has_full_turn_root,
    adapt_chat_trace,
)
from app.services.telemetry_era_registry import TelemetryEraRegistry, load_yaml_file
from app.services.telemetry_fetcher import (
    ClickHouseReader,
    TraceBundle,
    fetch_chat_bundles,
    fetch_trace_identities,
    fetch_voice_bundles,
)
from app.services.telemetry_mappings import ContractMapping
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary, adapt_voice_trace

DATABASE = "telemetry"

# Must match the tables in telemetry/clickhouse/voice.sql.
VOICE_TURN_COLUMNS = (
    "source_trace_id",
    "timestamp",
    "environment",
    "schema_version",
    "source_era",
    "source_schema_version",
    "source_era_extensions",
    "source_trace_name",
    "session_id",
    "process_id",
    "user_id_hash",
    "signed_in",
    "provider",
    "call_type",
    "route",
    "pipeline_profile",
    "source_lang",
    "target_lang",
    "question_chars",
    "question_sha256",
    "answer_chars",
    "answer_sha256",
    "outcome",
    "outcome_class",
    "full_turn_latency_ms",
    "stage_totals_ms",
    "timings_ms",
    "observation_names",
    "score_names",
    "field_availability",
    "imported_at",
    "is_deleted",
    "attributes",
    "service",
    "release",
    "user_id_hash_key",
)
# CanonicalVoiceTurn fields voice_turns leaves out on purpose. Every other field
# needs a column, or tests fail, so a new field can't quietly miss the table.
NOT_STORED = {
    "user_id": "the caller's phone number; user_id_hash is kept",
    "user_id_semantics": "the same for every voice turn",
    "channel": "always voice",
    "question_sanitized": "kept as question_chars and question_sha256",
    "answer_sanitized": "kept as answer_chars and answer_sha256",
}

# Must match the tables in telemetry/clickhouse/chat.sql.
CHAT_TURN_COLUMNS = (
    "source_trace_id",
    "timestamp",
    "environment",
    "schema_version",
    "source_era",
    "source_schema_version",
    "source_era_extensions",
    "source_trace_name",
    "session_id",
    "user_id_hash",
    "user_id_semantics",
    "channel",
    "pipeline",
    "pipeline_profile",
    "source_lang",
    "target_lang",
    "question_chars",
    "question_sha256",
    "answer_chars",
    "answer_sha256",
    "persona",
    "outcome",
    "outcome_class",
    "served_tier",
    "full_turn_latency_ms",
    "tool_names",
    "tool_call_count",
    "observation_names",
    "score_names",
    "field_availability",
    "imported_at",
    "is_deleted",
    "attributes",
    "service",
    "release",
    "stage_totals_ms",
    "user_id_hash_key",
)
# CanonicalChatTurn fields chat_turns leaves out on purpose.
CHAT_NOT_STORED = {
    "question_sanitized": "kept as question_chars and question_sha256",
    "answer_sanitized": "kept as answer_chars and answer_sha256",
    "tool_calls": "only tool names and a count; tool inputs and outputs carry farmer data",
}
IMPORT_DAY_COLUMNS = ("environment", "day", "traces", "turns", "rejected", "imported_at")
REJECTION_COLUMNS = ("environment", "day", "imported_at", "trace_name", "reason", "count")
# Must match telemetry/clickhouse/ledger.sql.
LEDGER_COLUMNS = (
    "environment",
    "channel",
    "day",
    "source_trace_id",
    "timestamp",
    "trace_name",
    "disposition",
    "reason",
    "schema_version",
    "imported_at",
    "is_deleted",
    "duration_ms",
    "outcome",
)
# A removed row: the table's sorting key, so it replaces that row, and is_deleted.
REMOVED_TURN_COLUMNS = ("source_trace_id", "timestamp", "environment", "imported_at", "is_deleted")
REMOVED_LEDGER_COLUMNS = ("environment", "channel", "day", "source_trace_id", "imported_at", "is_deleted")

# Turns a day's import has to remove. On that day: every turn it didn't accept
# (deleted in Langfuse, now rejected, or moved to another day). On other days:
# older rows of the turns it did accept, left there when a timestamp moved.
# A busy day has more trace ids than ClickHouse takes in one HTTP parameter
# (128 KiB), so the day's rows are compared here and ids only go in batches.
_DAY_TURNS_SQL = """
SELECT source_trace_id, timestamp
FROM {database}.{table} FINAL
WHERE environment = {{environment:String}}
  AND toDate(timestamp) = {{day:Date}}
"""

_MOVED_TURNS_SQL = """
SELECT source_trace_id, timestamp
FROM {database}.{table} FINAL
WHERE environment = {{environment:String}}
  AND toDate(timestamp) != {{day:Date}}
  AND source_trace_id IN {{ids:Array(String)}}
"""

# The same for the channel's ledger rows: root traces the day no longer has, and
# older rows of the ones it has under another day.
_DAY_LEDGER_SQL = """
SELECT day, source_trace_id
FROM {database}.trace_ledger FINAL
WHERE environment = {{environment:String}}
  AND channel = {{channel:String}}
  AND day = {{day:Date}}
"""

_MOVED_LEDGER_SQL = """
SELECT day, source_trace_id
FROM {database}.trace_ledger FINAL
WHERE environment = {{environment:String}}
  AND channel = {{channel:String}}
  AND day != {{day:Date}}
  AND source_trace_id IN {{ids:Array(String)}}
"""

# Trace ids per query: 1000 stay well under that limit.
_ID_BATCH = 1000


def default_non_turn_traces_path() -> Path:
    return Path(__file__).resolve().parents[2] / "telemetry" / "non_turn_traces.yaml"


@dataclass(frozen=True)
class NonTurnTraces:
    """Root trace names known not to be turns, from telemetry/non_turn_traces.yaml."""

    names: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "NonTurnTraces":
        payload = load_yaml_file(path or default_non_turn_traces_path()) or {}
        return cls(frozenset(payload.get("names") or ()), tuple(payload.get("prefixes") or ()))

    def __contains__(self, name: str) -> bool:
        return name in self.names or name.startswith(self.prefixes)


@dataclass(frozen=True)
class CallerKey:
    """The secret that turns a caller's hash into the one stored.

    Traces carry a SHA-256 of the caller id under a public prefix, which a list
    of phone numbers reverses. Rows keep an HMAC of it under this key instead,
    with the key's id, so rows written under an older key can be found and
    re-imported after a rotation."""

    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.secret) < 32:
            raise ValueError("The caller key must be at least 32 bytes, e.g. from `openssl rand -hex 32`.")

    @property
    def key_id(self) -> str:
        """Names the key without giving it away."""
        return hashlib.sha256(b"telemetry caller key id:" + self.secret).hexdigest()[:12]

    def pseudonym(self, caller_hash: str | None) -> str | None:
        if caller_hash is None:
            return None
        return hmac.new(self.secret, caller_hash.encode("utf-8"), hashlib.sha256).hexdigest()


class ClickHouseWriter(Protocol):
    def insert(self, table: str, data: Sequence[Sequence[Any]], column_names: Sequence[str], database: str) -> Any: ...

    def query(self, query: str, parameters: Mapping[str, Any] | None = None) -> Any: ...


@dataclass
class ImportReport:
    environment: str
    first_day: date
    last_day: date
    written: bool
    traces: int = 0
    turns: int = 0
    by_era: Counter = field(default_factory=Counter)
    by_outcome_class: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    # Turns where each field was recorded or derived, not unavailable.
    available: Counter = field(default_factory=Counter)
    # Every root trace read, by what became of it (telemetry.trace_ledger).
    ledger: Counter = field(default_factory=Counter)
    # Activities and unrecognised roots by name, to spot a new family of traces.
    not_turns: Counter = field(default_factory=Counter)
    # Turns from an earlier import that this one removed.
    removed: int = 0

    def add(self, turn: CanonicalVoiceTurn) -> None:
        self.turns += 1
        self.by_era[turn.source_era] += 1
        self.by_outcome_class[turn.outcome_class or "none"] += 1
        self.available.update(name for name, status in turn.field_availability.items() if status != "unavailable")

    def lines(self) -> list[str]:
        mode = "written" if self.written else "dry run, nothing written"
        lines = [
            f"{self.environment}, {self.first_day} to {self.last_day} ({mode})",
            f"traces read  {self.traces}",
            f"turns        {self.turns}",
            f"rejected     {sum(self.rejected.values())}",
        ]
        lines += [f"  {count}  {name}: {reason}" for (name, reason), count in self.rejected.most_common()]
        if self.written:
            lines.append(f"removed      {self.removed}")
        lines += ["every root trace, by what became of it"]
        lines += [f"  {count}  {disposition}" for disposition, count in self.ledger.most_common()]
        if self.not_turns:
            lines.append("not turns, by name")
            lines += [f"  {count}  {name} ({kind})" for (kind, name), count in self.not_turns.most_common()]
        lines += ["by era"] + [f"  {count}  {era}" for era, count in self.by_era.most_common()]
        lines += ["outcome class"] + [f"  {count}  {name}" for name, count in self.by_outcome_class.most_common()]
        if self.turns:
            lines.append("fields present")
            lines += [
                f"  {100 * count / self.turns:5.1f}%  {name}" for name, count in sorted(self.available.items())
            ]
        return lines


def import_voice_days(
    reader: ClickHouseReader,
    writer: ClickHouseWriter | None,
    *,
    environment: str,
    first_day: date,
    last_day: date,
    registry: TelemetryEraRegistry,
    vocabulary: VoiceOutcomeVocabulary,
    mappings: Mapping[str, ContractMapping],
    non_turn: NonTurnTraces | None = None,
    caller_key: CallerKey | None = None,
) -> ImportReport:
    """Adapt every voice turn from first_day to last_day (UTC, inclusive). Without a writer nothing is written;
    with one, caller_key is required."""
    return _import_days(
        reader,
        writer,
        environment=environment,
        first_day=first_day,
        last_day=last_day,
        fetch=fetch_voice_bundles,
        root_names=registry.root_trace_names() | {mapping.root for mapping in mappings.values()},
        adapt=lambda bundle: adapt_voice_trace(
            bundle.trace,
            observations=bundle.observations,
            scores=bundle.scores,
            era_registry=registry,
            outcome_vocabulary=vocabulary,
            voice_mappings=mappings,
        ),
        row=voice_turn_row,
        table="voice",
        columns=VOICE_TURN_COLUMNS,
        non_turn=non_turn or NonTurnTraces.from_yaml(),
        caller_key=caller_key,
    )


def import_chat_days(
    reader: ClickHouseReader,
    writer: ClickHouseWriter | None,
    *,
    environment: str,
    first_day: date,
    last_day: date,
    registry: TelemetryEraRegistry,
    mappings: Mapping[str, ContractMapping],
    non_turn: NonTurnTraces | None = None,
    caller_key: CallerKey | None = None,
) -> ImportReport:
    """Adapt every chat turn from first_day to last_day (UTC, inclusive). Without a writer nothing is written;
    with one, caller_key is required."""
    return _import_days(
        reader,
        writer,
        environment=environment,
        first_day=first_day,
        last_day=last_day,
        fetch=fetch_chat_bundles,
        root_names=registry.root_trace_names() | {mapping.root for mapping in mappings.values()},
        adapt=lambda bundle: adapt_chat_trace(
            bundle.trace,
            observations=bundle.observations,
            scores=bundle.scores,
            related_traces=bundle.related_traces,
            era_registry=registry,
            chat_mappings=mappings,
        ),
        row=chat_turn_row,
        table="chat",
        columns=CHAT_TURN_COLUMNS,
        non_turn=non_turn or NonTurnTraces.from_yaml(),
        caller_key=caller_key,
    )


def _import_days(
    reader: ClickHouseReader,
    writer: ClickHouseWriter | None,
    *,
    environment: str,
    first_day: date,
    last_day: date,
    fetch: Callable[..., Iterator[TraceBundle]],
    root_names: Iterable[str],
    adapt: Callable[[TraceBundle], Any],
    row: Callable[..., dict[str, Any]],
    table: str,
    columns: Sequence[str],
    non_turn: NonTurnTraces,
    caller_key: CallerKey | None,
) -> ImportReport:
    if writer is not None and caller_key is None:
        raise ValueError("Writing turns needs the caller key, so no caller hash is stored without it.")
    report = ImportReport(environment, first_day, last_day, written=writer is not None)
    root_names = set(root_names)
    for day in _days(first_day, last_day):
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        imported_at = datetime.now(timezone.utc)
        rows, rejected, traces, ledger = [], Counter(), 0, []
        accepted_turns = []

        def account(trace_id, name, timestamp, schema_version, disposition, reason="", duration_ms=None, outcome=None):
            ledger.append(
                [
                    environment,
                    table,
                    day,
                    trace_id,
                    timestamp,
                    name,
                    disposition,
                    reason,
                    schema_version,
                    imported_at,
                    0,
                    duration_ms,
                    outcome,
                ]
            )
            report.ledger[disposition] += 1
            if disposition in ("activity", "unrecognised"):
                report.not_turns[(disposition, name)] += 1

        for bundle in fetch(reader, environment=environment, start=start, end=end, root_names=root_names):
            traces += 1
            trace = bundle.trace
            stamp = str(trace["metadata"].get("amul.schema_version") or "")
            try:
                turn = adapt(bundle)
            except (UnsupportedTelemetryEra, ValidationError) as exc:
                reason = rejection_reason(exc)
                rejected[(trace["name"], reason)] += 1
                account(trace["id"], trace["name"], trace["timestamp"], stamp, "rejected", reason)
                continue
            accepted_turns.append(turn)
            account(
                trace["id"],
                trace["name"],
                trace["timestamp"],
                stamp,
                "turn",
                duration_ms=turn.full_turn_latency_ms,
                outcome=turn.outcome,
            )

        # Every other root trace of the day: known activities, and anything nobody
        # has looked at yet, so a new kind of trace shows up instead of vanishing.
        identities = fetch_trace_identities(reader, environment=environment, start=start, end=end)
        seen = {entry[3] for entry in ledger}
        for identity in identities:
            if identity.trace_id in seen:
                continue
            fields = (identity.trace_id, identity.name, identity.timestamp, identity.schema_version)
            if identity.name in root_names:
                # Written between the two reads; the next import of the day picks it up.
                # Counted as read too, so a day's traces still equal turns plus rejected.
                reason = "arrived during the import; re-import the day"
                traces += 1
                rejected[(identity.name, reason)] += 1
                account(*fields, "rejected", reason, identity.duration_ms, identity.outcome)
            elif identity.name in non_turn:
                account(*fields, "activity", duration_ms=identity.duration_ms, outcome=identity.outcome)
            else:
                account(
                    *fields,
                    "unrecognised",
                    "no importer reads this trace name",
                    identity.duration_ms,
                    identity.outcome,
                )

        # The identity pass includes every root, including roots already
        # adapted above. It supplies source duration for chat and historical
        # turns that have no canonical total, and source outcome where a score
        # or adapter did not provide one.
        identities_by_id = {
            identity.trace_id: identity
            for identity in identities
        }
        for entry in ledger:
            identity = identities_by_id.get(entry[3])
            if identity is None:
                continue
            if entry[11] is None:
                entry[11] = identity.duration_ms
            if entry[12] is None:
                entry[12] = identity.outcome

        # The identity pass measures the turn-root span from its timestamp to
        # the latest completed child. Earlier chat roots represent agent work,
        # not a whole farmer turn, so their duration stays ledger-only.
        for turn in accepted_turns:
            if table == "chat" and _chat_has_full_turn_root(turn) and turn.full_turn_latency_ms is None:
                identity = identities_by_id.get(turn.source_trace_id)
                if identity is not None and identity.duration_ms is not None:
                    turn = turn.model_copy(update={
                        "full_turn_latency_ms": identity.duration_ms,
                        "field_availability": {
                            **turn.field_availability,
                            "full_turn_latency_ms": "derived",
                        },
                    })
            report.add(turn)
            rows.append(row(turn, environment=environment, imported_at=imported_at, caller_key=caller_key))

        report.traces += traces
        report.rejected.update(rejected)
        if writer is not None:
            report.removed += _write_day(
                writer, table, columns, environment, day, imported_at, rows, rejected, traces, ledger
            )
    return report


def voice_turn_row(
    turn: CanonicalVoiceTurn, *, environment: str, imported_at: datetime, caller_key: CallerKey | None = None
) -> dict[str, Any]:
    question, answer = turn.question_sanitized, turn.answer_sanitized
    user_id_hash = caller_key.pseudonym(turn.user_id_hash) if caller_key else None
    return {
        "source_trace_id": turn.source_trace_id,
        "timestamp": turn.timestamp,
        "environment": environment,
        "schema_version": turn.schema_version,
        "source_era": turn.source_era,
        "source_schema_version": turn.source_schema_version,
        "source_era_extensions": list(turn.source_era_extensions),
        "source_trace_name": turn.source_trace_name,
        "session_id": turn.session_id,
        "process_id": turn.process_id,
        "user_id_hash": user_id_hash,
        "signed_in": turn.signed_in,
        "provider": turn.provider,
        "call_type": turn.call_type,
        "route": turn.route,
        "pipeline_profile": turn.pipeline_profile,
        "source_lang": turn.source_lang,
        "target_lang": turn.target_lang,
        "question_chars": question.chars if question else None,
        "question_sha256": question.sha256 if question else None,
        "answer_chars": answer.chars if answer else None,
        "answer_sha256": answer.sha256 if answer else None,
        "outcome": turn.outcome,
        "outcome_class": turn.outcome_class,
        "full_turn_latency_ms": turn.full_turn_latency_ms,
        "stage_totals_ms": turn.stage_totals_ms or {},
        "timings_ms": turn.timings_ms or {},
        "observation_names": list(turn.observation_names),
        "score_names": list(turn.score_names),
        "field_availability": dict(turn.field_availability),
        "imported_at": imported_at,
        "is_deleted": 0,
        "attributes": dict(turn.attributes),
        "service": turn.service,
        "release": turn.release,
        "user_id_hash_key": caller_key.key_id if user_id_hash else None,
    }


def chat_turn_row(
    turn: CanonicalChatTurn, *, environment: str, imported_at: datetime, caller_key: CallerKey | None = None
) -> dict[str, Any]:
    question, answer = turn.question_sanitized, turn.answer_sanitized
    tool_calls = turn.tool_calls
    user_id_hash = caller_key.pseudonym(turn.user_id_hash) if caller_key else None
    return {
        "source_trace_id": turn.source_trace_id,
        "timestamp": turn.timestamp,
        "environment": environment,
        "schema_version": turn.schema_version,
        "source_era": turn.source_era,
        "source_schema_version": turn.source_schema_version,
        "source_era_extensions": list(turn.source_era_extensions),
        "source_trace_name": turn.source_trace_name,
        "session_id": turn.session_id,
        "user_id_hash": user_id_hash,
        "user_id_semantics": turn.user_id_semantics,
        "channel": turn.channel,
        "pipeline": turn.pipeline,
        "pipeline_profile": turn.pipeline_profile,
        "source_lang": turn.source_lang,
        "target_lang": turn.target_lang,
        "question_chars": question.chars if question else None,
        "question_sha256": question.sha256 if question else None,
        "answer_chars": answer.chars if answer else None,
        "answer_sha256": answer.sha256 if answer else None,
        "persona": turn.persona,
        "outcome": turn.outcome,
        "outcome_class": turn.outcome_class,
        "served_tier": turn.served_tier,
        "full_turn_latency_ms": turn.full_turn_latency_ms,
        "stage_totals_ms": turn.stage_totals_ms or {},
        "tool_names": [name for call in tool_calls or [] if (name := _tool_name(call))],
        "tool_call_count": len(tool_calls) if tool_calls is not None else None,
        "observation_names": list(turn.observation_names),
        "score_names": list(turn.score_names),
        "field_availability": dict(turn.field_availability),
        "imported_at": imported_at,
        "is_deleted": 0,
        "attributes": dict(turn.attributes),
        "service": turn.service,
        "release": turn.release,
        "user_id_hash_key": caller_key.key_id if user_id_hash else None,
    }


def _tool_name(call: Any) -> str | None:
    """A tool call's name: CanonicalToolCall.tool_name, or the older dict shape."""
    if isinstance(call, Mapping):
        name = call.get("tool_name") or call.get("name")
    else:
        name = getattr(call, "tool_name", None)
    return name if isinstance(name, str) and name else None


def rejection_reason(exc: Exception) -> str:
    """One stable line per kind of rejection, so a day's rejections group together."""
    if isinstance(exc, ValidationError):
        error = exc.errors()[0]
        return f"invalid trace: {'.'.join(str(part) for part in error['loc'])} {error['type']}"
    return re.sub(r" timestamp=\S+", "", str(exc))


def _write_day(
    writer: ClickHouseWriter,
    table: str,
    columns: Sequence[str],
    environment: str,
    day: date,
    imported_at: datetime,
    rows: list[dict[str, Any]],
    rejected: Counter,
    traces: int,
    ledger: list[list[Any]],
) -> int:
    """Write one day's import and return how many earlier turns it removed.

    A re-import ends with the same rows as a first import: turns and ledger rows
    it no longer finds are marked is_deleted, which FINAL leaves out."""
    removed = [
        (found["source_trace_id"], found["timestamp"])
        for found in _rows_to_remove(
            writer,
            _DAY_TURNS_SQL.format(database=DATABASE, table=f"{table}_turns"),
            _MOVED_TURNS_SQL.format(database=DATABASE, table=f"{table}_turns"),
            {row["source_trace_id"] for row in rows},
            environment=environment,
            day=day,
        )
    ]
    removed_ledger = [
        (found["day"], found["source_trace_id"])
        for found in _rows_to_remove(
            writer,
            _DAY_LEDGER_SQL.format(database=DATABASE),
            _MOVED_LEDGER_SQL.format(database=DATABASE),
            {entry[3] for entry in ledger},
            environment=environment,
            channel=table,
            day=day,
        )
    ]
    if rows:
        writer.insert(
            f"{table}_turns",
            [[row[column] for column in columns] for row in rows],
            column_names=columns,
            database=DATABASE,
        )
    if removed:
        writer.insert(
            f"{table}_turns",
            [[trace_id, timestamp, environment, imported_at, 1] for trace_id, timestamp in removed],
            column_names=REMOVED_TURN_COLUMNS,
            database=DATABASE,
        )
    if rejected:
        writer.insert(
            f"{table}_rejections",
            [[environment, day, imported_at, name, reason, count] for (name, reason), count in rejected.items()],
            column_names=REJECTION_COLUMNS,
            database=DATABASE,
        )
    if ledger:
        writer.insert("trace_ledger", ledger, column_names=LEDGER_COLUMNS, database=DATABASE)
    if removed_ledger:
        writer.insert(
            "trace_ledger",
            [[environment, table, ledger_day, trace_id, imported_at, 1] for ledger_day, trace_id in removed_ledger],
            column_names=REMOVED_LEDGER_COLUMNS,
            database=DATABASE,
        )
    # Written last, so a day only shows as imported once its turns are in.
    writer.insert(
        f"{table}_import_days",
        [[environment, day, traces, len(rows), sum(rejected.values()), imported_at]],
        column_names=IMPORT_DAY_COLUMNS,
        database=DATABASE,
    )
    return len(removed)


def _rows_to_remove(
    writer: ClickHouseWriter, day_sql: str, moved_sql: str, kept: set[str], **parameters: Any
) -> list[dict[str, Any]]:
    """The day's rows whose trace it no longer keeps, and rows of the kept traces under another day."""
    found = [row for row in _query(writer, day_sql, **parameters) if row["source_trace_id"] not in kept]
    ids = sorted(kept)
    for index in range(0, len(ids), _ID_BATCH):
        found += _query(writer, moved_sql, ids=ids[index : index + _ID_BATCH], **parameters)
    return found


def _query(writer: ClickHouseWriter, sql: str, **parameters: Any) -> list[dict[str, Any]]:
    return list(writer.query(sql, parameters=parameters).named_results())


def _days(first: date, last: date) -> Iterator[date]:
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)
