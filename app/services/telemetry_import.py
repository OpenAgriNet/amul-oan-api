"""Import voice turns into the telemetry database (telemetry/clickhouse/voice.sql).

Rows never carry the caller's phone number or any of their words: user_id is
dropped and question/answer keep only their length and sha256.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterator, Mapping, Protocol, Sequence

from pydantic import ValidationError

from app.models.telemetry_voice_analytics import CanonicalVoiceTurn
from app.services.telemetry_era_adapters import UnsupportedTelemetryEra
from app.services.telemetry_era_registry import TelemetryEraRegistry
from app.services.telemetry_fetcher import ClickHouseReader, fetch_voice_bundles
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
)
IMPORT_DAY_COLUMNS = ("environment", "day", "traces", "turns", "rejected", "imported_at")
REJECTION_COLUMNS = ("environment", "day", "imported_at", "trace_name", "reason", "count")


class ClickHouseWriter(Protocol):
    def insert(self, table: str, data: Sequence[Sequence[Any]], column_names: Sequence[str], database: str) -> Any: ...


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
) -> ImportReport:
    """Adapt every voice turn from first_day to last_day (UTC, inclusive). Without a writer nothing is written."""
    report = ImportReport(environment, first_day, last_day, written=writer is not None)
    root_names = registry.root_trace_names() | {mapping.root for mapping in mappings.values()}
    for day in _days(first_day, last_day):
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        imported_at = datetime.now(timezone.utc)
        rows, rejected, traces = [], Counter(), 0
        for bundle in fetch_voice_bundles(
            reader, environment=environment, start=start, end=start + timedelta(days=1), root_names=root_names
        ):
            traces += 1
            try:
                turn = adapt_voice_trace(
                    bundle.trace,
                    observations=bundle.observations,
                    scores=bundle.scores,
                    era_registry=registry,
                    outcome_vocabulary=vocabulary,
                    voice_mappings=mappings,
                )
            except (UnsupportedTelemetryEra, ValidationError) as exc:
                rejected[(bundle.trace["name"], rejection_reason(exc))] += 1
                continue
            report.add(turn)
            rows.append(voice_turn_row(turn, environment=environment, imported_at=imported_at))

        report.traces += traces
        report.rejected.update(rejected)
        if writer is not None:
            _write_day(writer, environment, day, imported_at, rows, rejected, traces)
    return report


def voice_turn_row(turn: CanonicalVoiceTurn, *, environment: str, imported_at: datetime) -> dict[str, Any]:
    question, answer = turn.question_sanitized, turn.answer_sanitized
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
        "user_id_hash": turn.user_id_hash,
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
    }


def rejection_reason(exc: Exception) -> str:
    """One stable line per kind of rejection, so a day's rejections group together."""
    if isinstance(exc, ValidationError):
        error = exc.errors()[0]
        return f"invalid trace: {'.'.join(str(part) for part in error['loc'])} {error['type']}"
    return re.sub(r" timestamp=\S+", "", str(exc))


def _write_day(
    writer: ClickHouseWriter,
    environment: str,
    day: date,
    imported_at: datetime,
    rows: list[dict[str, Any]],
    rejected: Counter,
    traces: int,
) -> None:
    if rows:
        writer.insert(
            "voice_turns",
            [[row[column] for column in VOICE_TURN_COLUMNS] for row in rows],
            column_names=VOICE_TURN_COLUMNS,
            database=DATABASE,
        )
    if rejected:
        writer.insert(
            "voice_rejections",
            [[environment, day, imported_at, name, reason, count] for (name, reason), count in rejected.items()],
            column_names=REJECTION_COLUMNS,
            database=DATABASE,
        )
    # Written last, so a day only shows as imported once its turns are in.
    writer.insert(
        "voice_import_days",
        [[environment, day, traces, len(rows), sum(rejected.values()), imported_at]],
        column_names=IMPORT_DAY_COLUMNS,
        database=DATABASE,
    )


def _days(first: date, last: date) -> Iterator[date]:
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)
