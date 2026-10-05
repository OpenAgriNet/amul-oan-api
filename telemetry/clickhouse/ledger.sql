-- What became of every root trace the import read, filled by scripts/telemetry_import.py.
-- Run as the ClickHouse admin after voice.sql and chat.sql. Safe to re-run.
--
-- One row per root trace per day, for both channels: `turn` (it's in voice_turns
-- or chat_turns), `rejected` (with the reason), `activity` (a known non-turn
-- trace, see telemetry/non_turn_traces.yaml) or `unrecognised` (nobody has looked
-- at this name yet). It has no user id or text; only the schema stamp, raw
-- outcome and derived duration. Each channel's import keeps its own rows. A
-- re-imported day adds newer rows, and a trace it no longer has (deleted in
-- Langfuse, or moved to another day) gets a row with is_deleted = 1; read with
-- FINAL, which leaves those out.

CREATE DATABASE IF NOT EXISTS telemetry;

CREATE TABLE IF NOT EXISTS telemetry.trace_ledger
(
    environment LowCardinality(String),
    channel LowCardinality(String),
    day Date,
    source_trace_id String,
    timestamp DateTime64(3, 'UTC'),
    trace_name LowCardinality(String),
    disposition LowCardinality(String),
    reason String,
    schema_version LowCardinality(String),
    imported_at DateTime64(3, 'UTC'),
    is_deleted UInt8
)
ENGINE = ReplacingMergeTree(imported_at, is_deleted)
PARTITION BY toYYYYMM(day)
ORDER BY (environment, channel, day, source_trace_id);

-- New columns go below this line, one per statement, e.g.
--   ALTER TABLE telemetry.trace_ledger ADD COLUMN IF NOT EXISTS example LowCardinality(String);
-- Don't edit or remove the columns above.
ALTER TABLE telemetry.trace_ledger ADD COLUMN IF NOT EXISTS duration_ms Nullable(Float64);
ALTER TABLE telemetry.trace_ledger ADD COLUMN IF NOT EXISTS outcome LowCardinality(Nullable(String));
