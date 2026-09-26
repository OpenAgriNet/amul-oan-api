-- Canonical chat turns, filled by scripts/telemetry_import.py --channel chat.
-- Run as the ClickHouse admin after voice.sql. Safe to re-run: it only adds
-- what's missing, so run it again after pulling a new column.

CREATE DATABASE IF NOT EXISTS telemetry;

-- One row per chat turn. A re-import of the same trace replaces its row (read
-- with FINAL for exact counts). User ids are hashed, question and answer keep
-- only length and sha256, and tools only their names.
CREATE TABLE IF NOT EXISTS telemetry.chat_turns
(
    source_trace_id String,
    timestamp DateTime64(3, 'UTC'),
    environment LowCardinality(String),
    schema_version LowCardinality(String),
    source_era LowCardinality(String),
    source_schema_version LowCardinality(String),
    source_era_extensions Array(LowCardinality(String)),
    source_trace_name LowCardinality(String),
    session_id Nullable(String),
    user_id_hash Nullable(String),
    user_id_semantics LowCardinality(Nullable(String)),
    channel LowCardinality(Nullable(String)),
    pipeline LowCardinality(Nullable(String)),
    pipeline_profile LowCardinality(Nullable(String)),
    source_lang LowCardinality(Nullable(String)),
    target_lang LowCardinality(Nullable(String)),
    question_chars Nullable(UInt32),
    question_sha256 Nullable(String),
    answer_chars Nullable(UInt32),
    answer_sha256 Nullable(String),
    persona LowCardinality(Nullable(String)),
    outcome LowCardinality(Nullable(String)),
    outcome_class LowCardinality(Nullable(String)),
    served_tier LowCardinality(Nullable(String)),
    full_turn_latency_ms Nullable(Float64),
    tool_names Array(LowCardinality(String)),
    tool_call_count Nullable(UInt16),
    observation_names Array(String),
    score_names Array(String),
    field_availability Map(String, LowCardinality(String)),
    imported_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(imported_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (environment, toDate(timestamp), source_trace_id);

-- One row per imported day; the newest import of a day wins.
CREATE TABLE IF NOT EXISTS telemetry.chat_import_days
(
    environment LowCardinality(String),
    day Date,
    traces UInt32,
    turns UInt32,
    rejected UInt32,
    imported_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(imported_at)
ORDER BY (environment, day);

-- Traces no adapter took, per day and reason. Every import keeps its own rows;
-- match imported_at with chat_import_days to read the latest one.
CREATE TABLE IF NOT EXISTS telemetry.chat_rejections
(
    environment LowCardinality(String),
    day Date,
    imported_at DateTime64(3, 'UTC'),
    trace_name LowCardinality(String),
    reason String,
    count UInt32
)
ENGINE = MergeTree
ORDER BY (environment, day, imported_at, trace_name, reason);

-- New columns go below this line, one per statement, e.g.
--   ALTER TABLE telemetry.chat_turns ADD COLUMN IF NOT EXISTS example LowCardinality(Nullable(String));
-- Don't edit or remove the columns above. Dashboards read them, and a table
-- that already exists won't pick up a change there.
