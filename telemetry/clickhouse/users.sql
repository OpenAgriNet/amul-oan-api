-- Users for scripts/telemetry_import.py. Run as the ClickHouse admin.
-- <reader_hash> / <writer_hash> are sha256 of each password:
--   echo -n "$PASSWORD" | sha256sum
-- The limits keep a heavy query from slowing Langfuse down.

-- Reads Langfuse's tables, nothing else.
CREATE USER OR REPLACE telemetry_reader
IDENTIFIED WITH sha256_hash BY '<reader_hash>'
SETTINGS readonly = 2, max_execution_time = 120, max_memory_usage = 4000000000;
GRANT SELECT ON default.traces TO telemetry_reader;
GRANT SELECT ON default.observations TO telemetry_reader;
GRANT SELECT ON default.scores TO telemetry_reader;

-- Writes the telemetry database, and can't touch Langfuse's tables.
CREATE USER OR REPLACE telemetry_writer
IDENTIFIED WITH sha256_hash BY '<writer_hash>'
SETTINGS max_execution_time = 120, max_memory_usage = 4000000000;
GRANT SELECT, INSERT ON telemetry.* TO telemetry_writer;
