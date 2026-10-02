-- Users for scripts/telemetry_import.py and the dashboards. Run as the ClickHouse admin.
-- <reader_hash> / <writer_hash> / <dashboard_hash> are sha256 of each password:
--   echo -n "$PASSWORD" | sha256sum
-- The limits keep a heavy query from slowing Langfuse down. CREATE ... OR REPLACE
-- resets a user, so to add one user later run only its own statements.

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

-- Dashboards (e.g. Metabase): read the telemetry database, nothing else. No
-- writes, no Langfuse tables. readonly = 2 because BI drivers set query settings.
CREATE USER OR REPLACE telemetry_dashboard
IDENTIFIED WITH sha256_hash BY '<dashboard_hash>'
SETTINGS readonly = 2, max_execution_time = 60, max_memory_usage = 4000000000;
GRANT SELECT ON telemetry.* TO telemetry_dashboard;
