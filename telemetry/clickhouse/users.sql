-- Users for scripts/telemetry_import.py and the dashboards. Run as the ClickHouse admin.
-- <reader_hash> / <writer_hash> / <dashboard_hash> are sha256 of each password:
--   echo -n "$PASSWORD" | sha256sum
-- The limits keep a heavy query from slowing Langfuse down. Prod's ClickHouse has
-- 2 cores and refuses new work once all its queries hold about 6.5 GiB, so each
-- user gets 2 threads and 1 GiB a query, and MAX stops a client raising the memory
-- limit. CREATE ... OR REPLACE resets a user, so to add one user later run only its
-- own statements.

-- Reads Langfuse's tables, nothing else. A day's sort that outgrows 512 MiB goes
-- to disk instead of failing.
CREATE USER OR REPLACE telemetry_reader
IDENTIFIED WITH sha256_hash BY '<reader_hash>'
SETTINGS readonly = 2, max_execution_time = 180, max_threads = 2,
    max_memory_usage = 1073741824 MAX 1073741824, max_bytes_before_external_sort = 536870912;
GRANT SELECT ON default.traces TO telemetry_reader;
GRANT SELECT ON default.observations TO telemetry_reader;
GRANT SELECT ON default.scores TO telemetry_reader;

-- Writes the telemetry database, and can't touch Langfuse's tables.
CREATE USER OR REPLACE telemetry_writer
IDENTIFIED WITH sha256_hash BY '<writer_hash>'
SETTINGS max_execution_time = 180, max_threads = 2, max_memory_usage = 1073741824 MAX 1073741824;
GRANT SELECT, INSERT ON telemetry.* TO telemetry_writer;

-- Dashboards (e.g. Metabase) and the query API: read the telemetry database,
-- nothing else. No writes, no Langfuse tables. readonly = 2 because BI drivers
-- set query settings.
CREATE USER OR REPLACE telemetry_dashboard
IDENTIFIED WITH sha256_hash BY '<dashboard_hash>'
SETTINGS readonly = 2, max_execution_time = 60, max_threads = 2, max_memory_usage = 1073741824 MAX 1073741824;
GRANT SELECT ON telemetry.* TO telemetry_dashboard;
