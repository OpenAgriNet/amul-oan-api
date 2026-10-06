# Trace-duplication investigation

## Current code-level conclusion

There are two expected sources of **physical** duplicate rows. Neither is a
duplicate farmer turn by itself.

1. Langfuse source tables are event/update tables. A trace, observation, or
   score can have multiple rows with the same ID as it is created, updated, or
   deleted. The fetcher reads the newest `event_ts` row with `LIMIT 1 BY id`,
   then ignores a newest row marked `is_deleted`.
2. The canonical importer is intentionally rerunnable by UTC day. It writes a
   new row for a re-import into `ReplacingMergeTree(imported_at, is_deleted)`.
   ClickHouse retains earlier physical copies until a background merge; `FINAL`
   selects the newest copy for the same sorting key and omits turns whose latest
   re-import marked them `is_deleted = 1`.

This means dashboards and verification queries must read
`telemetry.chat_turns FINAL` / `telemetry.voice_turns FINAL`, not the raw
tables. `source_trace_id` is the canonical identity; do not deduplicate on a
session or process ID.

The actual database still needs the checks below. They distinguish the expected
storage behavior from a fetcher/importer defect.

## 1. Are source duplicates Langfuse update versions?

Run separately for chat and voice with an appropriate environment/date range.
This reads IDs and timestamps only; it does not select user or text fields.

```sql
SELECT
    id AS trace_id,
    count() AS physical_rows,
    countDistinct(event_ts) AS event_versions,
    min(event_ts) AS first_event_ts,
    max(event_ts) AS latest_event_ts,
    groupUniqArray(is_deleted) AS deleted_states
FROM traces
WHERE environment = 'chat-production'
  AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
  AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')
GROUP BY id
HAVING physical_rows > 1
ORDER BY physical_rows DESC, latest_event_ts DESC
LIMIT 100;
```

If rows have the same `id` with increasing `event_ts`, this is normal Langfuse
update history. If there are several rows with the same ID and the *same*
`event_ts`, capture the count and a redacted sample for a fetcher investigation.

## 2. Does the fetcher collapse source versions correctly?

Use the same root-name filter as the importer. The historical chat set is
`chat.default`, `chat.translation`, and `Amul AI Agent`; omitting either of the
first two undercounts older days, and omitting `Amul AI Agent` drops c3/c4.
Replace this list with the configured voice root names when checking voice. C2
deliberately reuses `chat.translation` for stage traces, so the result is the
maximum candidate-root count before adapters reject those non-turn rows.

```sql
SELECT
    count() AS physical_rows,
    uniqExact(id) AS distinct_trace_ids,
    physical_rows - distinct_trace_ids AS extra_source_versions
FROM traces
WHERE environment = '<environment>'
  AND name IN ('chat.default', 'chat.translation', 'Amul AI Agent')
  AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
  AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC');

SELECT count() AS latest_non_deleted_trace_ids
FROM
(
    SELECT id, is_deleted
    FROM traces
    WHERE environment = '<environment>'
      AND name IN ('chat.default', 'chat.translation', 'Amul AI Agent')
      AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
      AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')
    ORDER BY event_ts DESC
    LIMIT 1 BY id
)
WHERE is_deleted = 0;
```

The second result is the maximum number of live source traces before the
adapter rejects non-turn roots. It should match a dry-run fetch count for the
same root-name filter and day.

## 3. Are canonical duplicates only re-import copies?

Run this against the environment where canonical data exists, including dev
(for example replace `<environment>` with `chat-development`). The canonical
tables are separate from Langfuse, so this check works even when production
access is unavailable.

```sql
SELECT
    (SELECT count()
     FROM telemetry.chat_turns
     WHERE environment = '<environment>'
       AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
       AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')) AS physical_rows,
    (SELECT count()
     FROM telemetry.chat_turns FINAL
     WHERE environment = '<environment>'
       AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
       AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')) AS logical_rows,
    physical_rows - logical_rows AS replaceable_copies;

SELECT
    source_trace_id,
    count() AS physical_rows,
    min(imported_at) AS first_imported_at,
    max(imported_at) AS latest_imported_at
FROM telemetry.chat_turns
WHERE environment = '<environment>'
  AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
  AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')
GROUP BY source_trace_id
HAVING physical_rows > 1
ORDER BY physical_rows DESC
LIMIT 100;
```

`physical_rows > logical_rows` with differing `imported_at` values is the
expected result of re-importing a day. It is not a dashboard duplicate when
queries use `FINAL`.

## 4. Was a day re-imported?

This also runs on dev. Replace `<environment>` with the environment being
checked; use the same value as query 3.

```sql
SELECT
    environment,
    day,
    count() AS import_attempts,
    min(imported_at) AS first_imported_at,
    max(imported_at) AS latest_imported_at,
    argMax(turns, imported_at) AS latest_turns,
    argMax(rejected, imported_at) AS latest_rejected
FROM telemetry.chat_import_days
WHERE environment = '<environment>'
GROUP BY environment, day
HAVING import_attempts > 1
ORDER BY day DESC;
```

Repeated import attempts are intentional. If the latest `turns` count changes
without a source `event_ts` change, investigate importer code/configuration;
otherwise the source trace was updated after an earlier import.

## Outcome and action

The fetcher/importer owner should run the four checks for one day in each
environment and record only aggregate counts in the importer report. If the
results match the expectations above, retain
`ReplacingMergeTree(imported_at, is_deleted)` and require `FINAL` in dashboard
queries. If not, investigate the first mismatching layer:
Langfuse source versions, fetcher pagination/window overlap, or target-table
sorting key.
