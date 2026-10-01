# Telemetry Pipeline

Voice and chat turns go from Langfuse's ClickHouse into a separate `telemetry`
database on the same server:

```
Langfuse tables (traces, observations, scores)
  -> app/services/telemetry_fetcher.py   one UTC day at a time
  -> voice or chat adapters              CanonicalVoiceTurn / CanonicalChatTurn
  -> app/services/telemetry_import.py    telemetry.voice_turns / telemetry.chat_turns
                                         telemetry.trace_ledger (every root trace)
```

`scripts/telemetry_import.py` runs the whole thing, `--channel chat` for chat.

## What is stored

- `telemetry.voice_turns`: one row per turn. The caller's `user_id` (a phone
  number) is never stored, only `user_id_hash`. Question and answer keep only
  their length and sha256, never the text. Count with `FINAL`, since a
  re-imported day replaces its rows in the background.
- `telemetry.voice_import_days`: per day, how many traces were read, turned into
  turns, or rejected.
- `telemetry.voice_rejections`: per day, why traces were rejected.
- `telemetry.chat_turns`, `chat_import_days`, `chat_rejections`: the same for
  chat. The user id is only kept hashed, and tools keep only their names and a
  count, since their inputs and outputs carry farmer data.
- `telemetry.trace_ledger`: one row for **every** root trace of a day, both
  channels, saying what became of it. See "No trace goes missing".
- `attributes` on `voice_turns` and `chat_turns`: extra values a mapping names
  under `attributes`, as text. See "A new field".

The tables are in `telemetry/clickhouse/voice.sql`, `chat.sql` and `ledger.sql`.

Chat needs more than voice to build a turn, so the importer reads the root's
input and output, and the full rows of the few observations the chat adapters
use (the agent run, stream_translation and tools). They are hashed or measured
in memory and never stored. A c2 turn's question sits in a pretranslation trace
of the same session, so each day is read from 2 minutes before midnight.

## No trace goes missing

Besides the turns, the importer reads the id, name, time and schema stamp of every
live root trace of the day, whatever its name, and writes one ledger row each:

| `disposition` | Meaning |
| --- | --- |
| `turn` | Adapted; the turn is in `voice_turns` / `chat_turns`. |
| `rejected` | Has a turn root name but no adapter took it. `reason` says why. |
| `activity` | A known non-turn trace from `telemetry/non_turn_traces.yaml` (suggestions, background refresh, frontend events). Never attached to a turn; why, per family, is in `telemetry/non_turn_trace_inventory.md`. |
| `unrecognised` | A name nobody has looked at yet. It shows up in the import report under "not turns, by name"; add it to `non_turn_traces.yaml` or build an adapter. |

So a trace is never dropped without a record, and a re-import can find exactly
which traces to process again once an adapter exists. To check a day, the two
numbers below must match:

```sql
-- Live root traces in Langfuse (as telemetry_reader)
SELECT count() FROM (
    SELECT id, is_deleted FROM default.traces
    WHERE environment = 'voice-development'
      AND timestamp >= toDateTime64('2026-09-20 00:00:00', 3, 'UTC')
      AND timestamp <  toDateTime64('2026-09-21 00:00:00', 3, 'UTC')
    ORDER BY event_ts DESC LIMIT 1 BY id
) WHERE is_deleted = 0;

-- Ledger rows for the same day (as telemetry_writer or telemetry_dashboard)
SELECT count() FROM telemetry.trace_ledger FINAL
WHERE environment = 'voice-development' AND day = '2026-09-20';
```

A trace written to Langfuse after its day was imported is picked up the next
time that day is imported.

## Several rows per trace in Langfuse

Langfuse's `traces` and `observations` tables are
`ReplacingMergeTree(event_ts, is_deleted)`, and Langfuse never updates a row.
Each time its worker handles new events for a trace, it reads the newest row,
merges the events into it and inserts a new row. ClickHouse drops the older rows
when it merges parts in the background, at a time of its choosing. Until then a
trace has several rows, and Langfuse reads its own tables with
`ORDER BY event_ts DESC LIMIT 1 BY id`, the same as the importer.

How many rows a trace gets depends on our code:

- The SDK sends spans in batches, every 5 seconds by default
  (`LANGFUSE_FLUSH_INTERVAL`, not set in either repo). Each batch that carries an
  update to a trace is one more worker job for it, so one more row.
- Chat (`propagate_attributes` in `app/services/chat.py`) and voice
  (`VoiceTrace`) put the trace's user, session, tags and metadata on every span
  of a turn. Langfuse takes any span carrying those as a trace update, so even a
  batch with only child spans writes a new trace row. Without them it would write
  one for the root span and for the first span it sees of a trace, and none for
  the rest.

Each span is sent once, when it ends, so observations and scores rarely have more
than one row. Checked against Langfuse's source (main, 2026-10-01):
`packages/shared/src/server/otel/OtelIngestionProcessor.ts` and
`worker/src/services/IngestionService/index.ts`.

To see it on our data, as `telemetry_reader`, for a recent day (older days are
mostly merged already):

```sql
SELECT name, count() AS rows, uniqExact(id) AS traces, round(rows / traces, 1) AS rows_per_trace
FROM default.traces
WHERE environment = 'voice-production'
  AND timestamp >= toDateTime64('2026-09-30 00:00:00', 3, 'UTC')
  AND timestamp <  toDateTime64('2026-10-01 00:00:00', 3, 'UTC')
GROUP BY name
ORDER BY rows DESC
```

What can be done about it:

- The import needs nothing. It reads the newest row of each trace, and only for
  the day it imports.
- A longer `LANGFUSE_FLUSH_INTERVAL` (say 15 seconds) on the chat and voice
  deployments means fewer batches per turn, so fewer rows and fewer
  read-and-merge jobs for the Langfuse worker. Traces show up in Langfuse a few
  seconds later, and a pod killed without a normal shutdown loses more unsent
  spans. It's a setting, not a code change.
- Setting the trace attributes on the root span only would bring most turns down
  to one or two rows. But Langfuse's per-user and per-session numbers on
  observations rely on them, and it changes what every turn sends, so a new
  schema version in both repos. Not worth it.
- Forcing merges with `OPTIMIZE TABLE ... FINAL` on Langfuse's tables: no. They
  are Langfuse's, and it is a heavy operation.

## Setup, once per ClickHouse

1. Run `telemetry/clickhouse/voice.sql`, `chat.sql` and `ledger.sql` as the
   ClickHouse admin. They're safe to re-run, and need re-running whenever a
   change adds a column or a table.
2. Create the users in `telemetry/clickhouse/users.sql`. `telemetry_reader` can
   only read Langfuse's three tables, `telemetry_writer` can only write the
   `telemetry` database, and `telemetry_dashboard` can only read it. Each
   statement replaces its user, so to add one later run only its own lines.
3. Where the import runs, put each password in `~/.telemetry_reader_password`
   and `~/.telemetry_writer_password` (readable only by you), or set
   `TELEMETRY_READER_PASSWORD` / `TELEMETRY_WRITER_PASSWORD`.
   `TELEMETRY_CLICKHOUSE_HOST` and `TELEMETRY_CLICKHOUSE_PORT` default to
   `localhost:8123`.

## Running

```bash
# Read and report only
python scripts/telemetry_import.py --env voice-development --from 2026-09-20 --to 2026-09-25 --dry-run

# Write
python scripts/telemetry_import.py --env voice-development --from 2026-09-20 --to 2026-09-25

# Yesterday (UTC), for a daily cron
python scripts/telemetry_import.py --env voice-production

# Chat
python scripts/telemetry_import.py --channel chat --env chat-production
```

Days are UTC. Re-running a day is safe: its rows are replaced, not doubled.

## Querying, for dashboards

Log in as `telemetry_dashboard` and read only the `telemetry` tables, never
Langfuse's own.

- Always `FINAL`, or a re-imported day can count twice until ClickHouse merges it.
- Always filter `environment` (`voice-production`, `chat-production`), so dev
  traffic stays out.
- Work per day with a date range: `toDate(timestamp) AS day`, then
  `WHERE day BETWEEN {from} AND {to} GROUP BY day`. Roll turn counts up to weeks
  or months in the chart, not in the table. Unique users and sessions don't add
  up across days, so group those by `toMonday(timestamp)` or
  `toStartOfMonth(timestamp)` instead.
- A rate only counts turns that had the field:
  `WHERE field_availability['outcome'] = 'recorded'`. Voice had no outcome before
  v3, and that is not the same as success.
- Unique callers are `uniq(user_id_hash)`. Anonymous callers have no hash, so
  they're left out; count them with `countIf(user_id_hash IS NULL)`. Voice and
  chat hashes use different salts and can't be joined.
- Mapped extras are text: `attributes['farmer_type']`, and
  `toFloat64OrNull(attributes['retries'])` for a number.
- `source_era` tells eras apart, e.g. to compare before and after a change.

```sql
SELECT toDate(timestamp) AS day,
       count() AS turns,
       uniq(user_id_hash) AS callers,
       countIf(outcome_class = 'delivered') / countIf(field_availability['outcome'] = 'recorded') AS delivered_rate
FROM telemetry.voice_turns FINAL
WHERE environment = 'voice-production'
  AND day BETWEEN '2026-09-01' AND '2026-09-30'
GROUP BY day
ORDER BY day
```

### From the API

The dashboard service can ask this app instead of ClickHouse. Read-only
endpoints, for a server rather than a browser, with the key in `X-API-Key`:

- `GET /api/telemetry/stats?from=2026-09-01&to=2026-09-30`: questions, sessions,
  users and new users per channel, and both added up.
- `GET /api/telemetry/graph?from=...&to=...&granularity=day`: questions,
  sessions, users and failed turns per `hour`, `day`, `week` (from Monday) or
  `month`, for each channel. Buckets without turns are left out, and the first
  and last can be partial.
- `GET /api/telemetry/sessions?from=...&to=...`: average questions per session
  and seconds from its first turn to its last, over sessions with a question.
- `GET /api/telemetry/outcomes?from=...&to=...`: turns, sessions and users per
  `outcome_class`, and `not_recorded` for turns with no outcome.

Days are UTC and both ends count. Without `from` the range starts at the first
turn, without `to` it ends today. They read as `telemetry_dashboard` and follow
the rules above.

- A question is a turn, except voice's `non_question` ones (a stale re-dispatch,
  non-speech, a greeting...). Voice before v3 recorded no outcome, so all of its
  turns count there.
- A new user's first turn on record is in the range, so the first days after
  the import started count everyone as new.
- Someone who used both voice and chat is a user in each, since the hashes
  can't be matched, so `total.users` can count one person twice.

Set on the app: `TELEMETRY_DASHBOARD_PASSWORD`, `TELEMETRY_QUERY_API_KEY`, and
`TELEMETRY_CLICKHOUSE_HOST` / `TELEMETRY_CLICKHOUSE_PORT` if ClickHouse isn't on
localhost:8123. The environments read are `voice-production` and
`chat-production` unless `TELEMETRY_QUERY_VOICE_ENVIRONMENT` /
`TELEMETRY_QUERY_CHAT_ENVIRONMENT` say otherwise. Without the password or the
key the endpoints answer 503.

## A new field

A mapping can name extra values under `attributes`, and they land in the
`attributes` column as text, with no new column and no code change (see
`TELEMETRY_CHANGES.md`). The mapping refuses paths that can hold farmer text or a
phone number (`input`, `output`, `query`, `response`, `text`, `preview`,
`user_id`, ...), and a whole block is never flattened into it.

Give a value its own column only when dashboards need it typed or filter on it
a lot. Dashboards read `voice_turns`, so its columns never change name or type;
a new one is added instead. A field added to `CanonicalVoiceTurn` needs either a
column or an entry in `NOT_STORED` in `app/services/telemetry_import.py`, and
the tests fail until it has one. For a column:

1. Add an `ALTER TABLE telemetry.voice_turns ADD COLUMN IF NOT EXISTS ...` at the
   bottom of `voice.sql`. Never edit the `CREATE` above it.
2. Add the column to the end of `VOICE_TURN_COLUMNS` and fill it in `voice_turn_row`.
3. Add it to `RELEASED_VOICE_TURN_COLUMNS` in `tests/test_telemetry_import.py`
   once it ships.
4. Re-run `voice.sql`, then re-import the days you want it filled for.

## Rejections

The report lists every rejection reason. `Unknown voice schema version` means a
new stamp needs an entry in `telemetry/mappings/voice.yaml`. `No voice adapter
registered` means the trace falls outside every era in `telemetry/eras.yaml`.
The same reasons are on each rejected trace in `trace_ledger`.
