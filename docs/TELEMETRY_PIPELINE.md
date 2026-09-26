# Telemetry Pipeline

Voice and chat turns go from Langfuse's ClickHouse into a separate `telemetry`
database on the same server:

```
Langfuse tables (traces, observations, scores)
  -> app/services/telemetry_fetcher.py   one UTC day at a time
  -> voice or chat adapters              CanonicalVoiceTurn / CanonicalChatTurn
  -> app/services/telemetry_import.py    telemetry.voice_turns / telemetry.chat_turns
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

The tables are in `telemetry/clickhouse/voice.sql` and `chat.sql`.

Chat needs more than voice to build a turn, so the importer reads the root's
input and output, and the full rows of the few observations the chat adapters
use (the agent run, stream_translation and tools). They are hashed or measured
in memory and never stored. A c2 turn's question sits in a pretranslation trace
of the same session, so each day is read from 2 minutes before midnight.

## Setup, once per ClickHouse

1. Run `telemetry/clickhouse/voice.sql` and `chat.sql` as the ClickHouse admin.
   They're safe to re-run, and need re-running whenever a change adds a column.
2. Create the users in `telemetry/clickhouse/users.sql`. `telemetry_reader` can
   only read Langfuse's three tables, and `telemetry_writer` can only write the
   `telemetry` database.
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

## Adding a column

Dashboards read `voice_turns`, so its columns never change name or type; a new
one is added instead. A field added to `CanonicalVoiceTurn` needs either a
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
