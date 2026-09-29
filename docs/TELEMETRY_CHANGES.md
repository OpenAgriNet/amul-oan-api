# Changing voice telemetry

Step by step for adding or changing what a voice turn sends to Langfuse. Each
step names the file to edit. The telemetry tests check every step, and when one
fails its message says what's missing.

Two repos are involved:

- **voice-oan-api** sends the trace.
- **amul-oan-api** reads it: the mapping, the canonical model and `telemetry/eras.yaml`.

Chat has the same pieces, all in this repo: `telemetry/contracts/chat.turn.v1.json`,
`telemetry/mappings/chat.yaml` and `chat_outcome_vocabulary`. The examples below
use voice.

## Every change is a new version

A released contract never changes, because traces already in Langfuse follow it.
Whatever you change (a key or an outcome added, renamed or removed, the root, a
key inside a block, or what a key means), release it as a new version:

1. voice-oan-api: bump `VOICE_TELEMETRY_SCHEMA_VERSION` in
   `app/services/telemetry_stamps.py`, e.g. to `voice.turn.v2`.
2. voice-oan-api: copy `telemetry/contracts/voice.turn.v1.json` to
   `voice.turn.v2.json` and make the change there. Leave the v1 file as it is;
   the tests fail if it changes.
3. amul-oan-api: add the version to `telemetry/mappings/voice.yaml`. It extends
   the old one and lists only what moved. For a key that was only added, two
   lines are enough:

   ```yaml
   voice.turn.v2:
     extends: voice.turn.v1
   ```

4. Merge the amul-oan-api change first. voice-oan-api's CI checks that its main
   branch already reads the new version.
5. Once it ships, add the new version's fingerprint to `RELEASED_CONTRACTS` in
   voice-oan-api's `tests/test_voice_telemetry_contract.py` (the command is in its
   `telemetry/README.md`).

Chat is the same, all in this repo: `CHAT_TELEMETRY_SCHEMA_VERSION`,
`telemetry/contracts/chat.turn.v2.json` and `telemetry/mappings/chat.yaml`.

## Add a new field

Example: record `farmer_type` on every voice turn.

1. voice-oan-api: set it on the trace, e.g. `trace.metadata["farmer_type"] = ...`
   in `app/services/voice.py`, or in `VoiceTrace` if every turn has it.
2. Release it as a new version, with `"farmer_type"` in the new contract's
   `metadata_keys` (see "Every change is a new version").

That's enough if the field is only for looking at traces in Langfuse. To get it
into the telemetry tables that dashboards read:

3. amul-oan-api: in the new version's entry in `telemetry/mappings/voice.yaml`,
   name it under `attributes`:

   ```yaml
   voice.turn.v2:
     extends: voice.turn.v1
     attributes:
       farmer_type: [metadata.farmer_type]
   ```

   A key inside a block works too: `[metadata.farmer_context.source]`. No code,
   no new column: it lands in the `attributes` column as text, and dashboards
   read `attributes['farmer_type']`. The mapping refuses paths that can hold
   farmer text or a phone number. Attributes are read on stamped versions
   (`voice.turn.vN`, `chat.turn.vN`) only, and the readers checks fail if one
   reads a key the contract doesn't send.
4. Once it's live, re-import the days you want it filled for.

Traces from before the change don't have it, so `attributes` has no
`farmer_type` for them. That's expected.

### Promote it to its own column

Only when dashboards need it typed (a number, a boolean) or filter on it a lot:

1. amul-oan-api: add it to `CanonicalVoiceTurn` in
   `app/models/telemetry_voice_analytics.py`, e.g. `farmer_type: str | None = None`.
2. amul-oan-api: add it to `_MAPPED_FIELDS` in
   `app/services/telemetry_voice_era_adapters.py`, with how to read it:
   `_string_or_none` for text, `_float_or_none` for numbers, `_bool_or_none`,
   or `_identifier_or_none` for ids that can arrive as numbers.
3. amul-oan-api: move it from `attributes` to `fields` in the mapping:
   `farmer_type: [metadata.farmer_type]`.
4. Add a test with a stamped trace that carries the field, next to the other
   stamped tests in `tests/test_telemetry_voice_era_adapters.py`.
5. amul-oan-api: give it a column. Add
   `ALTER TABLE telemetry.voice_turns ADD COLUMN IF NOT EXISTS farmer_type LowCardinality(Nullable(String));`
   at the bottom of `telemetry/clickhouse/voice.sql`, then put the name at the end
   of `VOICE_TURN_COLUMNS` and the value in `voice_turn_row`, both in
   `app/services/telemetry_import.py`. More in "A new field" in
   `TELEMETRY_PIPELINE.md`.
6. Once it's merged, re-run `voice.sql` on the ClickHouse, then re-import the days
   you want the field filled for. Before the change it reads as `unavailable`.

## Add an outcome value

1. voice-oan-api: set it, e.g. `trace.set_outcome("cancelled")`.
2. Release it as a new version, with the outcome in the new contract's `outcomes`.
3. amul-oan-api: add it to one bucket of `voice_outcome_vocabulary` in
   `telemetry/eras.yaml`: `delivered`, `non_question`, `refused_or_blocked` or
   `failed`. Until then it's counted as `unclassified`.

## Rename or remove a field, or change what it means

Example: `outcome` becomes `turn_outcome`.

1. voice-oan-api: change the code.
2. Release it as a new version (see "Every change is a new version"). In the new
   mapping, list only what moved:

   ```yaml
   voice.turn.v2:
     extends: voice.turn.v1
     fields:
       outcome: [metadata.turn_outcome]
   ```

   If a field kept its name but changed meaning, point it at a key that means the
   right thing, or leave it out so it reads as unavailable. Never map a canonical
   field to a value that means something else.

No Python change is needed for this.

## Once a new version is in production

Add an era to `voice_eras` in `telemetry/eras.yaml` with the first production day
(UTC) and `schema_version: voice.turn.v2`, so the history shows when it went live.
The adapter doesn't wait for this: a stamped trace is read by its stamp, and its
`source_era` is the stamp itself.

## When something fails

| Message | What to do |
| --- | --- |
| `New metadata keys [...]` | A key was added: release a new version with it in the new contract. |
| `New keys inside metadata blocks` | Same, for a key inside a block like `agent`. |
| `Voice traces no longer send [...]` | A key was renamed or removed: see "Rename or remove". |
| `Voice traces no longer send these keys inside metadata blocks` | A key inside a block like `agent` was renamed or removed: see "Rename or remove". |
| `Voice traces no longer send the trace fields [...]` | `input`, `output`, `sessionId` or `userId` stopped being sent: see "Rename or remove". |
| `Voice turns are now sent as [...]` | The root was renamed: see "Rename or remove", and set the new `root:` in `voice.yaml`. |
| `New outcomes [...]` | See "Add an outcome value". |
| `Outcomes no longer emitted: [...]` | Release a new version without them, and note it in `eras.yaml` once it ships. |
| `telemetry/contracts/voice.turn.v1.json is released and can't change` | Undo the edit to the old file and put the change in a new version. |
| `Unclear key names [...]` | Rename the key to say what it holds, e.g. `error_type`, not `type`. |
| `No contract for voice.turn.vN` | Add the contract file for the version you bumped to. |
| `chat.turn.vN reads ... but the contract doesn't send it` | `chat.yaml` reads a key the chat contract no longer lists: point the field at a key that is sent. |
| `chat.turn.vN has no entry in telemetry/mappings/chat.yaml` | Add the version to `chat.yaml` in the same change as the contract. |
| `...: 'x' is not a canonical field` | Typo in `voice.yaml`, a new field meant for `attributes` is under `fields`, or step 2 of "Promote it to its own column" is missing. |
| `...attributes.x reads [...], which can hold farmer text or a phone number` | That path can't be stored. Map a key that holds no farmer text, or leave it out. |
| `...attributes: 'x' needs a specific lowercase snake_case name` | Rename the attribute to say what it holds. |
| `Unknown voice schema version` | The new version isn't in `voice.yaml` yet. |
| `... ends at ... but no ... root era starts then` | An `eras.yaml` boundary leaves a gap: start the next era at the same instant. |
| `CanonicalVoiceTurn has [...], but telemetry.voice_turns doesn't store it` | Step 5 of "Promote it to its own column" is missing, or list the field in `NOT_STORED` with the reason. |
| `telemetry.voice_turns and the importer disagree` | A column is in `voice.sql` but not in `VOICE_TURN_COLUMNS`, or the other way round. |
| `telemetry.voice_turns changed the released columns [...]` | A released column was renamed, retyped or removed. Put it back and add a new column instead. |
