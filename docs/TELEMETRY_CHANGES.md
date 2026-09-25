# Changing voice telemetry

Step by step for adding or changing what a voice turn sends to Langfuse. Each
step names the file to edit. The telemetry tests check every step, and when one
fails its message says what's missing.

Two repos are involved:

- **voice-oan-api** sends the trace.
- **amul-oan-api** reads it: the mapping, the canonical model and `telemetry/eras.yaml`.

Chat will follow the same steps once it has `telemetry/mappings/chat.yaml`.
Until then, chat changes go through the chat adapter.

## Add a new field

Example: record `farmer_type` on every voice turn.

1. voice-oan-api: set it on the trace, e.g. `trace.metadata["farmer_type"] = ...`
   in `app/services/voice.py`, or in `VoiceTrace` if every turn has it.
2. voice-oan-api: add `"farmer_type"` to `metadata_keys` in
   `telemetry/contracts/voice.turn.v1.json`. No version bump.

That's enough if the field is only for looking at traces in Langfuse. To get it
into the canonical output that dashboards read, also:

3. amul-oan-api: add it to `CanonicalVoiceTurn` in
   `app/models/telemetry_voice_analytics.py`, e.g. `farmer_type: str | None = None`.
4. amul-oan-api: add it to `_MAPPED_FIELDS` in
   `app/services/telemetry_voice_era_adapters.py`, with how to read it:
   `_string_or_none` for text, `_float_or_none` for numbers, `_bool_or_none`,
   or `_identifier_or_none` for ids that can arrive as numbers.
5. amul-oan-api: say where it lives in `telemetry/mappings/voice.yaml`, under the
   current version: `farmer_type: [metadata.farmer_type]`. A key inside a block
   works too: `[metadata.farmer_context.source]`.
6. Add a test with a stamped trace that carries the field, next to the other
   stamped tests in `tests/test_telemetry_voice_era_adapters.py`.

Traces from before the change don't have the field, so it reads as
`unavailable` for them. That's expected.

## Add an outcome value

1. voice-oan-api: set it, e.g. `trace.set_outcome("cancelled")`.
2. voice-oan-api: add it to `outcomes` in the contract file.
3. amul-oan-api: add it to one bucket of `voice_outcome_vocabulary` in
   `telemetry/eras.yaml`: `delivered`, `non_question`, `refused_or_blocked` or
   `failed`. Until then it's counted as `unclassified`.

## Rename or remove a field, or change what it means

Example: `outcome` becomes `turn_outcome`.

1. voice-oan-api: change the code.
2. voice-oan-api: bump `VOICE_TELEMETRY_SCHEMA_VERSION` in
   `app/services/telemetry_stamps.py`, e.g. to `voice.turn.v2`.
3. voice-oan-api: copy the contract to `telemetry/contracts/voice.turn.v2.json`
   and update it. Leave the v1 file as it is; old traces still follow it.
4. amul-oan-api: add the version to `telemetry/mappings/voice.yaml`, listing only
   what moved:

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
| `New metadata keys [...]` | List them in the contract file. |
| `Voice traces no longer send [...]` | A key was renamed or removed: see "Rename or remove". |
| `New outcomes [...]` | See "Add an outcome value". |
| `Outcomes no longer emitted: [...]` | Remove them from the contract, and note it in `eras.yaml` once it ships. |
| `No contract for voice.turn.vN` | Add the contract file for the version you bumped to. |
| `...: 'x' is not a canonical field` | Typo in `voice.yaml`, or step 4 of "Add a new field" is missing. |
| `Unknown voice schema version` | The new version isn't in `voice.yaml` yet. |
| `... ends at ... but no ... root era starts then` | An `eras.yaml` boundary leaves a gap: start the next era at the same instant. |
