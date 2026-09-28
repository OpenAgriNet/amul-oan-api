# Telemetry Standard

Rules for anyone changing what chat or voice sends to Langfuse. The goal is
that a change never silently breaks the adapters or the dashboards that read
this data.

## Every trace is stamped

Turn traces carry three metadata keys, set in `app/services/telemetry_stamps.py`
(same file name in both repos):

- `amul.schema_version`: the format of the trace, e.g. `voice.turn.v1`
- `service`: `amul-oan-api` or `voice-oan-api`
- `release`: the git commit of the running code, the same way in both repos:
  the checkout's HEAD, else `GIT_SHA` from the image build, else `unknown`

Adapters pick the format from `amul.schema_version`. Only traces from before
the stamp existed are matched by root name and date.

The stamp is the version of the trace. What the adapters produce has its own
version, `voice.canonical.v1` or `chat.canonical.v1`, which only changes when
the canonical model does.

## When you change what a trace sends

A released contract never changes: traces already in Langfuse follow it. So
**every** change to what a trace sends is a new schema version, adding a key
included. For each one: bump the stamp, copy the contract to the new version's
file and change it there, and add the version to
`telemetry/mappings/<channel>.yaml`. The new mapping `extends` the old one and
lists only what moved, so an added key is two lines.

| Change | New version, plus |
| --- | --- |
| Add a metadata key | Nothing else. To show it on a dashboard, map it (see "Add a new field" in `TELEMETRY_CHANGES.md`). |
| Add an outcome value | A bucket for it in `voice_outcome_vocabulary` or `chat_outcome_vocabulary` in `telemetry/eras.yaml`. |
| Rename or remove a key | In the new mapping, point the canonical field at the new key. |
| Rename the root, drop a trace field (`sessionId`, `userId`, input, output), or change a key inside a metadata block | The new `root:` or path in the mapping. The contracts list these too (voice: `root`, `trace_fields`, `nested_keys`; chat: `root`, `trace_input`). |
| Keep a key but change what it means | Point the canonical field at a key that means the right thing, or leave it out. Nothing can detect this for you. |

Names say what the value is: lowercase snake_case, never `data`, `id`, `result`,
`status`, `time`, `type` or `value` on their own. The contract tests check this.

Contract files live in `telemetry/contracts/<schema version>.json` in the repo
that sends the trace: `voice.turn.v1.json` in voice-oan-api, `chat.turn.v1.json`
here. The telemetry tests fail with the exact step to take when the code and
the contract disagree.

Once the new version is live in production, add an era to `telemetry/eras.yaml`
with the first production day and its `schema_version`. It's a record of when it
went live; the adapters read stamped traces without it.

Every step, with the file to edit: `TELEMETRY_CHANGES.md`.

## telemetry/eras.yaml

- Append eras, never rewrite one. A corrected boundary keeps a note of the old value.
- `valid_from` is the day it shows up in production data, not the merge date.
- When a root era ends, the next one starts at the same instant. Use a full UTC
  time (`2026-08-05T06:30:00Z`) when the change happened mid-day.
- `check_registry` in `app/services/telemetry_era_registry.py` checks these rules
  and runs in CI.

## Adapters

- Stamp first, then root name + date. Anything else is rejected with a reason, never guessed.
- Every canonical field says whether it was `recorded` (the trace carried it),
  `derived` (computed, or taken from an older name or another trace) or
  `unavailable`. Never fill in a value the trace didn't have.
- No outcome recorded means `outcome_class` is null. A recorded outcome missing
  from the vocabulary becomes `unclassified`, so it still shows up in counts.
- Anonymous users get a null `user_id_hash` in both channels, so they are not
  counted as one user.
- No trace is dropped without a record: every root trace the import reads gets a
  row in `telemetry.trace_ledger` (turn, rejected with the reason, activity, or
  unrecognised). A new kind of trace shows up there instead of disappearing.
- Extra values go under `attributes` in the mapping and are stored as text, so a
  new field needs no migration. Attributes may never read a path that can hold
  farmer text or a phone number.
- Test fixtures are redacted: no phone numbers or farmer text. This repo is public.

## Reading telemetry

Anything that reads this data (a dashboard, an export, a report) goes in
`telemetry/consumers.yaml`, with what it reads and what for. Before removing or
renaming something, check that file for who depends on it.

## CI

The `telemetry` workflow runs `tests/test_telemetry_*.py` and the stamp tests on
every PR. Keep it green; a new telemetry test file named `test_telemetry_*.py`
is picked up automatically.

See `TELEMETRY_VERSIONED_ADAPTERS.md` for the adapters themselves.
