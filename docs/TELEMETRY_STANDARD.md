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

| Change | What to do |
| --- | --- |
| Add a metadata key | Add it to the contract file. No version change. |
| Add an outcome value | Voice: add it to the contract file and to `voice_outcome_vocabulary` in `telemetry/eras.yaml`. Chat: add it to `chat_outcome_vocabulary`. No version change. |
| Rename or remove a key | Bump the schema version, add a contract file for it, and add the version to `telemetry/mappings/<channel>.yaml` (it can `extends` the old one and list only what moved). |
| Rename the root, drop a trace field (`sessionId`, `userId`, input, output), or rename a key inside a metadata block | Same as a rename. The contracts list these too (voice: `root`, `trace_fields`, `nested_keys`; chat: `root`, `trace_input`). |
| Keep a key but change what it means | Same as a rename. Nothing can detect this for you. |

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
