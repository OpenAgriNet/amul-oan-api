# Development interactive chat

`POST /api/chat/` is a development-only interactive chat endpoint. Milk UI is in amul-ui PR #18; AI booking requires the frontend additions below. **All accounts
and records come from the existing authenticated Amul APIs. There are no synthetic
accounts, generated milk records, fixture scenarios, or alternate data sources.**

## Enable and test

Both flags default off. The route is available only with `ENVIRONMENT=development`
and `CHAT_AGUI_PREVIEW_ENABLED=true`. Set `CHAT_AGUI_ENABLED=true` to enable controls.
The preview flag means development rollout, not mocked data. `/api/ui-flags/`
advertises `chat_agui_enabled` only when all three settings permit it. Changes
require restarting the service. Production always returns 404 for this POST.

Use a frontend build containing PR #18, with `VITE_AGUI_MOCK=false`, pointed at the
development API. Sign in with the registered farmer number. Ask to view milk
records, choose an account if required, select dates, and inspect the chart,
summary, daily records and deductions. Change account and repeat. Network requests
must be real POSTs, with `X-Chat-Format: agui-v1`, not the in-browser mock.

Enabled POST messages run the existing chat agent, including its moderation,
translation, and model fallback. The agent calls `request_milk_records` for a fresh
milk/payment lookup; unrelated questions receive ordinary agent answers. Clear,
complete dates can be supplied by the agent. Otherwise the backend shows the date
picker. Account selection and provider data stay under backend control. Existing
GET chat and voice are unchanged.
When AG-UI is off, POST messages delegate to the existing GET chat pipeline and
return its actual text/artifact stream with `X-Chat-Format: text-v1`. Disabling
controls clears the interactive selection on the next ordinary POST message.
WhatsApp and doctor-persona POST messages also use that existing text pipeline.

## Request and events

Send the existing `Authorization: Bearer <JWT>` plus `Content-Type: application/json`.
`threadId` is the frontend session UUID and `runId` identifies one new operation.
Use the same thread across turns, and a new run ID for each operation.

```json
{
  "threadId": "frontend-session-uuid",
  "runId": "new-operation-uuid",
  "state": {}, "tools": [], "context": [],
  "messages": [{"id": "new-message-uuid", "role": "user", "content": "Show my milk records"}],
  "forwardedProps": {"version": 1, "channel": "web", "source_lang": "gu", "target_lang": "gu"}
}
```

For a control submission use `messages: []` and put `ui_action` in `forwardedProps`:
`{component_id, revision, kind, value}`. Choice values contain `option_id`; date
values contain `fromdate` and `todate`. Account codes and mobile numbers are never
accepted from the client. The server re-fetches owned accounts before every
transition and fetches only the selected, currently owned account.

Account switching sends `command: {kind: "change_milk_account", state_revision}`
with the latest session revision. It carries the last selected period forward.
After a milk-provider failure, date-picker reselection is required before another
fetch, including after an account switch. A new message cannot bypass this by
supplying the failed period again. A successful date-picker submission clears the
requirement. Invalid/rejected submissions leave the stored control unchanged.
`confirm` is supported only for AI booking; chart submissions remain unsupported. The request and output models in `app/agui/models.py` are the source of truth;
JSON Schema can be generated from them when needed.

Dates must be real ISO calendar dates with `0 <= end - start <= 31 days`; this
preserves main's existing limit, which can cover 32 inclusive dates. No additional
provider history/future-date restriction is asserted by this backend.

Successful streams use standard AG-UI lifecycle, text, and full state-snapshot
events. An API failure is not an empty chart: after valid selections it produces
`CUSTOM amul.ui_error` followed by `RUN_ERROR`, keeping the selected values.
The frontend owns label/error-code translations; backend titles/anchors support
en/gu/hi/mr with English fallback. Provider names and numbers are not translated.

POST replies include `X-Trace-Id` and `Cache-Control: no-store`. Pre-stream errors
use `{error: {code, message, retryable, component_id?}}`. Validation errors never
echo raw input. Trace inputs contain only the server-owned operation description.
The user's free text is not stored in the interactive Redis record.
Masked conversation history follows the existing chat history keys and TTL.
The agent receives a bounded, ownership-checked account/result summary, and can
use `read_displayed_milk_records` for follow-ups without fetching new records.
Its pages contain at most 20 milk/deduction rows and 16 KiB, with explicit paging;
totals are calculated by the backend before pagination. Subranges outside the
saved period, or containing dates that cannot be attributed reliably, cannot be
reported as complete totals. A plain follow-up does not expire pending controls.

## State and limitations

Redis state is scoped to authenticated owner, persona and thread, separate from
normal chat history. The inactivity TTL follows chat history (default two hours).
Atomic commits reject stale controls, conflicting requests and repeated run IDs;
there is no replay of old streams. A short per-session lock protects provider
reads and agent execution, renews every 30 seconds with a 120-second lease, and is
checked before publishing committed state. History and UI changes commit together;
suggestion tasks are released only after that commit. Redis failure returns 503.
Agent text and artifacts are collected before publication, so this version sends
the completed turn as SSE after the atomic commit, rather than live token deltas.
Failed/cancelled model attempts publish no new controls. Repeated identical milk
tool calls coalesce into one workflow; conflicting ranges do not replace it.
Expired controls return 404: start a new chat/thread. Start a new thread after the
development session/component limit. The 50-component cap includes expired and
submitted controls/charts; ordinary messages and saved-result reads still work
at that cap. Reload restoration, additional interactive workflows, and full
interrupted-run recovery are deferred.

## AI booking contract — frontend handoff

@vishalrathore8oct: reuse the existing account-choice renderer for both new choice
purposes, and add the confirmation renderer. There is **no new account picker**.
Use the same POST transport, `agui-v1` snapshot wrapper, anchors and SDK versions.
This is an additive contract change; deploy the updated renderer before enabling
the coordinated booking demo. The backend rollout guards remain unchanged.

The agent stages `request_ai_booking(species=None)`. Backend account resolution
uses a still-owned milk selection, otherwise a sole authenticated farmer record;
missing, ambiguous or removed context stops booking. It never guesses an account.
The flow always shows technician selection, even for a single technician. It then
shows Cow/Buffalo only if the current booking's species was unclear, followed by
confirmation. Technician names and mobile descriptions come directly from API
records. Option IDs are opaque; do not submit technician IDs or account codes.

All controls retain the existing base fields:
`id, workflow_id, message_id, revision, title, status, kind, purpose, selected`.
The following examples show the additional fields on consecutive controls; IDs
are illustrative, not provider identifiers. Each control arrives inside
`STATE_SNAPSHOT.snapshot.amul_ui.components` with its assistant-message anchor.

```json
{
  "kind": "choice", "purpose": "ai_technician",
  "options": [{"id": "opaque-option", "label": "Technician name", "description": "Provider mobile"}],
  "selected": null
}
```

```json
{
  "kind": "choice", "purpose": "ai_species",
  "options": [
    {"id": "cow", "label": "Cow", "description": null},
    {"id": "buffalo", "label": "Buffalo", "description": null}
  ],
  "selected": null
}
```

Use the existing choice submission for both:
`ui_action: {component_id, revision, kind: "choice", value: {option_id}}`.
Labels/titles are localized by the backend (en/gu/hi/mr); preserve API names.
Do not hard-code account-specific labels on these two purposes.

```json
{
  "kind": "confirm", "purpose": "ai_booking",
  "summary": {
    "account_label": "Farmer / society label",
    "technician": {"id": "opaque-option", "label": "Technician name", "description": "Provider mobile"},
    "species": "cow", "species_label": "Cow"
  },
  "selected": null,
  "outcome": null
}
```

Render the summary with Confirm booking, Change technician, Change animal and
Cancel controls. Submit `kind: "confirm", value: {confirmed: true}` only after
the farmer presses Confirm booking. `confirmed: false` cancels without booking.
As for milk, actions use `messages: []`, a fresh `runId`, and the displayed control
ID/revision. Disable controls while a run is active and after submission; keep
the chosen values visible. Buttons/status labels are frontend translations.

At any pending booking stage, cancellation may also use:
`command: {kind: "cancel_ai_booking", workflow_id, state_revision}`.
Editing uses `command: {kind: "edit_ai_booking", workflow_id, state_revision,
field: "technician" | "species"}`. Use the latest session revision. Editing
invalidates the old confirmation, shows the relevant picker and requires another
confirmation. Neither command accepts provider identifiers. Unrelated chat and
milk controls preserve the pending booking, and vice versa.

After submission, the same confirmation carries `selected: {confirmed: true}` and
`outcome: {status, message, ticket}`. Status is `confirmed`, `rejected`, `not_sent`,
`unconfirmed`, or `already_requested`; ticket is nullable. The backend also emits
the actual result as assistant text. Only `confirmed` represents this workflow's
successful provider response. Existing session-limit messages may refer to a
previous ticket; do not present that as a new booking.

Failures emit `CUSTOM amul.ui_error` and `RUN_ERROR` using the existing error
envelope. New codes: `BOOKING_CONTEXT_UNAVAILABLE`, `TECHNICIANS_UNAVAILABLE`,
`NO_AI_TECHNICIANS`, `AI_BOOKING_NOT_ALLOWED`, and `AI_BOOKING_` followed by
`REJECTED`, `NOT_SENT`, `UNCONFIRMED`, or `ALREADY_REQUESTED`. Translate these on the
frontend, retaining the backend message as fallback. Never automatically retry a
confirmation or create a fresh booking after an uncertain result.

The backend revalidates ownership, union policy and technician availability on
each transition. Changed review details require another confirmation. Before any
external booking write it atomically consumes the confirmation and records the
run. A crash, disconnection, or failed result commit leaves it consumed with an
uncertain outcome, so neither an old action nor a fresh run ID can resend it.
This guarantee applies to the same confirmation, not to separately confirmed
bookings. Existing session-wide booking-guard settings still apply. No automatic
recovery/rebooking, production rollout, or live booking is included here.

Frontend acceptance: technician → optional species → review → confirm; editing
and cancellation; submitted/expired controls; duplicate submissions; provider
failures and uncertain outcomes; Gujarati and phone-width layout.

Python protocol package: `ag-ui-protocol==1.0.0`; frontend compatibility target:
`@ag-ui/client` / `@ag-ui/core` `1.0.4`. Versions need not share a number.


The backend stream was checked against Vishal's actual PR #18 adapter with
one-byte UTF-8/SSE fragments. This one-off verification tooling is not a runtime
dependency and is not included in the backend source. Real farmer-data validation
still requires signing in and checking the records in the UI.
