# Development milk UI

`POST /api/chat/` is a development-only milk UI for amul-ui PR #18. **All accounts
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

This initial development mode is deliberately limited: each new message on the
enabled POST starts a milk lookup. It does not run an agent or interpret dates in
free text. Select dates in the picker. Existing GET chat and voice are unchanged.
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
`confirm` and chart submissions are unsupported. The request and output models in `app/agui/models.py` are the source of truth;
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

## State and limitations

Redis state is scoped to authenticated owner, persona and thread, separate from
normal chat history. The inactivity TTL follows chat history (default two hours).
Atomic commits reject stale controls, conflicting requests and repeated run IDs;
there is no replay of old streams. A short per-session lock protects provider
reads and is checked before publishing committed state. Redis failure returns 503.
Expired controls return 404: start a new chat/thread. Start a new thread after the
development session/component limit. Reload restoration, general agent-driven
controls and full interrupted-run recovery are deferred.

Python protocol package: `ag-ui-protocol==1.0.0`; frontend compatibility target:
`@ag-ui/client` / `@ag-ui/core` `1.0.4`. Versions need not share a number.


The backend stream was checked against Vishal's actual PR #18 adapter with
one-byte UTF-8/SSE fragments. This one-off verification tooling is not a runtime
dependency and is not included in the backend source. Real farmer-data validation
still requires signing in and checking the records in the UI.
