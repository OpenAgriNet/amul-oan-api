# Partner Heat-Alert Webhook

Inbound partner alerts are accepted at:

`POST /api/webhooks/alerts`

The alert kind is not in the URL. `notification_type` in the body is required.
Today the only value is `heat`. Input is case-insensitive and stored as lowercase.
A missing or unknown value returns `422`.

Auth:
- `Authorization: Bearer <token>` (a static shared secret, not a JWT)
- token value must match `WEBHOOK_SHARED_TOKEN`
- missing/invalid token → `401`
- token unset in env → `503` (fail closed)

Request body uses the partner's wire keys (including spaces/casing), for example:
`alertID`, `deviceid`, `tag no`, `Pashuaadhar No`, `AI Window start_time`.
`farmer_contact`, `Pashuaadhar No`, `tag no`, and `booking_url` must be non-empty
because they are sent to the single-farmer Sarlaben WhatsApp API. A missing value
returns `422` before the event is stored.

For each newly stored alert, the service calls
`POST <SARLABEN_WHATSAPP_BASE_URL>/api/whatsapp/heat-alert` with this mapping:
- `farmer_contact` → `mobile`
- `Pashuaadhar No` → `animalNo`
- `tag no` → `tagNo`
- `booking_url` → `bookingLink`

The database idempotency key (`alertID`, `farmer_contact`) also prevents a webhook
retry from sending the same WhatsApp message twice. The webhook returns `503` when
the WhatsApp API key is missing and `502` when the provider does not accept the
message; in either case the new database row is rolled back so a retry can try again.
The send happens before the commit, so if the commit itself fails after a successful
send, the partner gets `500` and its retry sends a second message.

The internal `called` and `call_booked` status columns both default to `false`.
They are managed by application actions and are not accepted from webhook payloads.

## Outbound call

With `RAYA_HEAT_ALERT_AGENT_ID` and `RAYA_OUTBOUND_CALL_API_KEY` set, each new alert
gets a call `HEAT_ALERT_CALL_DELAY_SECONDS` (default 600) after it is stored, through
the Raya call API (`RAYA_OUTBOUND_CALL_API_URL`) with that agent. The agent receives
`agent_args`: `script` (`assets/prompts/heat_alert_call_script_gu.md` rendered
with the animal and tag numbers), `animal_no`, `tag_no`, `alert_id`.

Before dialling, the row is claimed by setting `called=true` in one statement that
also checks `call_booked=false` and that the AI window has not ended, so a second
replica or a restart cannot ring the same farmer twice. Calls due between 22:00 and
06:00 IST move to 06:00. Then two checks, each recorded in `call_error` when it
stops the call: `HEAT_ALERT_CALL_ONLY_NUMBERS` (test phase: when set, only those
numbers are rung, `not_in_only_numbers`) and Amul Pashudhan's `GetFarmerTicketSummary`
for the alert's account (`pending_ticket`). The account is the alert's union, society
and farmer code; when the partner omits the union code it is taken from the farmer
cache entry for that phone that matches the other two. A ticket lookup that fails
does not stop the call.

The provider's call id lands in `outbound_call_uuid`. A refused call (4xx) releases
the claim with the reason in `call_error`; a timeout or 5xx keeps `called=true`,
since the call may exist. Timers live in the process; on startup every row with a
due time and `called=false` is rescheduled.

When the farmer answers, Raya sends the voice turn with `call_type=outbound`. The
voice adapter matches that phone to a call we placed in the last three hours
(`outbound_call_uuid` set), keeps the match on the session in Redis, skips the
milk-readout consent gate, and tells the agent which animal and account the call
is about so a "yes" runs the AI booking for it. A booking made on that session
sets `call_booked=true`.

Successful response (`202 Accepted`):
```json
{
  "accepted": true,
  "id": "<uuid>",
  "received_at": "<iso8601-utc>"
}
```

Persistence:
- rows are stored in a dedicated Postgres database (`WEBHOOK_DB_URL`)
- table: `heat_alert_webhook_events` (see `migrations/webhook/001_init.sql`)
- full payload is kept in `payload_raw` (JSONB) plus selected searchable columns
- DB missing/unconfigured → `503`

Retention:
- background worker hard-deletes rows older than `WEBHOOK_RETENTION_HOURS` (default `24`)
- cadence: `WEBHOOK_CLEANUP_INTERVAL_SECONDS` (default `900`)
- worker is a no-op when `WEBHOOK_DB_URL` is unset

Env reference: the heat-alert block in `example.env`.

## Deploy checklist

1. Create a dedicated Postgres DB/user for webhook storage.
2. Apply the schema migration before deploying the matching application code:
   - fresh database: `psql "$WEBHOOK_DSN" -f migrations/webhook/001_init.sql`
   - existing database: apply `migrations/webhook/003_booking_url.sql`, `migrations/webhook/004_call_flags.sql`, then `migrations/webhook/005_outbound_call.sql`
3. Set runtime env on this service:
   - `WEBHOOK_SHARED_TOKEN`
   - `WEBHOOK_DB_URL` (`postgresql+asyncpg://...`)
   - `SARLABEN_WHATSAPP_API_KEY`
   - optional: `SARLABEN_WHATSAPP_BASE_URL` (defaults to `https://comm.amul.in/sarlaben`)
   - optional: `SARLABEN_WHATSAPP_TIMEOUT_SECONDS` (defaults to `15`)
   - optional: `WEBHOOK_DB_POOL_SIZE`, `WEBHOOK_RETENTION_HOURS`, `WEBHOOK_CLEANUP_INTERVAL_SECONDS`
   - outbound call: `RAYA_OUTBOUND_CALL_API_KEY`, `RAYA_HEAT_ALERT_AGENT_ID`; optional `RAYA_OUTBOUND_CALL_API_URL`, `HEAT_ALERT_CALL_DELAY_SECONDS`, `HEAT_ALERT_CALL_ONLY_NUMBERS`
4. Restart the API process and confirm startup log includes webhook cleanup worker start (or the "not started: WEBHOOK_DB_URL not configured" skip line).
5. Smoke test:
   ```bash
   SMOKE_ALERT_ID="smoke-booking-url-$(date +%s)"
   SMOKE_TIMESTAMP="$(date -u +'%Y-%m-%dT%H:%M:%S.000Z')"
   SMOKE_RESPONSE_FILE="$(mktemp)"
   : "${SMOKE_FARMER_MOBILE:?set a safe test WhatsApp recipient}"
   : "${SMOKE_ANIMAL_NO:?set a test animal number}"
   : "${SMOKE_TAG_NO:?set a test tag number}"
   : "${SMOKE_BOOKING_URL:?set a test booking URL}"

   SMOKE_HTTP_STATUS="$(curl -sS -o "$SMOKE_RESPONSE_FILE" -w '%{http_code}' \
     -X POST "$BASE_URL/api/webhooks/alerts" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $WEBHOOK_SHARED_TOKEN" \
     -d "{
       \"alertID\": \"$SMOKE_ALERT_ID\",
       \"deviceid\": \"SMOKE-BOOKING-URL\",
       \"alert_type\": \"Amber\",
       \"notification_type\": \"heat\",
       \"AI Window start_time\": \"$SMOKE_TIMESTAMP\",
       \"AI Window end_time\": \"$SMOKE_TIMESTAMP\",
       \"timestamp\": \"$SMOKE_TIMESTAMP\",
       \"farmer_contact\": \"$SMOKE_FARMER_MOBILE\",
       \"Pashuaadhar No\": \"$SMOKE_ANIMAL_NO\",
       \"tag no\": \"$SMOKE_TAG_NO\",
       \"booking_url\": \"$SMOKE_BOOKING_URL\"
     }")"

   cat "$SMOKE_RESPONSE_FILE"
   test "$SMOKE_HTTP_STATUS" = "202"
   ```
6. Verify both persisted copies of the URL using the same shell variables:
   ```bash
   SMOKE_MATCHES="$(psql "$WEBHOOK_DSN" -Atc "
     SELECT count(*)
     FROM heat_alert_webhook_events
     WHERE alert_id = '$SMOKE_ALERT_ID'
       AND booking_url = '$SMOKE_BOOKING_URL'
       AND payload_raw->>'booking_url' = '$SMOKE_BOOKING_URL';
   ")"

   test "$SMOKE_MATCHES" = "1"
   echo "booking_url persistence smoke test passed"
   rm -f "$SMOKE_RESPONSE_FILE"
   ```
7. Confirm the smoke row disappears after the configured retention period.
