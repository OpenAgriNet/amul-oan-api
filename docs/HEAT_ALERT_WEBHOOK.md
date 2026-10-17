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
   - existing database: apply `migrations/webhook/003_booking_url.sql`, then `migrations/webhook/004_call_flags.sql`
3. Set runtime env on this service:
   - `WEBHOOK_SHARED_TOKEN`
   - `WEBHOOK_DB_URL` (`postgresql+asyncpg://...`)
   - `SARLABEN_WHATSAPP_API_KEY`
   - optional: `SARLABEN_WHATSAPP_BASE_URL` (defaults to `https://comm.amul.in/sarlaben`)
   - optional: `SARLABEN_WHATSAPP_TIMEOUT_SECONDS` (defaults to `15`)
   - optional: `WEBHOOK_DB_POOL_SIZE`, `WEBHOOK_RETENTION_HOURS`, `WEBHOOK_CLEANUP_INTERVAL_SECONDS`
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
