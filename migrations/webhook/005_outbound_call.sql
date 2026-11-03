-- Outbound heat-alert call: when it is due, the provider's call id, why it was not
-- placed, and the partner's union code when it sends one.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS union_code VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_due_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS outbound_call_uuid VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_error VARCHAR(64);
