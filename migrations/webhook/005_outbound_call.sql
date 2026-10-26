-- Outbound heat-alert call: when it is due, the provider's call id, and why it failed.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS call_due_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS outbound_call_uuid VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_error VARCHAR(64);
