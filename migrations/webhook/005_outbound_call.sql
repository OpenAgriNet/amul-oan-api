-- Outbound heat-alert call: when it is due, the provider's call id, and why it failed.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS union_code VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_due_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS call_status VARCHAR(32) NOT NULL DEFAULT 'scheduled',
    ADD COLUMN IF NOT EXISTS call_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS outbound_call_uuid VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_error VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS voice_session_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS voice_process_id VARCHAR(128);

CREATE INDEX IF NOT EXISTS ix_heat_alert_events_union_code
    ON heat_alert_webhook_events (union_code);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_call_due_at
    ON heat_alert_webhook_events (call_due_at);
CREATE UNIQUE INDEX IF NOT EXISTS ix_heat_alert_events_outbound_call_uuid
    ON heat_alert_webhook_events (outbound_call_uuid)
    WHERE outbound_call_uuid IS NOT NULL;
