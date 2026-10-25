-- Durable, rate-limited outbound calls for newly received heat alerts.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS call_due_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS call_next_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS call_status VARCHAR(32) NOT NULL DEFAULT 'scheduled',
    ADD COLUMN IF NOT EXISTS call_eligibility_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS call_dispatch_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS call_lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS call_lease_token VARCHAR(36),
    ADD COLUMN IF NOT EXISTS call_last_error VARCHAR(64),
    ADD COLUMN IF NOT EXISTS call_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS resolved_union_code VARCHAR(64),
    ADD COLUMN IF NOT EXISTS outbound_call_uuid VARCHAR(64),
    ADD COLUMN IF NOT EXISTS voice_session_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS voice_process_id VARCHAR(128);

CREATE INDEX IF NOT EXISTS ix_heat_alert_events_call_due_at
    ON heat_alert_webhook_events (call_due_at);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_call_next_attempt_at
    ON heat_alert_webhook_events (call_next_attempt_at);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_call_lease_expires_at
    ON heat_alert_webhook_events (call_lease_expires_at);
CREATE UNIQUE INDEX IF NOT EXISTS ix_heat_alert_events_outbound_call_uuid
    ON heat_alert_webhook_events (outbound_call_uuid)
    WHERE outbound_call_uuid IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_voice_session_id
    ON heat_alert_webhook_events (voice_session_id);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_voice_process_id
    ON heat_alert_webhook_events (voice_process_id);

CREATE TABLE IF NOT EXISTS outbound_call_dispatch_state (
    id INTEGER PRIMARY KEY,
    next_allowed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO outbound_call_dispatch_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;
