-- Partner heat-alert webhook payload storage (short-lived; retained for 24h by app job)
-- Apply in the dedicated webhook Postgres database on vm2.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS heat_alert_webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    alert_id VARCHAR(64) NOT NULL,
    device_id VARCHAR(128) NOT NULL,
    notification_type VARCHAR(64),
    alert_type VARCHAR(32),
    heat_score VARCHAR(32),

    tag_no VARCHAR(64),
    system_tag_no VARCHAR(64),
    pashuaadhar_no VARCHAR(64),

    farm_name VARCHAR(256),
    farm_id VARCHAR(64),
    union_code VARCHAR(64),
    society_code VARCHAR(64),
    farmer_code VARCHAR(64),
    farmer_contact VARCHAR(32),

    ai_window_start_time_raw TEXT,
    ai_window_end_time_raw TEXT,
    alert_time_raw TEXT,
    timestamp_raw TEXT,
    msg_title TEXT,
    msg_body TEXT,
    booking_url TEXT,
    called BOOLEAN NOT NULL DEFAULT FALSE,
    call_booked BOOLEAN NOT NULL DEFAULT FALSE,
    call_due_at TIMESTAMPTZ,
    call_next_attempt_at TIMESTAMPTZ,
    call_status VARCHAR(32) NOT NULL DEFAULT 'scheduled',
    call_eligibility_attempts INTEGER NOT NULL DEFAULT 0,
    call_dispatch_attempts INTEGER NOT NULL DEFAULT 0,
    call_lease_expires_at TIMESTAMPTZ,
    call_lease_token VARCHAR(36),
    call_last_error VARCHAR(64),
    call_updated_at TIMESTAMPTZ,
    outbound_call_uuid VARCHAR(64),
    voice_session_id VARCHAR(128),
    voice_process_id VARCHAR(128),

    ai_window_start_at TIMESTAMPTZ,
    ai_window_end_at TIMESTAMPTZ,
    partner_timestamp_at TIMESTAMPTZ,
    partner_alert_epoch_s BIGINT,

    payload_raw JSONB NOT NULL,
    CONSTRAINT uq_heat_alert_events_alert_id_farmer_contact
        UNIQUE (alert_id, farmer_contact)
);

CREATE INDEX IF NOT EXISTS ix_heat_alert_events_received_at
    ON heat_alert_webhook_events (received_at);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_farmer_contact
    ON heat_alert_webhook_events (farmer_contact);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_device_id
    ON heat_alert_webhook_events (device_id);
CREATE INDEX IF NOT EXISTS ix_heat_alert_webhook_events_alert_id
    ON heat_alert_webhook_events (alert_id);
CREATE INDEX IF NOT EXISTS ix_heat_alert_webhook_events_society_code
    ON heat_alert_webhook_events (society_code);
CREATE INDEX IF NOT EXISTS ix_heat_alert_webhook_events_farmer_code
    ON heat_alert_webhook_events (farmer_code);
CREATE INDEX IF NOT EXISTS ix_heat_alert_events_union_code
    ON heat_alert_webhook_events (union_code);
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
    next_allowed_at TIMESTAMPTZ
);

INSERT INTO outbound_call_dispatch_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;
