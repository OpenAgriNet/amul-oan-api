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
