-- Persist farmer call status flags on existing webhook deployments.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS called BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS call_booked BOOLEAN NOT NULL DEFAULT FALSE;
