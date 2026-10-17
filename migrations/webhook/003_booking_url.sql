-- Persist the optional partner-provided AI booking URL on existing deployments.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS booking_url TEXT;
