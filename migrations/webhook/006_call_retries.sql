-- Heat-alert call retries: calls placed so far for the alert, how the latest one ended
-- (Completed / Failure from Raya), so an unanswered call is rung again, and when it was
-- last dialled, so dials from all replicas keep Raya's spacing.

ALTER TABLE heat_alert_webhook_events
    ADD COLUMN IF NOT EXISTS call_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS call_outcome VARCHAR(16),
    ADD COLUMN IF NOT EXISTS call_dialled_at TIMESTAMPTZ;

-- Rows from before this migration are settled, so the rollout that brings it rings nobody:
-- a placed call is not followed up, a refused one is not dialled again.
UPDATE heat_alert_webhook_events
SET call_attempts = 1, call_outcome = COALESCE(call_outcome, 'unknown')
WHERE outbound_call_uuid IS NOT NULL AND call_attempts = 0;

UPDATE heat_alert_webhook_events
SET called = TRUE
WHERE called IS FALSE AND call_error IS NOT NULL;
