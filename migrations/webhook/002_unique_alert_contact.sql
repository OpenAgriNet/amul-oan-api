-- Add idempotency key for existing webhook deployments.
-- Keeps the latest row per (alert_id, farmer_contact), then adds a unique constraint.

-- Deduplicate existing data before creating the unique constraint.
WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY alert_id, farmer_contact
            ORDER BY received_at DESC, id DESC
        ) AS rn
    FROM heat_alert_webhook_events
    WHERE farmer_contact IS NOT NULL
),
dupes AS (
    SELECT id
    FROM ranked
    WHERE rn > 1
)
DELETE FROM heat_alert_webhook_events t
USING dupes d
WHERE t.id = d.id;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_heat_alert_events_alert_id_farmer_contact'
          AND conrelid = 'heat_alert_webhook_events'::regclass
    ) THEN
        ALTER TABLE heat_alert_webhook_events
            ADD CONSTRAINT uq_heat_alert_events_alert_id_farmer_contact
            UNIQUE (alert_id, farmer_contact);
    END IF;
END
$$;
