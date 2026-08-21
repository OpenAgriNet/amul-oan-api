-- Micro-loan: per-farmer eligible amount (AMUL-51).
-- Additive and nullable, so it is safe to apply BEFORE the app rollout and
-- leaves the running version untouched. Apply once per environment (dev / prod):
--   psql "$LOAN_DB_DSN" -f migrations/loan/002_max_loan_amount.sql
--
-- MUST be applied before deploying the app version that reads it: the ORM maps
-- every column explicitly, so the new app SELECTs max_loan_amount and errors on
-- a database that does not have it yet.
--
-- NULL = the bank did not set an amount for this member; the app then falls back
-- to LOAN_MAX_AMOUNT. Existing rows keep NULL, so behaviour is unchanged until
-- the bank uploads a sheet carrying the column.

ALTER TABLE loan_eligibility_list
    ADD COLUMN IF NOT EXISTS max_loan_amount NUMERIC(12,2);
