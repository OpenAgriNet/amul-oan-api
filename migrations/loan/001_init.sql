-- Micro-loan eligibility feature — initial schema.
-- Idempotent: safe to run against the pre-existing per-env Postgres.
-- Apply once per environment (dev / prod):
--   psql "$LOAN_DB_DSN" -f migrations/loan/001_init.sql

-- The cooperative-bank eligibility list (loaded from the SABHSAD export).
CREATE TABLE IF NOT EXISTS loan_eligibility_list (
    id                  BIGSERIAL PRIMARY KEY,
    phone               VARCHAR(15)  NOT NULL,     -- normalized last-10-digits
    farmer_code         VARCHAR(64),
    mandali_name        VARCHAR(128),
    sabhsad_name        VARCHAR(256),
    ac_no               VARCHAR(32),
    milk_payment_amount NUMERIC(12,2),             -- snapshot from sheet (informational)
    source_batch        VARCHAR(64),
    is_active           BOOLEAN      NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_loan_eligibility_phone_farmer UNIQUE (phone, farmer_code)
);
CREATE INDEX IF NOT EXISTS ix_loan_eligibility_phone   ON loan_eligibility_list (phone);
CREATE INDEX IF NOT EXISTS ix_loan_eligibility_batch   ON loan_eligibility_list (source_batch);

-- Every issued approval code (authoritative store for bank-side verification).
CREATE TABLE IF NOT EXISTS loan_codes (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code              VARCHAR(12)  NOT NULL UNIQUE,
    phone             VARCHAR(15)  NOT NULL,
    farmer_name       VARCHAR(256),
    farmer_code       VARCHAR(64),
    mandali_name      VARCHAR(128),
    union_code        VARCHAR(64),
    society_code      VARCHAR(64),
    loan_amount       NUMERIC(12,2) NOT NULL,
    milk_amount_month NUMERIC(12,2),
    milk_threshold    NUMERIC(12,2),
    channel           VARCHAR(16),                 -- 'chat' | 'voice'
    status            VARCHAR(16)  NOT NULL DEFAULT 'active',  -- active|redeemed|expired|cancelled
    sms_status        VARCHAR(16),                 -- sent|failed|skipped|dry_run
    sms_message_id    VARCHAR(128),
    sms_error         TEXT,
    checks_applied    JSONB,
    session_id        VARCHAR(128),
    issued_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    redeemed_at       TIMESTAMPTZ,
    redeemed_by       VARCHAR(128),
    expires_at        TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_loan_codes_phone  ON loan_codes (phone);
CREATE INDEX IF NOT EXISTS ix_loan_codes_status ON loan_codes (status);

-- gen_random_uuid() requires pgcrypto (bundled with modern Postgres; enable if absent).
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;
