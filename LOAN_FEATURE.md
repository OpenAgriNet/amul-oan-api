# Micro-loan eligibility feature

Farmer asks for a loan via **chat** (`amul-oan-api`) or **voice** (`voice-oan-api`);
the bot verifies eligibility, issues a unique approval code, stores it in Postgres,
and sends it by SMS. The stored codes are the source of truth for the (future)
bank-side verification portal.

Everything is **deterministic** — the LLM never decides eligibility, the amount,
or the code. It only relays the tool's outcome message to the farmer.

## Flow

Phone number is **mandatory** (no phone → the tool asks for it). Then, in order,
each step gated by an env toggle:

| Step | Check | Env toggle | Fail outcome |
|------|-------|-----------|--------------|
| 1 | Already has an active code | `LOAN_CHECK_ALREADY_AVAILED_ENABLED` | Already availed |
| 2 | Phone in bank eligibility list | `LOAN_CHECK_BANK_LIST_ENABLED` | Not in bank list |
| 3 | Last-N-days milk ≥ threshold (milk API) | `LOAN_CHECK_MILK_ENABLED` | Milk below ₹3,000 |
| 4 | otherwise → **eligible** | — | issue code + store + SMS |

**A disabled check is bypassed (treated as pass).** This is how product tests the
full flow without real Amul submissions or bank-list rows — e.g. set
`LOAN_CHECK_BANK_LIST_ENABLED=false` and `LOAN_CHECK_MILK_ENABLED=false`, keep
`LOAN_SMS_ENABLED=false` (dry-run), and any caller with a phone gets a code
issued + stored (never actually sent).

## Files (mirrored in both `amul-oan-api` and `voice-oan-api`)

- `app/models/loan.py` — ORM: `loan_eligibility_list`, `loan_codes`
- `app/core/loan_db.py` — lazy async engine/session (only built when `LOAN_DB_URL` set)
- `migrations/loan/001_init.sql` — idempotent DDL
- `migrations/loan/002_max_loan_amount.sql` — per-farmer `max_loan_amount` (AMUL-51)
- `agents/tools/onex_sms.py` — Onex-Aura SMS client (DLT template)
- `agents/services/loan_eligibility.py` — `evaluate_and_issue(...)` (the state machine)
- `agents/tools/loan.py` — `check_loan_eligibility` tool (`LOAN_CHANNEL` = chat/voice)
- Registered: chat `TOOLS`, voice `BASE_TOOLS` (hidden unless feature on + phone resolved)
- `../scripts/load_sabhsad.py` — one-off SABHSAD xlsx → `loan_eligibility_list` loader
- `tests/test_loan_eligibility.py`

## Deploy per environment (dev, prod)

1. **DB**: apply the DDL to that env's Postgres
   `psql "$DSN" -f migrations/loan/001_init.sql`  (needs `pgcrypto` for `gen_random_uuid`)
   `psql "$DSN" -f migrations/loan/002_max_loan_amount.sql`
   **Apply 002 BEFORE rolling out the app** — the ORM maps every column explicitly, so a
   build that knows `max_loan_amount` errors against a database that does not have it
   (the error is caught and surfaces as "service temporarily unavailable", not a crash).
   Applying it early is free: the column is nullable and the running build ignores it.
2. **Load the eligibility list**
   `python scripts/load_sabhsad.py --xlsx "SABHSAD FILE - TOP 10.xlsx" --dsn "postgresql+pg8000://USER:PASS@HOST:5432/DB" --source-batch SABHSAD_TOP10_2026-07-04`
3. **Set env** (secrets in env only — see `example.env` / `example.loan.env`):
   - `LOAN_FEATURE_ENABLED=true`
   - `LOAN_DB_URL=postgresql+asyncpg://USER:PASS@HOST:5432/DB`
   - `ONEX_SMS_KEY`, `ONEX_SMS_ENTITY_ID`, `ONEX_SMS_TEMPLATE_ID` (DLT secrets)
   - `ONEX_SMS_FROM=AMULHO`
   - Testing: `LOAN_CHECK_*_ENABLED=false` as needed, `LOAN_SMS_ENABLED=false`
   - Go-live: all checks `true`, `LOAN_SMS_ENABLED=true`
4. **New deps**: `asyncpg`, `greenlet` (added to `requirements.txt`).

## Notes / follow-ups

- `LOAN_INTEREST_RATE_PCT=7` — annual interest rate quoted in the offer (waived on regular
  repayment). Change the rate here; the tool message and system prompts both read it.
- `LOAN_MAX_AMOUNT=5000`, `LOAN_MILK_THRESHOLD=3000`, `LOAN_MILK_LOOKBACK_DAYS=30`
  are configurable. **The amount is per farmer** (AMUL-51): the bank sets
  `loan_eligibility_list.max_loan_amount` through the loan-portal upload, and
  `LOAN_MAX_AMOUNT` is only the fallback for a member the bank left blank. The offer, the
  stored `loan_codes.loan_amount`, and the SMS all read the resolved amount, and the system
  prompts quote no figure of their own.
- SABHSAD sample had 60 rows; 56 loaded — 4 skipped for phone < 10 digits (source
  data hygiene). Fix the source or those members won't match.
- The bank-side **verify/redeem portal** is not built yet (storage-only for now).
  `loan_codes` carries `status`, `redeemed_at`, `redeemed_by` for it.
- DLT sender/body: sender `AMULHO` and the Gujarati body template are configurable
  via `ONEX_SMS_FROM` / `ONEX_SMS_BODY_TEMPLATE` if DLT approval differs.
