"""Deterministic micro-loan eligibility evaluation + code issuance.

This is intentionally NOT driven by the LLM. The agent tool calls
``evaluate_and_issue`` with the caller's phone and resolved farmer accounts; all
gating, milk arithmetic, code generation, persistence and SMS happen here so the
model can never fabricate an approval, a code, or a milk figure.

Flow (each step gated by an env toggle; a disabled check is BYPASSED so product
can test end-to-end without real Amul submissions / bank-list rows):

  0. phone required                       -> NO_PHONE
  1. existing active code (and not LOAN_ALLOW_MULTIPLE_CODES) -> ELIGIBLE, re-share
     that same code (optionally re-send SMS if LOAN_RESEND_SMS_ON_REQUEST)
  2. bank eligibility list (by phone)      -> NOT_IN_BANK_LIST   [LOAN_CHECK_BANK_LIST_ENABLED]
  3. last-N-days milk >= threshold         -> MILK_BELOW_THRESHOLD [LOAN_CHECK_MILK_ENABLED]
  4. otherwise                             -> ELIGIBLE (issue a new code, store, SMS)

The loan is framed as ALREADY SANCTIONED for eligible members; asking for the loan
(or the code) again returns the same code rather than a rejection.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from sqlalchemy import select

from agents.deps import FarmerAccount
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
from agents.tools.onex_sms import send_loan_approval_sms
from app.config import settings
from app.core.loan_db import get_loan_session, loan_db_configured
from app.models.loan import LoanCode, LoanEligibilityRow
from app.models.milk_collection import FarmerMilkCollectionRequestModel
from helpers.utils import get_logger

logger = get_logger(__name__)

_DATE_FORMAT = "%Y-%m-%d"

# Outcome codes — the tool maps these to the script-aligned user message.
NO_PHONE = "no_phone"
ALREADY_AVAILED = "already_availed"  # legacy, no longer returned
ALREADY_ISSUED = "already_issued"    # loan already disbursed (a redeemed code exists)
NOT_IN_BANK_LIST = "not_in_bank_list"
MILK_BELOW_THRESHOLD = "milk_below_threshold"
ELIGIBLE_OFFER = "eligible_offer"   # eligible, loan offered, NOT yet issued (awaiting confirmation)
ELIGIBLE = "eligible"               # confirmed/issued (or existing code re-shared)
DISABLED = "disabled"
ERROR = "error"


@dataclass
class LoanResult:
    outcome: str
    phone: Optional[str] = None
    code: Optional[str] = None
    loan_amount: Optional[float] = None
    milk_amount_month: Optional[float] = None
    milk_threshold: Optional[float] = None
    farmer_name: Optional[str] = None
    sms_status: Optional[str] = None
    reshared: bool = False  # True when an existing code was returned (not newly minted)
    error: Optional[str] = None
    checks_applied: dict = field(default_factory=dict)


def _checks_applied() -> dict:
    return {
        "bank_list": settings.loan_check_bank_list_enabled,
        "milk": settings.loan_check_milk_enabled,
        "allow_multiple": settings.loan_allow_multiple_codes,
        "resend_sms": settings.loan_resend_sms_on_request,
    }


async def _maybe_resend_sms(session, record, mobile: str) -> Optional[str]:
    """Re-send the approval SMS for an existing code when the resend flag is on.

    Returns the resulting sms_status. When the resend flag is off, the record is
    left untouched and its current status is returned. This is what powers "every
    time the farmer asks for their OTP, send the SMS again" — behind a config flag.
    """
    if not settings.loan_resend_sms_on_request:
        return record.sms_status
    if not settings.loan_sms_enabled:
        record.sms_status = "dry_run"
        await session.commit()
        logger.info("Loan SMS resend dry-run (LOAN_SMS_ENABLED=false) code=%s to=%s", record.code, mobile)
        return "dry_run"
    sms = await send_loan_approval_sms(
        mobile, record.farmer_name or "", float(record.loan_amount or settings.loan_max_amount), record.code
    )
    record.sms_status = sms.status
    record.sms_message_id = sms.message_id
    record.sms_error = sms.error
    await session.commit()
    logger.info("Loan SMS re-sent code=%s to=%s status=%s", record.code, mobile, sms.status)
    return sms.status


async def _active_code_for_phone(session, phone: str) -> Optional[LoanCode]:
    """Return an existing non-expired active code for this phone, if any."""
    now = datetime.now(timezone.utc)
    stmt = (
        select(LoanCode)
        .where(LoanCode.phone == phone, LoanCode.status == "active")
        .order_by(LoanCode.issued_at.desc())
    )
    for row in (await session.execute(stmt)).scalars():
        if row.expires_at is None or row.expires_at > now:
            return row
    return None


async def _redeemed_code_for_phone(session, phone: str) -> Optional[LoanCode]:
    """Return the most recently redeemed (issued/disbursed) code for this phone, if
    any. Used to block a second loan once one has been issued at the bank."""
    stmt = (
        select(LoanCode)
        .where(LoanCode.phone == phone, LoanCode.status == "redeemed")
        .order_by(LoanCode.redeemed_at.desc().nullslast())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def _eligibility_row_for_phone(session, phone: str) -> Optional[LoanEligibilityRow]:
    stmt = (
        select(LoanEligibilityRow)
        .where(LoanEligibilityRow.phone == phone, LoanEligibilityRow.is_active.is_(True))
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def _compute_last_month_milk(accounts: Sequence[FarmerAccount]) -> Optional[float]:
    """Sum milk-collection amount over the lookback window across all accounts.

    Returns the total (float), or None when the milk API could not be reached for
    any account (so the caller can distinguish "genuinely below threshold" from
    "couldn't check"). A reachable-but-empty result totals 0.0.
    """
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN not set; cannot compute milk total")
        return None

    today = datetime.now(timezone.utc).date()
    fromdate = (today - timedelta(days=settings.loan_milk_lookback_days)).strftime(_DATE_FORMAT)
    todate = today.strftime(_DATE_FORMAT)

    total = 0.0
    any_ok = False
    for acct in accounts:
        if not (acct.union_code and acct.society_code and acct.farmer_code):
            continue
        request = FarmerMilkCollectionRequestModel(
            unionCode=acct.union_code,
            societyCode=acct.society_code,
            farmerCode=acct.farmer_code,
            fromdate=fromdate,
            todate=todate,
        )
        resp = await get_farmer_milk_collection_details_api(request, token)
        if resp is None:
            continue
        any_ok = True
        for rec in resp.milk:
            if rec.amount is not None:
                total += float(rec.amount)

    if not any_ok:
        return None
    logger.info("Milk total over %sd across %s account(s) = %.2f",
                settings.loan_milk_lookback_days, len(accounts), total)
    return total


async def _generate_unique_code(session) -> str:
    """Generate a numeric code unique against loan_codes (retry on collision)."""
    length = max(4, settings.loan_code_length)
    lo, hi = 10 ** (length - 1), 10 ** length - 1
    for _ in range(20):
        code = str(secrets.randbelow(hi - lo + 1) + lo)
        exists = (
            await session.execute(select(LoanCode.id).where(LoanCode.code == code))
        ).first()
        if not exists:
            return code
    raise RuntimeError("could not generate a unique loan code after 20 attempts")


def _resolve_name(
    passed_name: Optional[str],
    accounts: Sequence[FarmerAccount],
    elig_row: Optional[LoanEligibilityRow],
) -> Optional[str]:
    if passed_name and passed_name.strip():
        return passed_name.strip()
    for acct in accounts:
        if acct.farmer_name and acct.farmer_name.strip():
            return acct.farmer_name.strip()
    if elig_row and elig_row.sabhsad_name:
        return elig_row.sabhsad_name.strip()
    return None


async def evaluate_and_issue(
    *,
    phone: Optional[str],
    accounts: Sequence[FarmerAccount],
    farmer_name: Optional[str] = None,
    channel: str,
    session_id: Optional[str] = None,
    confirm: bool = False,
) -> LoanResult:
    checks = _checks_applied()

    if not settings.loan_feature_enabled:
        return LoanResult(outcome=DISABLED, checks_applied=checks)

    mobile = normalize_phone_to_mobile(phone or "")
    if not mobile:
        return LoanResult(outcome=NO_PHONE, checks_applied=checks)

    if not loan_db_configured():
        logger.error("Loan feature enabled but LOAN_DB_URL not configured")
        return LoanResult(outcome=ERROR, phone=mobile, error="db_not_configured", checks_applied=checks)

    accounts = list(accounts or [])
    amount = float(settings.loan_max_amount)
    threshold = float(settings.loan_milk_threshold)

    try:
        async with get_loan_session() as session:
            # Existing codes (only consulted when a single loan per farmer is enforced).
            existing = None
            issued = None
            if not settings.loan_allow_multiple_codes:
                existing = await _active_code_for_phone(session, mobile)
                if existing is None:
                    issued = await _redeemed_code_for_phone(session, mobile)

            # A loan already ISSUED/disbursed at the bank blocks a new one outright.
            if issued is not None:
                logger.info("Loan already issued for %s (redeemed code %s)", mobile, issued.code)
                return LoanResult(
                    outcome=ALREADY_ISSUED, phone=mobile, code=issued.code,
                    loan_amount=float(issued.loan_amount) if issued.loan_amount else amount,
                    farmer_name=issued.farmer_name, checks_applied=checks,
                )

            # Eligibility. An existing active code means the farmer was already found
            # eligible, so we do NOT re-run the bank-list / milk checks for it.
            elig_row: Optional[LoanEligibilityRow] = None
            milk_total: Optional[float] = None
            if existing is None:
                if settings.loan_check_bank_list_enabled:
                    elig_row = await _eligibility_row_for_phone(session, mobile)
                    if elig_row is None:
                        logger.info("Phone %s not in bank eligibility list", mobile)
                        return LoanResult(outcome=NOT_IN_BANK_LIST, phone=mobile, checks_applied=checks)
                else:
                    elig_row = await _eligibility_row_for_phone(session, mobile)
                if settings.loan_check_milk_enabled:
                    milk_total = await _compute_last_month_milk(accounts)
                    if milk_total is None:
                        return LoanResult(outcome=ERROR, phone=mobile, error="milk_lookup_failed", checks_applied=checks)
                    if milk_total < threshold:
                        logger.info("Milk %.2f below threshold %.2f for %s", milk_total, threshold, mobile)
                        return LoanResult(
                            outcome=MILK_BELOW_THRESHOLD, phone=mobile,
                            milk_amount_month=milk_total, milk_threshold=threshold, checks_applied=checks,
                        )

            name = existing.farmer_name if existing is not None else _resolve_name(farmer_name, accounts, elig_row)

            # OFFER step — eligible but NOT yet confirmed: reveal no code, send no SMS.
            if not confirm:
                logger.info("Loan OFFER for %s (awaiting confirmation, has_existing=%s)", mobile, existing is not None)
                return LoanResult(
                    outcome=ELIGIBLE_OFFER, phone=mobile,
                    loan_amount=float(existing.loan_amount) if existing and existing.loan_amount else amount,
                    milk_amount_month=milk_total, milk_threshold=threshold,
                    farmer_name=name, checks_applied=checks,
                )

            # CONFIRMED. Re-share an existing code (+ optional resend), or mint a new one.
            if existing is not None:
                resent = await _maybe_resend_sms(session, existing, mobile)
                logger.info("Loan code re-shared (confirmed) for %s (code %s, sms=%s)", mobile, existing.code, resent)
                return LoanResult(
                    outcome=ELIGIBLE, phone=mobile, code=existing.code,
                    loan_amount=float(existing.loan_amount) if existing.loan_amount else amount,
                    farmer_name=existing.farmer_name, sms_status=resent, reshared=True, checks_applied=checks,
                )

            # Confirmed, no existing code — issue, store, send.
            code = await _generate_unique_code(session)
            expires_at = None
            if settings.loan_code_expiry_days and settings.loan_code_expiry_days > 0:
                expires_at = datetime.now(timezone.utc) + timedelta(days=settings.loan_code_expiry_days)

            first = accounts[0] if accounts else None
            record = LoanCode(
                code=code,
                phone=mobile,
                farmer_name=name,
                farmer_code=(elig_row.farmer_code if elig_row else None) or (first.farmer_code if first else None),
                mandali_name=(elig_row.mandali_name if elig_row else None) or (first.society_name if first else None),
                union_code=first.union_code if first else None,
                society_code=first.society_code if first else None,
                loan_amount=amount,
                milk_amount_month=milk_total,
                milk_threshold=threshold,
                channel=channel,
                status="active",
                checks_applied=checks,
                session_id=session_id,
                expires_at=expires_at,
            )
            session.add(record)
            await session.flush()  # persist before SMS so a code is never lost

            # Send SMS (or dry-run). Record the outcome; never fail issuance on SMS.
            if settings.loan_sms_enabled:
                sms = await send_loan_approval_sms(mobile, name or "", amount, code)
                record.sms_status = sms.status
                record.sms_message_id = sms.message_id
                record.sms_error = sms.error
            else:
                record.sms_status = "dry_run"
                logger.info("Loan SMS dry-run (LOAN_SMS_ENABLED=false) code=%s to=%s", code, mobile)

            await session.commit()
            logger.info("Loan code issued code=%s phone=%s amount=%s sms=%s channel=%s",
                        code, mobile, amount, record.sms_status, channel)
            return LoanResult(
                outcome=ELIGIBLE, phone=mobile, code=code, loan_amount=amount,
                milk_amount_month=milk_total, milk_threshold=threshold,
                farmer_name=name, sms_status=record.sms_status, checks_applied=checks,
            )
    except Exception as e:
        logger.error("Loan evaluate_and_issue failed for %s: %s", mobile, e)
        return LoanResult(outcome=ERROR, phone=mobile, error=str(e), checks_applied=checks)
