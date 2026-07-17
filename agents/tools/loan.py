"""Micro-loan eligibility tool for the agent (chat + voice).

Thin wrapper over the deterministic ``loan_eligibility.evaluate_and_issue``
service. The tool passes NO loan decisioning to the LLM: the caller's phone and
farmer accounts are read from ``ctx.deps`` (never from model-supplied args), and
the service decides eligibility, generates + stores the code, and sends the SMS.
The tool only turns the structured outcome into a script-aligned message.
"""
from __future__ import annotations

from typing import Optional

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.services import loan_eligibility as le
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

# Deployed-surface channel for this service. voice-oan-api = "voice";
# amul-oan-api (chat) overrides this to "chat".
LOAN_CHANNEL = "chat"


async def prepare_check_loan_eligibility(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Expose the loan tool whenever the feature is enabled.

    Even without a resolved caller mobile, the tool runs and returns a clear
    "no profile - visit your local cooperative bank" message, so the model does not
    improvise a request for a mobile number it cannot act on."""
    if not settings.loan_feature_enabled:
        return None
    # Expose whenever the feature is on. If the caller's profile/mobile is not
    # resolved, the tool still runs and returns a clear "no profile - visit your
    # local cooperative bank" message, rather than the model improvising a request
    # for a mobile number (which we cannot act on).
    return tool_def


def _message_for(result: "le.LoanResult") -> str:
    """Map an outcome to a script-aligned instruction for the agent to convey.

    Returns English content; the normal translation path renders it in the
    caller's language. The authoritative Gujarati approval text is the SMS."""
    amt = int(result.loan_amount) if result.loan_amount else int(settings.loan_max_amount)
    thr = int(result.milk_threshold) if result.milk_threshold else int(settings.loan_milk_threshold)

    if result.outcome == le.ELIGIBLE_OFFER:
        return (
            f"ELIGIBLE — OFFER ONLY (do NOT issue or mention a code yet, do NOT say it is approved). "
            f"Tell the farmer they are eligible for a KDCC Bank micro loan of ₹{amt:,}, which "
            f"carries {int(settings.loan_interest_rate_pct)}% annual interest that is waived if the loan is repaid "
            f"regularly. Ask whether they would like to avail "
            f"this loan. If the farmer agrees, call check_loan_eligibility again with confirmed=true. "
            f"If the farmer declines, do not call again — close the conversation politely."
        )
    if result.outcome == le.ELIGIBLE:
        return (
            f"APPROVED. Confirm to the farmer that their KDCC Bank micro loan of up to ₹{amt:,} has been "
            f"approved/processed. Their loan reference code is {result.code}, and it has been sent to "
            f"their registered mobile number by SMS. They should visit their KDCC cooperative bank branch "
            f"and present this code, and must carry only two documents: their Aadhaar card and their "
            f"proof of milk cooperative society membership. Also briefly "
            f"tell them this micro-loan facility is currently in a pilot phase."
        )
    if result.outcome == le.ALREADY_ISSUED:
        return (
            "ALREADY ISSUED. Our records show this farmer's micro loan has already been issued "
            "(disbursed) at the bank. Do NOT issue a new code. Ask them to contact their cooperative "
            "bank branch for any questions about their existing loan."
        )
    if result.outcome in (le.NOT_IN_BANK_LIST, le.MILK_BELOW_THRESHOLD):
        return (
            "NOT ELIGIBLE. Tell the farmer they are not eligible for the micro loan on this platform. "
            "If they ask where to go for a loan, direct them to their NEAREST cooperative bank branch — "
            "do NOT name KDCC Bank or any specific bank, and do NOT refer them to the KDCC micro-loan "
            "facility. Do NOT disclose the specific reason for ineligibility. Also briefly tell them "
            "this micro-loan facility is currently in a pilot phase."
        )
    if result.outcome == le.NO_PHONE:
        return (
            "NO PROFILE. Tell the farmer, warmly: \"I don't have your profile information, so I can't "
            "process a micro loan for you on this platform. Please visit your local cooperative bank "
            "branch for assistance.\" Do NOT ask them to provide or type a mobile number — eligibility "
            "uses only their registered session profile."
        )
    # DISABLED / ERROR
    return (
        "The micro-loan eligibility service is temporarily unavailable. Apologize briefly and ask "
        "the farmer to try again later or contact their cooperative bank."
    )


async def _resolve_accounts(ctx: RunContext[FarmerContext]):
    """Accounts from context; for the chat surface (which doesn't pre-collect
    them) fetch cache-first when a milk check will need union/society codes."""
    accounts = list(ctx.deps.farmer_accounts or [])
    if accounts or not ctx.deps.mobile or not settings.loan_check_milk_enabled:
        return accounts
    try:
        from agents.services.farmer_cache import get_or_fetch_farmer_data
        from app.services.voice import _collect_farmer_accounts
        envelope = await get_or_fetch_farmer_data(ctx.deps.mobile)
        return _collect_farmer_accounts(envelope)
    except Exception as e:  # non-fatal: milk check will simply report couldn't-check
        logger.warning("Could not resolve farmer accounts for loan milk check: %s", e)
        return accounts


async def check_loan_eligibility(ctx: RunContext[FarmerContext], confirmed: bool = False) -> str:
    """
    Check a farmer's micro-loan eligibility and — only after the farmer CONFIRMS —
    issue the approval code and send it by SMS. Two-step flow:

    1. Call with confirmed=false (the default) FIRST. If eligible, this returns an
       OFFER for you to convey; it does NOT issue a code or send an SMS yet.
    2. After the farmer explicitly agrees to take the loan, call AGAIN with
       confirmed=true — only then is the code issued and the SMS sent.
    3. If the farmer declines, do not call again; close politely.

    Use when the farmer asks for a loan / micro loan / credit. The farmer's registered
    mobile is read from the session (you never pass it); if it is not available the
    tool returns a 'no profile - visit your local cooperative bank' message (it does
    NOT ask the farmer for a mobile number). Do not pass any codes or amounts.

    Args:
        confirmed: Set true ONLY after the farmer has explicitly agreed to avail the
            loan (their yes to the offer). Leave false for the initial eligibility/offer.
    """
    accounts = await _resolve_accounts(ctx)
    name: Optional[str] = None
    for acct in accounts:
        if acct.farmer_name:
            name = acct.farmer_name
            break

    result = await le.evaluate_and_issue(
        phone=ctx.deps.mobile,
        accounts=accounts,
        farmer_name=name,
        channel=LOAN_CHANNEL,
        session_id=ctx.deps.session_id,
        confirm=confirmed,
    )
    logger.info("check_loan_eligibility outcome=%s phone=%s code=%s sms=%s",
                result.outcome, result.phone, result.code, result.sms_status)
    return _message_for(result)
