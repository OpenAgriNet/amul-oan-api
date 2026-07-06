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

    if result.outcome == le.ELIGIBLE:
        return (
            f"ELIGIBLE. Tell the farmer that their micro loan of up to ₹{amt:,} is ALREADY SANCTIONED. "
            f"Their loan reference code is {result.code}. Tell them this code has also been sent to "
            f"their registered mobile number by SMS. They should visit their KDCC cooperative bank "
            f"branch and present this code, and must carry their Aadhaar card, their milk "
            f"cooperative society membership certificate, and a one-month milk deposit transaction "
            f"statement. The bank will share the remaining loan details (interest, repayment). Also "
            f"briefly tell them this micro-loan facility is currently in a pilot phase."
        )
    if result.outcome == le.ALREADY_ISSUED:
        return (
            "ALREADY ISSUED. Our records show this farmer's micro loan has already been issued "
            "(disbursed) at the bank. Do NOT issue a new code. Ask them to contact their cooperative "
            "bank branch for any questions about their existing loan."
        )
    if result.outcome in (le.NOT_IN_BANK_LIST, le.MILK_BELOW_THRESHOLD):
        return (
            "NOT ELIGIBLE. Tell the farmer they are not eligible for the micro loan on this platform, "
            "but they can visit their local cooperative bank for assistance. Do NOT disclose the "
            "specific reason for ineligibility. Also briefly tell them this micro-loan facility is "
            "currently in a pilot phase."
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


async def check_loan_eligibility(ctx: RunContext[FarmerContext]) -> str:
    """
    Check whether the farmer is eligible for an Amul micro loan and, if eligible,
    issue an approval code and send it by SMS.

    Use this when the farmer asks for a loan / micro loan / credit. It requires
    the farmer's registered mobile number to already be known from the session; if
    it is not, this tool returns a 'no profile - visit your local cooperative bank' message (it does NOT ask the farmer for a mobile number). Do not pass any codes or
    amounts yourself — this tool determines eligibility and the loan amount and
    generates the code on its own. Convey its returned message to the farmer.
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
    )
    logger.info("check_loan_eligibility outcome=%s phone=%s code=%s sms=%s",
                result.outcome, result.phone, result.code, result.sms_status)
    return _message_for(result)
