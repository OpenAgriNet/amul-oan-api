"""Tool for fetching farmer bonus amount records (GetFarmerBonusAmount)."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.voice.tools.farmer_animal_backends import get_farmer_bonus_amount_api
from app.voice.models.bonus import (
    FarmerBonusAmountRecordModel,
    FarmerBonusAmountRequestModel,
)
from helpers.utils import get_logger

logger = get_logger(__name__)


async def prepare_get_farmer_bonus_amount(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Hide get_farmer_bonus_amount unless the caller is authenticated.

    Account codes are resolved server-side from FarmerContext.farmer_accounts
    during execution. Do not gate on optional union display names — a signed-in
    farmer with valid codes must still see the tool when unionName is absent.
    """
    if (getattr(ctx.deps, "mobile", None) or "").strip():
        return tool_def
    logger.info(
        "Hiding get_farmer_bonus_amount tool because authenticated mobile is missing"
    )
    return None


def _num(value) -> str:
    """Render a numeric field compactly, dropping a trailing .0 (e.g. 2.0 -> '2')."""
    if value is None:
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_period_date(value: str | None) -> str:
    """Render API ISO datetimes as YYYY-MM-DD when possible."""
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if "T" in text:
        return text.split("T", 1)[0] or text
    return text


def _format_bonus_summary(records: list[FarmerBonusAmountRecordModel]) -> str:
    """Deterministic plain-text summary for the voice agent.

    Voice runs on a small OSS model and speaks aloud, so use a flat labelled
    list instead of a markdown table.
    """
    if not records:
        return "No bonus records found."

    lines = [f"Bonus amount records ({len(records)}):"]
    for i, record in enumerate(records, 1):
        period = (
            f"{_format_period_date(record.from_date)} to "
            f"{_format_period_date(record.to_date)}"
        )
        society = record.society_name or record.society_code or "unknown society"
        farmer = record.farmer_name or record.farmer_code or "unknown farmer"
        lines.append(
            f"  {i}. Period {period}: society {society}, farmer {farmer}, "
            f"bonus amount {_num(record.bonus_amount)} rupees."
        )
    return "\n".join(lines)


def _temporary_failure_message() -> str:
    return (
        "Bonus amount lookup failed. "
        "Unable to fetch bonus amount details at the moment. "
        "Bonus lookup is only available for unions whose data source is AMCS; "
        "if your union uses a different source, this may not be supported yet."
    )


async def get_farmer_bonus_amount(ctx: RunContext[FarmerContext]) -> str:
    """
    Fetch bonus amount(s) credited to every account owned by the signed-in farmer.

    Use when the farmer asks for their personal bonus / બોનસ amount (e.g.
    "what is my bonus amount?", "મારું બોનસ કેટલું છે?"). Identity and
    union/society/farmer codes come only from authenticated context — never
    ask the farmer for those codes and never invent them.

    Args:
        ctx: Authenticated farmer context supplied by the agent runtime.

    Returns:
        str: Deterministic plain-text bonus records, or a clear failure message.
    """
    logger.info("Farmer bonus amount tool invoked")

    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        logger.info("Farmer bonus amount tool refused: no authenticated mobile")
        return (
            "Bonus amount lookup failed. "
            "Your signed-in farmer profile is not available, so bonus amount "
            "details can't be fetched."
        )

    accounts = list(ctx.deps.farmer_accounts) if ctx.deps and ctx.deps.farmer_accounts else []
    if not accounts:
        return (
            "Bonus amount lookup failed. "
            "No union, society, and farmer account was found for your signed-in mobile."
        )

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Bonus amount lookup failed. Service is not configured."

    outcomes = await asyncio.gather(
        *(
            get_farmer_bonus_amount_api(
                FarmerBonusAmountRequestModel(
                    union_code=account.union_code or "",
                    society_code=account.society_code or "",
                    farmer_code=account.farmer_code or "",
                ),
                token,
            )
            for account in accounts
        ),
        return_exceptions=True,
    )

    # Empty list [] is a successful "no records" response; None / exceptions are
    # account-level failures. Preserve partial-failure state to avoid reporting
    # incomplete financial data as complete/no-records.
    successes: list[list[FarmerBonusAmountRecordModel]] = []
    failed_accounts = 0
    for outcome in outcomes:
        if outcome is None or isinstance(outcome, BaseException):
            failed_accounts += 1
            continue
        successes.append(outcome)

    if not successes:
        logger.info(
            "Farmer bonus amount lookup failed for all authenticated accounts "
            "(count=%s)",
            len(accounts),
        )
        return _temporary_failure_message()

    if failed_accounts:
        logger.warning(
            "Farmer bonus amount lookup incomplete: failed_accounts=%s total_accounts=%s",
            failed_accounts,
            len(accounts),
        )
        return _temporary_failure_message()

    records = [record for result in successes for record in result]
    if not records:
        logger.info(
            "Farmer bonus amount lookup returned no records accounts=%s",
            len(accounts),
        )
        return (
            "Bonus amount lookup completed. "
            "No bonus records were found for your signed-in farmer account(s)."
        )

    formatted = _format_bonus_summary(records)
    logger.info(
        "Farmer bonus amount lookup succeeded accounts=%s records=%s",
        len(accounts),
        len(records),
    )
    return f"Bonus amount details fetched successfully:\n\n{formatted}"
