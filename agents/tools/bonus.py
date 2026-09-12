"""
Tool for fetching farmer bonus amount records (GetFarmerBonusAmount).
"""
import asyncio
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.services.beckn_amul import authenticated_accounts
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_animal_backends import get_farmer_bonus_amount_api
from app.config import get_config_value
from app.models.bonus import (
    FarmerBonusAmountRecordModel,
    FarmerBonusAmountRequestModel,
)
from helpers.utils import get_logger

logger = get_logger(__name__)


async def prepare_get_farmer_bonus_amount(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Hide get_farmer_bonus_amount unless a farmer is resolved.

    The tool needs union/society/farmer codes that only exist in the farmer
    context (populated when a farmer record is resolved). With no farmer context
    the LLM has no codes and would otherwise hallucinate placeholders that
    reach the live backend. farmer_unions is non-empty exactly when a farmer
    was resolved, so we gate on it (mirrors milk collection / union schemes).
    """
    farmer_unions = [
        cleaned
        for cleaned in ((u or "").strip().lower() for u in (ctx.deps.farmer_unions or []))
        if cleaned
    ]
    if farmer_unions and (getattr(ctx.deps, "mobile", None) or "").strip():
        return tool_def
    logger.info(
        "Hiding get_farmer_bonus_amount tool because farmer_unions is empty "
        "or authenticated mobile is missing (no resolved farmer context)"
    )
    return None


def _escape_markdown_cell(value) -> str:
    """Escape markdown table delimiter characters in cell content.

    None-safe: lenient bonus records may omit fields — render None as '-'.
    """
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _format_number(value, decimals: int = 2) -> str:
    """Format numeric values for compact table display."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _format_period_date(value: str | None) -> str:
    """Render API ISO datetimes as YYYY-MM-DD when possible."""
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback: date portion before 'T' if present
    if "T" in text:
        return text.split("T", 1)[0] or text
    return text


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Create a markdown table with fixed headers and row ordering."""
    header_line = f"| {' | '.join(headers)} |"
    separator_line = f"| {' | '.join(['---'] * len(headers))} |"
    row_lines = [
        f"| {' | '.join(_escape_markdown_cell(cell) for cell in row)} |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_bonus_markdown(records: list[FarmerBonusAmountRecordModel]) -> str:
    """Format bonus records as a deterministic markdown table."""
    sections: list[str] = ["### Bonus Amount"]
    if not records:
        sections.append("No bonus records found.")
        return "\n".join(sections)

    rows = [
        [
            f"{_format_period_date(record.from_date)} - {_format_period_date(record.to_date)}",
            record.society_name or record.society_code,
            record.farmer_name or record.farmer_code,
            _format_number(record.bonus_amount, 2),
        ]
        for record in records
    ]
    sections.append(
        _build_markdown_table(
            ["Period", "Society", "Farmer", "Bonus Amount"],
            rows,
        )
    )
    return "\n".join(sections)


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
        str: Deterministic markdown table of bonus records, or a clear failure message.
    """
    logger.info("Farmer bonus amount tool invoked")

    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        logger.info("Farmer bonus amount tool refused: no authenticated mobile")
        return (
            "Bonus amount lookup failed.\n\n"
            "Your signed-in farmer profile is not available, so bonus amount "
            "details can't be fetched."
        )

    try:
        farmers = await get_farmer_data_by_mobile(mobile)
    except Exception as exc:
        logger.warning("Farmer profile lookup for bonus amount failed: %s", exc)
        return (
            "Bonus amount lookup failed.\n\n"
            "Unable to fetch bonus amount details at the moment."
        )

    accounts = authenticated_accounts(farmers or [])
    if not accounts:
        return (
            "Bonus amount lookup failed.\n\n"
            "No union, society, and farmer account was found for your signed-in mobile."
        )

    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Bonus amount lookup failed.\n\nProvider access is not configured."

    outcomes = await asyncio.gather(
        *(
            get_farmer_bonus_amount_api(
                FarmerBonusAmountRequestModel(
                    unionCode=account.union_code,
                    societyCode=account.society_code,
                    farmerCode=account.farmer_code,
                ),
                token,
            )
            for account in accounts
        ),
        return_exceptions=True,
    )

    # Empty list [] is a successful "no records" response; None / exceptions are failures.
    successes: list[list[FarmerBonusAmountRecordModel]] = [
        response
        for response in outcomes
        if response is not None and not isinstance(response, BaseException)
    ]
    if not successes:
        logger.info(
            "Farmer bonus amount lookup failed for all authenticated accounts "
            "(count=%s)",
            len(accounts),
        )
        return (
            "Bonus amount lookup failed.\n\n"
            "Unable to fetch bonus amount details at the moment. "
            "Bonus lookup is only available for unions whose data source is AMCS; "
            "if your union uses a different source, this may not be supported yet."
        )

    records = [record for result in successes for record in result]
    if not records:
        logger.info(
            "Farmer bonus amount lookup returned no records accounts=%s",
            len(accounts),
        )
        return (
            "Bonus amount lookup completed.\n\n"
            "No bonus records were found for your signed-in farmer account(s)."
        )

    formatted = _format_bonus_markdown(records)
    logger.info(
        "Farmer bonus amount lookup succeeded accounts=%s records=%s",
        len(accounts),
        len(records),
    )
    return f"Farmer bonus amount details fetched successfully:\n\n{formatted}"
