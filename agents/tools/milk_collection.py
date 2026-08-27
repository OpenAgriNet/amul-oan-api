"""
Tool for fetching farmer milk collection and deduction details.
"""
import asyncio
import os

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.services.beckn_amul import (
    authenticated_accounts,
    fetch_authenticated_farmers,
    fetch_milk_collection,
)
from agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
from agents.tools.farmer import get_farmer_data_by_mobile
from app.config import settings
from app.models.milk_collection import FarmerMilkCollectionRequestModel
from helpers.utils import get_logger

logger = get_logger(__name__)


async def prepare_get_farmer_milk_collection_details(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Hide get_farmer_milk_collection_details unless a farmer is resolved.

    The tool needs union/society/farmer codes that only exist in the farmer
    context (populated when a farmer record is resolved). With no farmer context
    the LLM has no codes and would otherwise hallucinate placeholders (e.g.
    0/0/0) that reach the live backend. farmer_unions is non-empty exactly when a
    farmer was resolved, so we gate on it (mirrors prepare_get_union_scheme_data).
    The LLM won't see the tool in its schema this turn, so it can't call it.
    """
    farmer_unions = [
        cleaned
        for cleaned in ((u or "").strip().lower() for u in (ctx.deps.farmer_unions or []))
        if cleaned
    ]
    # The lookup is keyed exclusively by the authenticated mobile in deps. A
    # visible farmer context without that identity is insufficient because the
    # model must never be trusted to supply union/society/farmer identifiers.
    if farmer_unions and (getattr(ctx.deps, "mobile", None) or "").strip():
        return tool_def
    logger.info(
        "Hiding get_farmer_milk_collection_details tool because farmer_unions is "
        "empty (no resolved farmer context)"
    )
    return None


def _is_missing_code(value: str) -> bool:
    """True when a backend code is absent or a placeholder (empty, or non-positive
    like '0'). Defense-in-depth: refuse instead of sending junk to the live
    backend even if the tool is somehow reached without valid codes."""
    text = (value or "").strip()
    if not text:
        return True
    try:
        return int(text) <= 0
    except ValueError:
        return False


def _escape_markdown_cell(value) -> str:
    """Escape markdown table delimiter characters in cell content.

    None-safe: the lenient FarmerMilkCollection model (#12) allows missing
    fields (a partial PashuGPT row), so a cell may be None — render it as '-'.
    """
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _format_number(value, decimals: int = 2) -> str:
    """Format numeric values for compact table display.

    None-safe: the lenient model allows missing numeric fields (None) — render
    them as '-' instead of crashing on f-string formatting.
    """
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Create a markdown table with fixed headers and row ordering."""
    header_line = f"| {' | '.join(headers)} |"
    separator_line = f"| {' | '.join(['---'] * len(headers))} |"
    row_lines = [
        f"| {' | '.join(_escape_markdown_cell(cell) for cell in row)} |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_milk_collection_markdown(response) -> str:
    """
    Format tool response as deterministic markdown tables for frontend rendering.
    """
    sections: list[str] = []

    sections.append("### Milk Collection")
    if response.milk:
        milk_rows = [
            [
                record.date,
                record.shift,
                _format_number(record.qty, 2),
                _format_number(record.fat, 2),
                _format_number(record.snf, 2),
                _format_number(record.amount, 2),
            ]
            for record in response.milk
        ]
        sections.append(
            _build_markdown_table(
                ["Date", "Shift", "Qty (L)", "FAT", "SNF", "Amount"],
                milk_rows,
            )
        )
    else:
        sections.append("No milk records found for the selected date range.")

    sections.append("")
    sections.append("### Deductions")
    if response.deduction:
        deduction_rows = [
            [
                record.date,
                record.account_name,
                _format_number(record.amount, 2),
            ]
            for record in response.deduction
        ]
        sections.append(
            _build_markdown_table(
                ["Date", "Account", "Amount"],
                deduction_rows,
            )
        )
    else:
        sections.append("No deductions found for the selected date range.")

    return "\n".join(sections)


async def get_farmer_milk_collection_details(
    ctx: RunContext[FarmerContext],
    fromdate: str,
    todate: str,
) -> str:
    """
    Retrieve farmer milk collection records and deduction entries for a date range.

    Args:
        ctx: Authenticated farmer context supplied by the agent runtime.
        fromdate: Start date in YYYY-MM-DD format.
        todate: End date in YYYY-MM-DD format.

    Returns:
        str: Deterministic markdown tables for milk and deductions, or a clear failure message.
    """
    logger.info(
        "Farmer milk collection tool invoked from=%s to=%s",
        fromdate,
        todate,
    )

    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        logger.info("Farmer milk collection tool refused: no authenticated mobile")
        return (
            "Milk collection lookup failed.\n\n"
            "Your signed-in farmer profile is not available, so milk collection "
            "details can't be fetched."
        )

    try:
        validation_request = FarmerMilkCollectionRequestModel(
            unionCode="1",
            societyCode="1",
            farmerCode="1",
            fromdate=fromdate,
            todate=todate,
        )
        validation_request.validate_date_range()
    except ValueError as exc:
        logger.info(
            "Farmer milk collection validation failed: %s", str(exc)
        )
        return f"Milk collection lookup failed.\n\n{str(exc)}"

    use_amul_bpp = settings.enable_network and settings.beckn_callback_transactions_enabled
    session_id = ctx.deps.session_id if ctx and ctx.deps else None
    tool_call_id = getattr(ctx, "tool_call_id", None)
    try:
        farmers = (
            await fetch_authenticated_farmers(
                mobile,
                session_id=session_id,
                tool_call_id=tool_call_id,
            )
            if use_amul_bpp
            else await get_farmer_data_by_mobile(mobile)
        )
    except Exception as exc:
        logger.warning("Farmer profile lookup for milk collection failed: %s", exc)
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    accounts = authenticated_accounts(farmers or [])
    if not accounts:
        return (
            "Milk collection lookup failed.\n\n"
            "No union, society, and farmer account was found for your signed-in mobile."
        )

    if use_amul_bpp:
        outcomes = await asyncio.gather(
            *(
                fetch_milk_collection(
                    account,
                    fromdate=fromdate,
                    todate=todate,
                    session_id=session_id,
                    # One tool invocation may fan out across several owned
                    # accounts. Keep each durable operation distinct without
                    # putting farmer identifiers into its correlation key.
                    tool_call_id=(f"{tool_call_id}:account-{index}" if tool_call_id else None),
                )
                for index, account in enumerate(accounts)
            ),
            return_exceptions=True,
        )
    else:
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            return "Milk collection lookup failed.\n\nProvider access is not configured."
        outcomes = await asyncio.gather(
            *(
                get_farmer_milk_collection_details_api(
                    FarmerMilkCollectionRequestModel(
                        unionCode=account.union_code,
                        societyCode=account.society_code,
                        farmerCode=account.farmer_code,
                        fromdate=fromdate,
                        todate=todate,
                    ),
                    token,
                )
                for account in accounts
            ),
            return_exceptions=True,
        )

    responses = [
        response for response in outcomes
        if response is not None and not isinstance(response, BaseException)
    ]
    if not responses:
        logger.info(
            "Farmer milk collection lookup failed for all authenticated accounts from=%s to=%s",
            fromdate,
            todate,
        )
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    from app.models.milk_collection import FarmerMilkCollectionResponseModel

    response = FarmerMilkCollectionResponseModel(
        result="success",
        milk=[record for result in responses for record in result.milk],
        deduction=[record for result in responses for record in result.deduction],
    )

    formatted = _format_milk_collection_markdown(response)
    logger.info(
        "Farmer milk collection lookup succeeded accounts=%s from=%s to=%s milk_records=%s deductions=%s",
        len(accounts),
        fromdate,
        todate,
        len(response.milk),
        len(response.deduction),
    )
    return f"Farmer milk collection details fetched successfully:\n\n{formatted}"
