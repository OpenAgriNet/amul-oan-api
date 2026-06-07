"""
Tool for fetching farmer milk collection and deduction details.
"""
import os

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
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
    if farmer_unions:
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


def _escape_markdown_cell(value: str) -> str:
    """Escape markdown table delimiter characters in cell content."""
    return value.replace("|", "\\|")


def _format_number(value: float, decimals: int = 2) -> str:
    """Format numeric values for compact table display."""
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
    union_code: str,
    society_code: str,
    farmer_code: str,
    fromdate: str,
    todate: str,
) -> str:
    """
    Retrieve farmer milk collection records and deduction entries for a date range.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        fromdate: Start date in YYYY-MM-DD format.
        todate: End date in YYYY-MM-DD format.

    Returns:
        str: Deterministic markdown tables for milk and deductions, or a clear failure message.
    """
    logger.info(
        "Farmer milk collection tool invoked for union=%s society=%s farmer=%s from=%s to=%s",
        union_code,
        society_code,
        farmer_code,
        fromdate,
        todate,
    )

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Milk collection lookup failed.\n\nPASHUGPT_TOKEN is not configured."

    missing = [
        name
        for name, value in (
            ("union_code", union_code),
            ("society_code", society_code),
            ("farmer_code", farmer_code),
        )
        if _is_missing_code(value)
    ]
    if missing:
        logger.info(
            "Farmer milk collection tool refused: missing/placeholder codes %s "
            "(union=%s society=%s farmer=%s)",
            missing,
            union_code,
            society_code,
            farmer_code,
        )
        return (
            "Milk collection lookup failed.\n\n"
            "Could not determine your union, society, and farmer codes from the "
            "current farmer context, so milk collection details can't be fetched."
        )

    try:
        request = FarmerMilkCollectionRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            fromdate=fromdate,
            todate=todate,
        )
        request.validate_date_range()
    except ValueError as exc:
        logger.info(
            "Farmer milk collection validation failed for union=%s society=%s farmer=%s: %s",
            union_code,
            society_code,
            farmer_code,
            str(exc),
        )
        return f"Milk collection lookup failed.\n\n{str(exc)}"

    response = await get_farmer_milk_collection_details_api(request, token)
    if response is None:
        logger.info(
            "Farmer milk collection lookup failed for union=%s society=%s farmer=%s from=%s to=%s",
            union_code,
            society_code,
            farmer_code,
            fromdate,
            todate,
        )
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    formatted = _format_milk_collection_markdown(response)
    logger.info(
        "Farmer milk collection lookup succeeded for union=%s society=%s farmer=%s from=%s to=%s milk_records=%s deductions=%s",
        union_code,
        society_code,
        farmer_code,
        fromdate,
        todate,
        len(response.milk),
        len(response.deduction),
    )
    return f"Farmer milk collection details fetched successfully:\n\n{formatted}"
