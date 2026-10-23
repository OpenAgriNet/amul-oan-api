"""
Tool for fetching farmer milk collection and deduction details.
"""
import asyncio
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.beckn.amul import (
    authenticated_accounts,
    fetch_authenticated_farmers,
    fetch_milk_collection,
)
from agents.tools.milk_summary import MilkSummary, summarize_milk
from agents.tools.models.milk_collection import FarmerMilkCollectionRequestModel
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


def _account_label(account) -> str:
    parts = [account.society_name] if account.society_name else []
    parts.append(f"farmer code {account.farmer_code}")
    return ", ".join(parts)


def _display_date(value: str) -> str:
    """YYYY-MM-DD (validated tool input) -> dd-mm-yyyy, as the slips show dates."""
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d-%m-%Y")


def _format_day(day: tuple[str, float] | None) -> str:
    return "-" if day is None else f"{day[0]} ({_format_number(day[1], 2)} L)"


def _format_summary_markdown(
    columns: list[tuple[str, MilkSummary]], fromdate: str, todate: str
) -> str:
    """One column per account, plus "All accounts" when there are several."""
    shifts = [label for label in ("morning", "evening") if any(label in s.shift_qty for _, s in columns)]
    shifts += sorted({label for _, s in columns for label in s.shift_qty} - set(shifts))

    def slips(s: MilkSummary) -> str:
        split = ", ".join(f"{s.shift_slips.get(label, 0)} {label}" for label in shifts)
        return f"{s.slips} ({split})" if split else str(s.slips)

    rows = [
        ("Period", lambda s: f"{_display_date(fromdate)} to {_display_date(todate)}"),
        ("Days with milk", lambda s: str(s.days)),
        ("Slips", slips),
        ("Total milk (L)", lambda s: _format_number(s.qty, 2)),
        ("Total amount (Rs)", lambda s: _format_number(s.amount, 2)),
    ]
    for label in shifts:
        rows.append((f"{label.capitalize()} milk (L)", lambda s, k=label: _format_number(s.shift_qty.get(k, 0.0), 2)))
        rows.append((f"{label.capitalize()} amount (Rs)", lambda s, k=label: _format_number(s.shift_amount.get(k, 0.0), 2)))
    rows += [
        ("Average milk per day (L)", lambda s: _format_number(s.qty_per_day, 2)),
        ("Average FAT", lambda s: _format_number(s.fat, 2)),
        ("Average SNF", lambda s: _format_number(s.snf, 2)),
        ("Average rate (Rs per L)", lambda s: _format_number(s.rate, 2)),
        ("Highest day", lambda s: _format_day(s.highest_day)),
        ("Lowest day", lambda s: _format_day(s.lowest_day)),
        ("Deductions (Rs)", lambda s: f"{_format_number(s.deductions, 2)} ({s.deduction_entries} entries)"),
    ]
    if any(s.incomplete_slips for _, s in columns):
        rows.append(("Slips missing qty or amount", lambda s: str(s.incomplete_slips)))

    headers = ["Measure", *(name for name, _ in columns)] if len(columns) > 1 else ["Measure", "Value"]
    return _build_markdown_table(
        headers, [[label, *(render(s) for _, s in columns)] for label, render in rows]
    )


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
        str: A summary table (totals, averages, best and worst day, computed here so the
             model never adds slips up), then the milk and deduction tables, or a
             clear failure message.
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

    session_id = ctx.deps.session_id if ctx and ctx.deps else None
    tool_call_id = getattr(ctx, "tool_call_id", None)
    try:
        farmers = await fetch_authenticated_farmers(
            mobile,
            session_id=session_id,
            tool_call_id=tool_call_id,
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

    fetched = [
        (account, response) for account, response in zip(accounts, outcomes)
        if response is not None and not isinstance(response, BaseException)
    ]
    responses = [response for _, response in fetched]
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

    from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel

    response = FarmerMilkCollectionResponseModel(
        result="success",
        milk=[record for result in responses for record in result.milk],
        deduction=[record for result in responses for record in result.deduction],
    )

    columns = [
        (_account_label(account), summarize_milk(result.milk, result.deduction))
        for account, result in fetched
    ]
    if len(columns) > 1:
        columns.append(("All accounts", summarize_milk(response.milk, response.deduction)))
    sections = ["### Summary", _format_summary_markdown(columns, fromdate, todate)]
    missing = [
        _account_label(account) for account, outcome in zip(accounts, outcomes)
        if outcome is None or isinstance(outcome, BaseException)
    ]
    if missing:
        sections.append(
            f"Records for {'; '.join(missing)} could not be fetched right now, "
            "so the figures above leave that account out."
        )

    formatted = "\n".join([*sections, "", _format_milk_collection_markdown(response)])
    logger.info(
        "Farmer milk collection lookup succeeded accounts=%s from=%s to=%s milk_records=%s deductions=%s",
        len(accounts),
        fromdate,
        todate,
        len(response.milk),
        len(response.deduction),
    )
    return f"Farmer milk collection details fetched successfully:\n\n{formatted}"
