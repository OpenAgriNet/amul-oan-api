"""
Tool for fetching farmer milk collection and deduction details.
"""
import os

from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerAccount, FarmerContext
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


def _num(value) -> str:
    """Render a numeric field compactly, dropping a trailing .0 (e.g. 2.0 -> '2')."""
    if value is None:
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_SHIFT_LABELS = {"M": "morning", "E": "evening"}


def _format_milk_collection_summary_voice(response) -> str:
    """Deterministic plain-text summary for the voice agent.

    The voice agent runs on a small OSS model and speaks its answer aloud, so
    it gets a flat, labelled, per-record list (one record per line) instead of
    raw JSON or a markdown table. Each field is named so the model cannot
    confuse quantity with fat/SNF/amount when several records are present.
    """
    lines: list[str] = []

    if response.milk:
        lines.append(f"Milk collection records ({len(response.milk)}):")
        for i, r in enumerate(response.milk, 1):
            shift = _SHIFT_LABELS.get((r.shift or "").upper(), r.shift or "unknown")
            lines.append(
                f"  {i}. Date {r.date or 'unknown'}, {shift} shift: "
                f"quantity {_num(r.qty)} liters, "
                f"fat {_num(r.fat)}, SNF {_num(r.snf)}, "
                f"amount {_num(r.amount)} rupees."
            )
    else:
        lines.append("No milk collection records for the selected date range.")

    if response.deduction:
        lines.append(f"Deduction records ({len(response.deduction)}):")
        for i, d in enumerate(response.deduction, 1):
            lines.append(
                f"  {i}. Date {d.date or 'unknown'}, "
                f"{d.account_name or 'account'}: amount {_num(d.amount)} rupees."
            )
    else:
        lines.append("No deductions for the selected date range.")

    return "\n".join(lines)


async def _fetch_one_account_voice(
    account: FarmerAccount,
    fromdate: str,
    todate: str,
    token: str,
):
    """Fetch milk/deduction for one account. Returns the response, or None on failure.

    Dates are validated once by the caller before fan-out, so a per-account
    failure here is always an upstream/API issue, not a date problem.
    """
    request = FarmerMilkCollectionRequestModel(
        unionCode=account.union_code or "",
        societyCode=account.society_code or "",
        farmerCode=account.farmer_code or "",
        fromdate=fromdate,
        todate=todate,
    )
    response = await get_farmer_milk_collection_details_api(request, token)
    logger.info(
        "Milk collection lookup: union=%s society=%s farmer=%s from=%s to=%s ok=%s milk=%s ded=%s",
        account.union_code, account.society_code, account.farmer_code, fromdate, todate,
        response is not None,
        len(response.milk) if response else 0,
        len(response.deduction) if response else 0,
    )
    return response


def _account_label_voice(account: FarmerAccount, multi: bool) -> str:
    """Header for an account's section, only shown when fanning out over >1 account."""
    if not multi:
        return ""
    parts = []
    if account.society_name:
        parts.append(account.society_name)
    if account.farmer_code:
        parts.append(f"farmer code {account.farmer_code}")
    label = ", ".join(parts) if parts else f"account {account.farmer_code or '?'}"
    return f"Account — {label}:"


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


async def get_farmer_milk_collection_details_voice(
    ctx: RunContext[FarmerContext],
    union_code: str,
    society_code: str,
    farmer_code: str,
    fromdate: str,
    todate: str,
) -> str:
    """
    Fetch milk collection and deduction details for the signed-in farmer.

    A single mobile number can have more than one account (for example a
    separate cow account and buffalo account). This tool automatically looks
    up every account on the caller's mobile and reports them together, so you
    do not need to pick one. The codes you pass are only a fallback used when
    no farmer accounts are available in context.

    Args:
        ctx: The run context (automatically provided).
        union_code: Union code from farmer context (fallback only).
        society_code: Society code from farmer context (fallback only).
        farmer_code: Farmer code from farmer context (fallback only).
        fromdate: Start date in YYYY-MM-DD format (ISO).
        todate: End date in YYYY-MM-DD format (ISO).

    Returns:
        str: Formatted milk collection and deduction details across all of the
             farmer's accounts, or a clear failure message.
    """
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Milk collection lookup failed. Service is not configured."

    # Prefer the structured accounts from context (every account on the mobile).
    # Fall back to the LLM-supplied codes only when context has none.
    accounts = list(ctx.deps.farmer_accounts) if ctx.deps and ctx.deps.farmer_accounts else []
    if not accounts:
        accounts = [
            FarmerAccount(
                union_code=union_code,
                society_code=society_code,
                farmer_code=farmer_code,
            )
        ]
    logger.info(
        "Milk collection tool invoked: accounts=%s from=%s to=%s (llm_codes=%s/%s/%s)",
        len(accounts), fromdate, todate, union_code, society_code, farmer_code,
    )

    # Validate the date range once — it is the same for every account, so a bad
    # date is a single clear failure rather than a per-account error.
    try:
        FarmerMilkCollectionRequestModel(
            unionCode=accounts[0].union_code or "",
            societyCode=accounts[0].society_code or "",
            farmerCode=accounts[0].farmer_code or "",
            fromdate=fromdate,
            todate=todate,
        ).validate_date_range()
    except (ValidationError, ValueError) as e:
        logger.info("Milk collection date validation failed: from=%s to=%s error=%s", fromdate, todate, e)
        return f"Milk collection lookup failed. {e}"

    multi = len(accounts) > 1
    sections: list[str] = []
    any_success = False
    total_milk = 0

    for account in accounts:
        result = await _fetch_one_account_voice(account, fromdate, todate, token)
        if result is None:
            sections.append(
                (_account_label_voice(account, multi) + "\n" if multi else "")
                + "Unable to fetch milk collection details for this account right now."
            )
            continue

        any_success = True
        total_milk += len(result.milk)
        summary = _format_milk_collection_summary_voice(result)
        label = _account_label_voice(account, multi)
        sections.append(f"{label}\n{summary}" if label else summary)

    if not any_success:
        return "Milk collection lookup failed. Unable to fetch details at the moment."

    body = "\n\n".join(sections)
    logger.info(
        "Milk collection aggregate: accounts=%s any_success=%s total_milk_records=%s",
        len(accounts), any_success, total_milk,
    )
    return f"Milk collection details fetched successfully:\n\n{body}"
