"""
Tool for fetching farmer milk collection and deduction details.
"""
import asyncio
import json
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.tools.telemetry import reason, report_tool
from agents.deps import FarmerContext
from agents.tools.beckn.amul import (
    authenticated_accounts,
    fetch_authenticated_farmers,
    fetch_milk_collection,
)
from agents.tools.models.milk_collection import FarmerMilkCollectionRequestModel
from helpers.utils import get_logger

logger = get_logger(__name__)


class InteractiveMilk:
    """Private, per-turn intent and the latest ownership-checked display result."""

    def __init__(self, selected=None, result=None, *, failed=False):
        self.selected, self.result, self.failed = selected, result, failed
        self.request = None

    def context(self) -> str:
        context = dict(selected_account=self.selected, date_reselection_required=self.failed)
        if self.result:
            context.update({k: self.result[k] for k in ("status", "account", "period", "summary")})
        encoded = json.dumps(context, ensure_ascii=False)
        if len(encoded.encode()) > 4096:
            # Provider labels can be arbitrarily long; omit them rather than truncate facts.
            context.pop("selected_account", None)
            context.pop("account", None)
            encoded = json.dumps(context, ensure_ascii=False)
        return encoded

    def stage(self, fromdate, todate) -> str:
        from app.agui.models import Period

        period = None
        if not self.failed and fromdate and todate:
            try:
                period = Period(fromdate=fromdate, todate=todate).model_dump()
            except ValueError:
                pass  # Missing or invalid dates are resolved by the backend picker.
        intent = dict(period=period)
        if self.request is not None and self.request != intent:
            report_tool("error", "one milk workflow per turn")
            return "A milk lookup is already staged. Do not request another range in this turn."
        self.request = intent
        return "Milk workflow requested. The backend will show required account/date controls or the chart. Do not invent records or print tables."

    def read(self, fromdate=None, todate=None, offset=0) -> str:
        from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel
        from app.agui.milk import collection_day, milk_result
        from app.agui.models import Period

        if self.result is None:
            report_tool("no_result", "no owned displayed result")
            return "No currently owned milk result is available in this chat. Request a new lookup if needed."
        if offset < 0:
            return "Offset must be nonnegative."
        data = self.result
        if fromdate is not None or todate is not None:
            try:
                period = Period(fromdate=fromdate, todate=todate).model_dump()
            except ValueError:
                return "Supply both dates as a valid ISO range of at most 31 days."
            if not data["period"]["fromdate"] <= period["fromdate"] <= period["todate"] <= data["period"]["todate"]:
                return "The displayed result does not cover that range. Request a fresh lookup; do not quote partial totals."
            rows = dict(milk=[], deduction=[])
            excluded = 0
            for source, destination in (("milk", "milk"), ("deductions", "deduction")):
                for row in data[source]:
                    day = collection_day(row["date"])
                    if day is None:
                        excluded += 1
                    elif period["fromdate"] <= day <= period["todate"]:
                        rows[destination].append(row)
            if excluded:
                return "Some saved rows have unparseable dates. Reliable totals for this subrange are unavailable; do not quote partial totals."
            data = milk_result(data["account"], period, FarmerMilkCollectionResponseModel.model_validate(rows))
        combined = [("milk", row) for row in data["milk"]] + [("deductions", row) for row in data["deductions"]]
        page = {k: data[k] for k in ("status", "account", "period", "summary", "warnings")}
        page.update(milk=[], deductions=[], total_rows=len(combined), next_offset=None, complete=True,
                    total_deductions_rupees=round(sum(row["amount"] or 0 for row in data["deductions"]), 2))
        for kind, row in combined[offset:offset + 20]:
            page[kind].append(row)
            if len(json.dumps(page, ensure_ascii=False).encode()) > 16 * 1024 - 128:
                page[kind].pop()
                break
        taken = len(page["milk"]) + len(page["deductions"])
        if offset < len(combined) and not taken:
            return "The displayed record is too large to return safely. No rows or totals were truncated."
        page["complete"] = offset + taken >= len(combined)
        page["next_offset"] = None if page["complete"] else offset + taken
        encoded = json.dumps(page, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 16 * 1024:
            return "The displayed record is too large to return safely. No rows or totals were truncated."
        return encoded


def _interactive_milk_available(deps) -> bool:
    return deps.interactive_milk and deps.persona == "farmer" and deps.has_farmer_profile() and deps._milk_ui is not None


async def prepare_interactive_milk(ctx: RunContext[FarmerContext], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if _interactive_milk_available(ctx.deps) else None


async def request_milk_records(ctx: RunContext[FarmerContext], fromdate: str | None = None, todate: str | None = None) -> str:
    """Request the farmer's own milk records/payment as interactive controls and a chart.

    Args:
        ctx: Trusted farmer and interactive session context.
        fromdate: Clear user-supplied start date in YYYY-MM-DD, otherwise omit.
        todate: Clear user-supplied end date in YYYY-MM-DD, otherwise omit.
    """
    if not _interactive_milk_available(ctx.deps):
        report_tool("error", "interactive milk unavailable")
        return "Interactive milk lookup is unavailable for this turn."
    booking = getattr(ctx.deps, "_booking_ui", None)
    selector = getattr(ctx.deps, "_selector_ui", None)
    if selector is not None and selector.request is not None:
        return "A selector workflow is already staged. Request one workflow per turn."
    if booking is not None and booking.request is not None:
        return "An AI booking workflow is already staged. Request one workflow per turn."
    return ctx.deps._milk_ui.stage(fromdate, todate)


async def read_displayed_milk_records(ctx: RunContext[FarmerContext], fromdate: str | None = None,
                                     todate: str | None = None, offset: int = 0) -> str:
    """Read saved milk records to answer a follow-up about the chart already displayed.

    Args:
        ctx: Trusted farmer and interactive session context.
        fromdate: Optional start of a subrange within the displayed period, YYYY-MM-DD.
        todate: Optional end of that subrange, YYYY-MM-DD; supply both dates or neither.
        offset: Row offset from the previous page's next_offset; defaults to zero.
    """
    if not _interactive_milk_available(ctx.deps):
        report_tool("error", "interactive milk unavailable")
        return "Interactive milk lookup is unavailable for this turn."
    return ctx.deps._milk_ui.read(fromdate, todate, offset)


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
    if getattr(ctx.deps, "interactive_milk", False):
        return None
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


def milk_totals(milk) -> tuple[int, float, float, float | None, float | None]:
    """Days with milk, litres, amount, and FAT/SNF averaged by litres (as a dairy does).

    Computed here so the model quotes a total instead of adding slips up: live chat
    stated a wrong total in 80% of the answers that gave one (29 Sep-12 Oct 2026).
    """
    def by_litres(attr: str) -> float | None:
        pairs = [(getattr(r, attr), r.qty) for r in milk if getattr(r, attr) is not None and r.qty]
        litres = sum(q for _, q in pairs)
        return sum(v * q for v, q in pairs) / litres if litres else None

    days = len({r.date for r in milk if r.qty})
    return (
        days,
        sum(r.qty or 0 for r in milk),
        sum(r.amount or 0 for r in milk),
        by_litres("fat"),
        by_litres("snf"),
    )


def _display_date(value: str) -> str:
    """YYYY-MM-DD (validated tool input) -> dd-mm-yyyy, as the slips show dates."""
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d-%m-%Y")


def _format_milk_collection_markdown(response, fromdate: str, todate: str) -> str:
    """
    Format tool response as deterministic markdown tables for frontend rendering.
    """
    sections: list[str] = []

    if response.milk:
        days, litres, amount, fat, snf = milk_totals(response.milk)
        sections.append("### Summary")
        sections.append(
            _build_markdown_table(
                ["Period", "Days with milk", "Total milk (L)", "Total amount (Rs)", "Average FAT", "Average SNF"],
                [[
                    f"{_display_date(fromdate)} to {_display_date(todate)}",
                    str(days),
                    _format_number(litres, 2),
                    _format_number(amount, 2),
                    _format_number(fat, 2),
                    _format_number(snf, 2),
                ]],
            )
        )
        sections.append("")

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
        str: Deterministic markdown tables (summary, milk, deductions), or a clear failure message.
    """
    logger.info(
        "Farmer milk collection tool invoked from=%s to=%s",
        fromdate,
        todate,
    )

    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        logger.info("Farmer milk collection tool refused: no authenticated mobile")
        report_tool("error", "no signed-in mobile")
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
        report_tool("error", f"date range: {exc}")
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
        report_tool("unavailable", f"farmer profile lookup: {reason(exc)}")
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    accounts = authenticated_accounts(farmers or [])
    if not accounts:
        report_tool("no_result", "no account for the signed-in mobile")
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
        report_tool("unavailable", f"milk collection provider failed for all {len(accounts)} accounts")
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    if len(responses) < len(accounts):
        report_tool(
            "unavailable",
            f"milk collection provider failed for {len(accounts) - len(responses)} of {len(accounts)} accounts",
        )

    from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel

    response = FarmerMilkCollectionResponseModel(
        result="success",
        milk=[record for result in responses for record in result.milk],
        deduction=[record for result in responses for record in result.deduction],
    )

    formatted = _format_milk_collection_markdown(response, fromdate, todate)
    logger.info(
        "Farmer milk collection lookup succeeded accounts=%s from=%s to=%s milk_records=%s deductions=%s",
        len(accounts),
        fromdate,
        todate,
        len(response.milk),
        len(response.deduction),
    )
    return f"Farmer milk collection details fetched successfully:\n\n{formatted}"
