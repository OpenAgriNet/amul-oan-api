"""Signed-in text-chat tool for farmer ticket and breeding summaries."""

import asyncio
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.beckn.amul import authenticated_accounts
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_ticket_summary_backend import get_farmer_ticket_summary_api
from agents.tools.models.farmer_ticket_summary import (
    FarmerTicketSummaryRequestModel,
    FarmerTicketSummaryResponseModel,
    LastBreedingActivityModel,
)
from app.config import get_config_value
from helpers.utils import get_logger

logger = get_logger(__name__)


async def prepare_get_farmer_ticket_summary(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if (getattr(ctx.deps, "mobile", None) or "").strip():
        return tool_def
    logger.info("Hiding farmer ticket summary tool: authenticated mobile is missing")
    return None


def _display(value: object, fallback: str = "-") -> str:
    text = str(value).strip() if value is not None else ""
    return text.replace("|", "\\|") if text else fallback


def _date(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "-"
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).strftime("%d-%b-%Y")
    except ValueError:
        return text.split("T", 1)[0] if "T" in text else text


def _activity_parts(activity: LastBreedingActivityModel | None) -> list[str]:
    if activity is None:
        return ["No breeding activity available"]
    parts: list[str] = []
    if activity.last_ai or activity.last_ai_ticket_number:
        ai = f"Last AI {_date(activity.last_ai)}"
        if activity.last_ai_ticket_number:
            ai += f" (ticket {_display(activity.last_ai_ticket_number)})"
        bull = activity.bull_id or (
            str(activity.ai_bull_id) if activity.ai_bull_id is not None else None
        )
        if bull:
            ai += f", bull {_display(bull)}"
        if activity.semen_type:
            ai += f", semen {_display(activity.semen_type)}"
        parts.append(ai)
    if activity.last_pd or activity.last_pd_ticket_number or activity.pd_status:
        pd = f"Last PD {_date(activity.last_pd)}"
        if activity.last_pd_ticket_number:
            pd += f" (ticket {_display(activity.last_pd_ticket_number)})"
        if activity.pd_status:
            pd += f", status {_display(activity.pd_status)}"
        parts.append(pd)
    if activity.last_calving or activity.last_calving_ticket_number:
        calving = f"Last calving {_date(activity.last_calving)}"
        if activity.last_calving_ticket_number:
            calving += f" (ticket {_display(activity.last_calving_ticket_number)})"
        parts.append(calving)
    return parts or ["No breeding activity available"]


def _format_account(response: FarmerTicketSummaryResponseModel, account) -> str:
    farmer = response.farmer_details
    farmer_name = getattr(farmer, "farmer_name", None) or getattr(
        account, "farmer_name", None
    )
    farmer_code = getattr(farmer, "farmer_code", None) or getattr(
        account, "farmer_code", None
    )
    society_name = getattr(farmer, "society_name", None) or getattr(
        account, "society_name", None
    )
    society_code = getattr(farmer, "society_code", None) or getattr(
        account, "society_code", None
    )

    lines = [
        f"### {_display(farmer_name, 'Farmer')} ({_display(farmer_code, 'code unavailable')})",
        f"Society: {_display(society_name, 'Unknown')} ({_display(society_code, 'code unavailable')})",
        "",
        "**Pending tickets**",
    ]
    if response.pending_tickets:
        for ticket in response.pending_tickets:
            lines.append(
                f"- Ticket {_display(ticket.ticket_number)} booked {_date(ticket.book_date_time)}"
            )
    else:
        lines.append("- No pending tickets found.")

    lines.extend(["", "**Animal breeding and calving history**"])
    if response.last_lactation_detail:
        for detail in response.last_lactation_detail:
            parts = "; ".join(_activity_parts(detail.last_breeding_activity))
            lines.append(f"- Animal {_display(detail.animal_tag_number)}: {parts}.")
    else:
        lines.append("- No animal breeding or calving history found.")
    return "\n".join(lines)


async def get_farmer_ticket_summary(ctx: RunContext[FarmerContext]) -> str:
    """Fetch ticket, AI, PD, and calving summaries for the signed-in farmer.

    Use for personal pending-ticket questions or the caller's own latest
    artificial insemination, pregnancy diagnosis, or calving history. Account
    codes come only from authenticated context; never ask for or invent them.

    Args:
        ctx: Authenticated farmer context supplied by the agent runtime.

    Returns:
        str: Deterministic account-grouped Markdown summary.
    """
    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        return (
            "Farmer ticket summary lookup failed.\n\n"
            "Your signed-in farmer profile is not available."
        )

    try:
        farmers = await get_farmer_data_by_mobile(mobile)
    except Exception:
        logger.warning("Farmer profile lookup for ticket summary failed")
        return (
            "Farmer ticket summary lookup failed.\n\n"
            "Unable to fetch ticket details at the moment."
        )
    accounts = authenticated_accounts(farmers or [])
    if not accounts:
        return (
            "Farmer ticket summary lookup failed.\n\n"
            "No union, society, and farmer account was found for your signed-in mobile."
        )

    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Farmer ticket summary lookup failed.\n\nProvider access is not configured."

    outcomes = await asyncio.gather(
        *(
            get_farmer_ticket_summary_api(
                FarmerTicketSummaryRequestModel(
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
    successful: list[tuple[object, FarmerTicketSummaryResponseModel]] = []
    failed = 0
    for account, outcome in zip(accounts, outcomes):
        if outcome is None or isinstance(outcome, BaseException):
            failed += 1
        else:
            successful.append((account, outcome))

    if not successful:
        return (
            "Farmer ticket summary lookup failed.\n\n"
            "Unable to fetch ticket details at the moment."
        )
    if not any(response.has_data() for _, response in successful):
        message = (
            "Farmer ticket summary lookup completed.\n\n"
            "No ticket, breeding, or calving records were found for your signed-in account(s)."
        )
        if failed:
            message += f"\n\nSome linked accounts could not be checked ({failed} of {len(accounts)})."
        return message

    sections = [
        _format_account(response, account)
        for account, response in successful
    ]
    result = "Farmer ticket summary fetched successfully:\n\n" + "\n\n".join(sections)
    if failed:
        result += f"\n\nSome linked accounts could not be checked ({failed} of {len(accounts)})."
    return result
