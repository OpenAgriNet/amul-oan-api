"""Signed-in text-chat tool for farmer ticket and breeding summaries."""

from datetime import datetime

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.beckn.amul import authenticated_accounts
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_ticket_summary_backend import (
    format_breeding_activity,
    get_linked_farmer_ticket_summaries,
    prepare_get_farmer_ticket_summary,
)
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryResponseModel
from app.config import get_config_value
from helpers.utils import get_logger

logger = get_logger(__name__)


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
            parts = "; ".join(
                format_breeding_activity(
                    detail.last_breeding_activity,
                    _date,
                    _display,
                    spoken=False,
                )
            )
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

    successful, failed = await get_linked_farmer_ticket_summaries(accounts, token)

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
