"""Signed-in voice tool for farmer ticket and breeding summaries."""

import os
from datetime import datetime

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_ticket_summary_backend import (
    format_breeding_activity,
    get_linked_farmer_ticket_summaries,
    prepare_get_farmer_ticket_summary,
)
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryResponseModel
from helpers.utils import get_logger

logger = get_logger(__name__)


def _text(value: object, fallback: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def _date(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "unknown date"
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).strftime("%d %B %Y")
    except ValueError:
        return text.split("T", 1)[0] if "T" in text else text


def _format_account(response: FarmerTicketSummaryResponseModel, account) -> str:
    farmer = response.farmer_details
    farmer_name = getattr(farmer, "farmer_name", None) or getattr(
        account, "farmer_name", None
    )
    society_name = getattr(farmer, "society_name", None) or getattr(
        account, "society_name", None
    )
    lines = [
        f"Farmer {_text(farmer_name)} at society {_text(society_name)}."
    ]
    if response.pending_tickets:
        lines.append(f"Pending tickets: {len(response.pending_tickets)}.")
        for ticket in response.pending_tickets:
            lines.append(
                f"Ticket {_text(ticket.ticket_number)} was booked on {_date(ticket.book_date_time)}."
            )
    else:
        lines.append("No pending tickets were found.")

    if not response.last_lactation_detail:
        lines.append("No animal breeding or calving history was found.")
    for detail in response.last_lactation_detail:
        facts = format_breeding_activity(
            detail.last_breeding_activity,
            _date,
            _text,
            spoken=True,
        )
        lines.append(
            f"Animal {_text(detail.animal_tag_number)}: "
            + "; ".join(facts)
            + "."
        )
    return "\n".join(lines)


async def get_farmer_ticket_summary(ctx: RunContext[FarmerContext]) -> str:
    """Fetch the signed-in farmer's ticket, AI, PD, and calving summaries.

    Use for personal pending-ticket questions or the caller's own latest
    artificial insemination, pregnancy diagnosis, or calving history. Account
    codes come only from authenticated context; never ask for or invent them.

    Args:
        ctx: Authenticated farmer context supplied by the voice runtime.

    Returns:
        str: Concise speakable account summaries.
    """
    mobile = (ctx.deps.mobile or "").strip() if ctx and ctx.deps else ""
    if not mobile:
        return "Farmer ticket summary lookup failed. Your signed-in profile is not available."
    accounts = list(ctx.deps.farmer_accounts or [])
    if not accounts:
        return (
            "Farmer ticket summary lookup failed. No union, society, and farmer "
            "account was found for your signed-in mobile."
        )
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Farmer ticket summary lookup failed. Service is not configured."

    successful, failed = await get_linked_farmer_ticket_summaries(accounts, token)
    if not successful:
        return "Farmer ticket summary lookup failed. Unable to fetch ticket details now."
    if not any(response.has_data() for _, response in successful):
        result = "No ticket, breeding, or calving records were found for your signed-in accounts."
    else:
        result = "Ticket summary fetched successfully.\n" + "\n".join(
            _format_account(response, account)
            for account, response in successful
        )
    if failed:
        result += f" Some linked accounts could not be checked: {failed} of {len(accounts)}."
    return result
