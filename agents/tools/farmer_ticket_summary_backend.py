"""PII-safe provider adapter for GetFarmerTicketSummary."""

import asyncio
import json
from collections.abc import Callable

import httpx
from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.models.farmer_ticket_summary import (
    FarmerTicketSummaryRequestModel,
    FarmerTicketSummaryResponseModel,
    LastBreedingActivityModel,
)
from app.config import settings
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)


async def prepare_get_farmer_ticket_summary(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if (getattr(ctx.deps, "mobile", None) or "").strip():
        return tool_def
    logger.info("Hiding farmer ticket summary tool: authenticated mobile is missing")
    return None


def format_breeding_activity(
    activity: LastBreedingActivityModel | None,
    format_date: Callable[[str | None], str],
    format_value: Callable[[object], str],
    *,
    spoken: bool,
) -> list[str]:
    """Build the same breeding facts for Markdown and spoken renderers."""
    empty = "no breeding activity available" if spoken else "No breeding activity available"
    if activity is None:
        return [empty]

    def add_ticket(text: str, number: str | None) -> str:
        if not number:
            return text
        ticket = format_value(number)
        return f"{text}, ticket {ticket}" if spoken else f"{text} (ticket {ticket})"

    parts: list[str] = []
    if activity.last_ai or activity.last_ai_ticket_number:
        label = "last A I on " if spoken else "Last AI "
        fact = add_ticket(
            f"{label}{format_date(activity.last_ai)}", activity.last_ai_ticket_number
        )
        bull = activity.bull_identifier()
        if bull:
            fact += f", bull {format_value(bull)}"
        if activity.semen_type:
            semen_label = "semen type" if spoken else "semen"
            fact += f", {semen_label} {format_value(activity.semen_type)}"
        parts.append(fact)
    if activity.last_pd or activity.last_pd_ticket_number or activity.pd_status:
        label = "last P D on " if spoken else "Last PD "
        fact = add_ticket(
            f"{label}{format_date(activity.last_pd)}", activity.last_pd_ticket_number
        )
        if activity.pd_status:
            fact += f", status {format_value(activity.pd_status)}"
        parts.append(fact)
    if activity.last_calving or activity.last_calving_ticket_number:
        label = "last calving on " if spoken else "Last calving "
        parts.append(
            add_ticket(
                f"{label}{format_date(activity.last_calving)}",
                activity.last_calving_ticket_number,
            )
        )
    return parts or [empty]


def _response_shape(payload: object) -> dict[str, object]:
    """Return counts/keys only; never send farmer values to telemetry."""
    if not isinstance(payload, dict):
        return {"json_type": type(payload).__name__}
    return {
        "json_type": "object",
        "keys": sorted(payload),
        "pending_ticket_count": len(payload.get("pendingTickets") or [])
        if isinstance(payload.get("pendingTickets"), list)
        else None,
        "lactation_detail_count": len(payload.get("lastLactationDetail") or [])
        if isinstance(payload.get("lastLactationDetail"), list)
        else None,
    }


def _is_record_not_found(payload: object) -> bool:
    if isinstance(payload, str):
        return "record not found" in payload.lower()
    if isinstance(payload, dict):
        return any(_is_record_not_found(value) for value in payload.values())
    return False


async def get_farmer_ticket_summary_api(
    request: FarmerTicketSummaryRequestModel,
    token: str,
) -> FarmerTicketSummaryResponseModel | None:
    """Fetch one authenticated account's ticket/breeding summary.

    A provider business response containing "Record Not Found" is normalized to
    an empty successful response. Other transport or contract failures return
    None so the caller can preserve partial success across linked accounts.
    """
    api_url = f"{settings.amulpashudhan_base_url}/GetFarmerTicketSummary"
    try:
        with start_observation(
            "get_farmer_ticket_summary_api",
            input={"lookup": "authenticated_farmer_account"},
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(
                timeout=settings.farmer_backend_http_timeout_seconds
            ) as client:
                response = await client.get(
                    api_url,
                    params=request.to_query_params(),
                    headers={
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
                response.raise_for_status()
            payload = response.json()
            if _is_record_not_found(payload):
                if observation is not None:
                    observation.update(
                        output={"status_code": response.status_code, "record_found": False}
                    )
                return FarmerTicketSummaryResponseModel()
            if not isinstance(payload, dict):
                raise ValueError("ticket summary provider response is not an object")
            expected_keys = {
                "farmerDetails", "pendingTickets", "lastLactationDetail"
            }
            if not expected_keys.intersection(payload):
                raise ValueError("ticket summary provider response has no known fields")
            if observation is not None:
                observation.update(
                    output={
                        "status_code": response.status_code,
                        **_response_shape(payload),
                    }
                )
        return FarmerTicketSummaryResponseModel.model_validate(payload)
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.text or ""
        except Exception:
            body = ""
        if "record not found" in body.lower():
            logger.info(
                "Farmer ticket summary provider returned no record (status=%s)",
                exc.response.status_code,
            )
            return FarmerTicketSummaryResponseModel()
        logger.error(
            "Farmer ticket summary provider HTTP failure (status=%s)",
            exc.response.status_code,
        )
    except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError):
        logger.error("Farmer ticket summary provider returned no usable result")
    except Exception:
        logger.error("Farmer ticket summary provider request failed")
    return None


async def get_linked_farmer_ticket_summaries(
    accounts: list[object], token: str
) -> tuple[list[tuple[object, FarmerTicketSummaryResponseModel]], int]:
    """Fetch linked accounts concurrently and separate successes from failures."""
    outcomes = await asyncio.gather(
        *(
            get_farmer_ticket_summary_api(
                FarmerTicketSummaryRequestModel(
                    unionCode=str(getattr(account, "union_code", "") or ""),
                    societyCode=str(getattr(account, "society_code", "") or ""),
                    farmerCode=str(getattr(account, "farmer_code", "") or ""),
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
    return successful, failed
