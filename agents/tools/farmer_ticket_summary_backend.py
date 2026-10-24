"""PII-safe provider adapter for GetFarmerTicketSummary."""

import json

import httpx
from pydantic import ValidationError

from agents.tools.models.farmer_ticket_summary import (
    FarmerTicketSummaryRequestModel,
    FarmerTicketSummaryResponseModel,
)
from app.config import settings
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)


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
