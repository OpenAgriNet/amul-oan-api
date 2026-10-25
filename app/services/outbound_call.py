"""Generic Raya/LitWiz outbound-call adapter.

This module deliberately knows nothing about heat alerts, ticket eligibility,
or database state. Callers supply the number, agent, and agent arguments.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)


class OutboundCallError(RuntimeError):
    """Base error whose category is safe to persist and log."""

    category = "outbound_error"


class OutboundCallConfigurationError(OutboundCallError):
    category = "configuration"


class OutboundCallDefiniteFailure(OutboundCallError):
    """The request is known not to have created a call."""

    def __init__(self, category: str, *, retryable: bool = False):
        super().__init__(category)
        self.category = category
        self.retryable = retryable


class OutboundCallAmbiguousFailure(OutboundCallError):
    """The provider may have accepted the call; automatic retry is unsafe."""

    def __init__(self, category: str = "ambiguous"):
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class OutboundCallAccepted:
    call_uuid: str


def normalize_indian_mobile(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if not re.fullmatch(r"\d{10}", digits):
        raise OutboundCallDefiniteFailure("invalid_phone")
    return digits


async def initiate_outbound_call(
    to_number: str,
    agent_id: str,
    agent_args: Mapping[str, Any] | None = None,
    *,
    out_did: str | None = None,
    country_code: str = "91",
    timezone: str = "Asia/Kolkata",
) -> OutboundCallAccepted:
    """Initiate one call and return its provider UUID.

    Only connect failures and 429 responses are classified as safely retryable.
    A timeout, 5xx, malformed success body, or invalid UUID is ambiguous because
    the provider may already have started the call.
    """
    api_key = (settings.raya_outbound_call_api_key or "").strip()
    url = (settings.raya_outbound_call_api_url or "").strip()
    try:
        uuid.UUID(str(agent_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise OutboundCallConfigurationError("invalid agent id") from exc
    if not api_key or not url:
        raise OutboundCallConfigurationError("outbound provider is not configured")

    payload: dict[str, Any] = {
        "agent_id": str(agent_id),
        "to_number": normalize_indian_mobile(to_number),
        "agent_args": dict(agent_args or {}),
        "country_code": country_code,
        "timezone": timezone,
    }
    if out_did:
        payload["out_did"] = out_did

    started = time.perf_counter()
    status_code: int | None = None
    try:
        async with httpx.AsyncClient(
            timeout=settings.raya_outbound_call_timeout_seconds
        ) as client:
            response = await client.post(
                url,
                headers={"X-API-Key": api_key, "Accept": "application/json"},
                json=payload,
            )
        status_code = response.status_code
        if status_code == 429:
            raise OutboundCallDefiniteFailure("rate_limited", retryable=True)
        if status_code in {400, 401, 403, 404, 405, 422}:
            raise OutboundCallDefiniteFailure(f"http_{status_code}")
        if status_code < 200 or status_code >= 300:
            raise OutboundCallAmbiguousFailure(f"http_{status_code}_ambiguous")
        data = response.json()
        call_uuid = str(data.get("uuid") or "") if isinstance(data, dict) else ""
        uuid.UUID(call_uuid)
        return OutboundCallAccepted(call_uuid=call_uuid)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise OutboundCallDefiniteFailure("connect_failure", retryable=True) from exc
    except OutboundCallError:
        raise
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise OutboundCallAmbiguousFailure("transport_ambiguous") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise OutboundCallAmbiguousFailure("invalid_success_response") from exc
    finally:
        logger.info(
            "Outbound call provider request completed status=%s latency_ms=%s",
            status_code,
            round((time.perf_counter() - started) * 1000),
        )
