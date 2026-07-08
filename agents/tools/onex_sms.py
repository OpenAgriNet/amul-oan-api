"""Onex-Aura / OneXtel SMS gateway client for micro-loan approval messages.

Sends the DLT-approved KDCC micro-loan template. The message body is built from
a configurable Gujarati template (``ONEX_SMS_BODY_TEMPLATE``) with three
placeholders: ``{name}``, ``{amount}``, ``{code}``.

The client only sends — the caller decides whether sending is enabled
(``LOAN_SMS_ENABLED``) and records the outcome. All gateway credentials
(key / entityid / templateid) come from config/env, never hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

# The gateway returns status 100 on successful submission.
_SUCCESS_STATUS = 100


@dataclass
class SmsResult:
    ok: bool
    status: str  # 'sent' | 'failed'
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)


def _format_amount(amount: float) -> str:
    """Render the loan amount like the approved template (e.g. 5000 -> '5,000')."""
    try:
        return f"{int(round(float(amount))):,}"
    except (TypeError, ValueError):
        return str(amount)


def _to_msisdn(phone: str) -> str:
    """Build the 91-prefixed 12-digit destination from any phone form."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    last10 = digits[-10:]
    return f"91{last10}"


def build_loan_sms_body(farmer_name: str, amount: float, code: str) -> str:
    """Render the DLT body from the configured template. Kept public so tests and
    the dry-run path can assert on the exact text without sending anything."""
    name = (farmer_name or "").strip() or "ખેડૂત મિત્ર"  # "farmer friend" fallback
    return settings.onex_sms_body_template.format(
        name=name,
        amount=_format_amount(amount),
        code=code,
    )


async def send_loan_approval_sms(
    phone: str,
    farmer_name: str,
    amount: float,
    code: str,
) -> SmsResult:
    """Send the micro-loan approval SMS. Never raises — returns an SmsResult."""
    if not settings.onex_sms_key:
        logger.error("Onex SMS not configured: ONEX_SMS_KEY missing")
        return SmsResult(ok=False, status="failed", error="sms_gateway_not_configured")

    body = build_loan_sms_body(farmer_name, amount, code)
    params = {
        "key": settings.onex_sms_key,
        "to": _to_msisdn(phone),
        "from": settings.onex_sms_from,
        "body": body,
        "entityid": settings.onex_sms_entity_id or "",
        "templateid": settings.onex_sms_template_id or "",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.onex_sms_timeout_secs) as client:
            resp = await client.get(settings.onex_sms_base_url, params=params)
            resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = {}

        status_val = data.get("status")
        if status_val == _SUCCESS_STATUS or str(status_val) == str(_SUCCESS_STATUS):
            message_id = data.get("messageid") or data.get("messageId")
            logger.info("Loan SMS submitted to=%s messageid=%s", _to_msisdn(phone), message_id)
            return SmsResult(ok=True, status="sent", message_id=message_id, raw=data)

        # Gateway accepted the request but reported a non-success status.
        err = data.get("description") or f"gateway_status={status_val}"
        logger.error("Loan SMS rejected to=%s status=%s desc=%s", _to_msisdn(phone), status_val, err)
        return SmsResult(ok=False, status="failed", error=str(err), raw=data)
    except httpx.HTTPStatusError as e:
        logger.error("Loan SMS HTTP %s: %s", e.response.status_code, e.response.text[:200])
        return SmsResult(ok=False, status="failed", error=f"http_{e.response.status_code}")
    except Exception as e:  # network / timeout / unexpected
        logger.error("Loan SMS send error: %s", e)
        return SmsResult(ok=False, status="failed", error=str(e))
