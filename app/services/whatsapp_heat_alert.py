"""Send a single-farmer heat alert through the Sarlaben WhatsApp API."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


class WhatsAppHeatAlertConfigurationError(RuntimeError):
    """Raised when outbound WhatsApp delivery is not configured."""


class WhatsAppHeatAlertDeliveryError(RuntimeError):
    """Raised when the provider does not accept a WhatsApp message."""


async def send_heat_alert_to_farmer(
    *,
    mobile: str,
    animal_no: str,
    tag_no: str,
    booking_link: str,
) -> str | None:
    """Queue the approved heat-alert template for one farmer.

    Returns the provider's message UUID when one is included in the success
    response. The API key and provider response body are deliberately excluded
    from raised errors so logs cannot leak credentials or farmer data.
    """
    api_key = (settings.sarlaben_whatsapp_api_key or "").strip()
    if not api_key:
        raise WhatsAppHeatAlertConfigurationError(
            "SARLABEN_WHATSAPP_API_KEY is not configured"
        )

    endpoint = (
        f"{settings.sarlaben_whatsapp_base_url.rstrip('/')}"
        "/api/whatsapp/heat-alert"
    )
    request_body = {
        "mobile": mobile,
        "animalNo": animal_no,
        "tagNo": tag_no,
        "bookingLink": booking_link,
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.sarlaben_whatsapp_timeout_seconds
        ) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json=request_body,
            )
    except httpx.RequestError as exc:
        raise WhatsAppHeatAlertDeliveryError(
            "Sarlaben WhatsApp API request failed"
        ) from exc

    if response.status_code != 200:
        raise WhatsAppHeatAlertDeliveryError(
            f"Sarlaben WhatsApp API returned HTTP {response.status_code}"
        )

    try:
        response_body: Any = response.json()
    except ValueError as exc:
        raise WhatsAppHeatAlertDeliveryError(
            "Sarlaben WhatsApp API returned invalid JSON"
        ) from exc

    if not isinstance(response_body, dict) or response_body.get("success") is not True:
        raise WhatsAppHeatAlertDeliveryError(
            "Sarlaben WhatsApp API did not accept the message"
        )

    message_uuid = response_body.get("messageUuid")
    return message_uuid if isinstance(message_uuid, str) else None
