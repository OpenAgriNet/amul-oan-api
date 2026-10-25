"""Inbound partner webhook endpoints."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.auth.webhook_token import require_webhook_token
from app.config import settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.heat_alert import HeatAlertWebhookRequest, HeatAlertWebhookResponse
from app.models.webhook import HeatAlertWebhookEvent
from app.services.whatsapp_heat_alert import (
    WhatsAppHeatAlertConfigurationError,
    WhatsAppHeatAlertDeliveryError,
    send_heat_alert_to_farmer,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _parse_iso8601_utc(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_epoch_seconds(value: str | None) -> Optional[int]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


@router.post(
    "/alerts",
    response_model=HeatAlertWebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_webhook_token)],
)
async def receive_heat_alert(payload: HeatAlertWebhookRequest) -> HeatAlertWebhookResponse:
    """Accept a partner heat-alert webhook payload.

    Validates the partner contract, persists to the webhook Postgres database,
    queues one WhatsApp message for a newly inserted alert, and returns a 202
    acknowledgement.
    """
    if not webhook_db_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook persistence DB is not configured",
        )

    whatsapp_values = {
        "farmer_contact": payload.farmer_contact.strip(),
        "Pashuaadhar No": (payload.pashuaadhar_no or "").strip(),
        "tag no": (payload.tag_no or "").strip(),
        "booking_url": (payload.booking_url or "").strip(),
    }
    missing_whatsapp_fields = [
        name for name, value in whatsapp_values.items() if not value
    ]
    if missing_whatsapp_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Missing fields required for WhatsApp delivery: "
                + ", ".join(missing_whatsapp_fields)
            ),
        )

    received_at = datetime.now(timezone.utc)
    call_due_at = received_at + timedelta(
        seconds=max(0, settings.heat_alert_call_delay_seconds)
    )
    payload_raw = payload.model_dump(by_alias=True, mode="json")
    partner_alert_epoch_s = _parse_epoch_seconds(payload.alert_time)
    ai_window_start_at = _parse_iso8601_utc(payload.ai_window_start_time)
    ai_window_end_at = _parse_iso8601_utc(payload.ai_window_end_time)
    partner_timestamp_at = _parse_iso8601_utc(payload.timestamp)

    event_values = dict(
        received_at=received_at,
        alert_id=payload.alert_id,
        device_id=payload.device_id,
        notification_type=payload.notification_type,
        alert_type=payload.alert_type,
        heat_score=payload.heat_score,
        tag_no=payload.tag_no,
        system_tag_no=payload.system_tag_no,
        pashuaadhar_no=payload.pashuaadhar_no,
        farm_name=payload.farm_name,
        farm_id=payload.farm_id,
        society_code=payload.society_code,
        farmer_code=payload.farmer_code,
        farmer_contact=payload.farmer_contact,
        ai_window_start_time_raw=payload.ai_window_start_time,
        ai_window_end_time_raw=payload.ai_window_end_time,
        alert_time_raw=payload.alert_time,
        timestamp_raw=payload.timestamp,
        msg_title=payload.msg_title,
        msg_body=payload.msg_body,
        booking_url=payload.booking_url,
        ai_window_start_at=ai_window_start_at,
        ai_window_end_at=ai_window_end_at,
        partner_timestamp_at=partner_timestamp_at,
        partner_alert_epoch_s=partner_alert_epoch_s,
        payload_raw=payload_raw,
        call_due_at=call_due_at,
        call_next_attempt_at=call_due_at,
        call_status="scheduled",
        call_updated_at=received_at,
    )

    try:
        async with get_webhook_session() as session:
            # DB-level idempotency on (alert_id, farmer_contact): retries with the
            # same pair are accepted without creating duplicate rows.
            insert_stmt = (
                pg_insert(HeatAlertWebhookEvent)
                .values(**event_values)
                .on_conflict_do_nothing(
                    index_elements=["alert_id", "farmer_contact"]
                )
                .returning(
                    HeatAlertWebhookEvent.id,
                    HeatAlertWebhookEvent.received_at,
                )
            )
            inserted_row = (await session.execute(insert_stmt)).first()
            if inserted_row is not None:
                record_id, stored_received_at = inserted_row
                message_uuid = await send_heat_alert_to_farmer(
                    mobile=whatsapp_values["farmer_contact"],
                    animal_no=whatsapp_values["Pashuaadhar No"],
                    tag_no=whatsapp_values["tag no"],
                    booking_link=whatsapp_values["booking_url"],
                )
                logger.info(
                    "Heat alert WhatsApp queued id=%s alert_id=%s message_uuid=%s",
                    record_id,
                    payload.alert_id,
                    message_uuid,
                )
            else:
                existing_stmt = (
                    select(
                        HeatAlertWebhookEvent.id,
                        HeatAlertWebhookEvent.received_at,
                    )
                    .where(
                        HeatAlertWebhookEvent.alert_id == payload.alert_id,
                        HeatAlertWebhookEvent.farmer_contact == payload.farmer_contact,
                    )
                    .order_by(HeatAlertWebhookEvent.received_at.desc())
                    .limit(1)
                )
                existing_row = (await session.execute(existing_stmt)).first()
                if existing_row is None:
                    raise RuntimeError(
                        "Webhook upsert conflict but existing row not found"
                    )
                record_id, stored_received_at = existing_row
            await session.commit()
    except WhatsAppHeatAlertConfigurationError as exc:
        logger.error(
            "Heat alert WhatsApp is not configured alert_id=%s error=%s",
            payload.alert_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WhatsApp delivery is not configured",
        ) from exc
    except WhatsAppHeatAlertDeliveryError as exc:
        logger.error(
            "Heat alert WhatsApp delivery failed alert_id=%s error=%s",
            payload.alert_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="WhatsApp provider did not accept the message",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Heat alert webhook persistence failed alert_id=%s device_id=%s error=%s",
            payload.alert_id,
            payload.device_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist webhook payload",
        ) from exc

    record_id = UUID(str(record_id))
    # farmer_contact is a phone number, so it stays out of the log.
    logger.info(
        "Heat alert webhook accepted id=%s alert_id=%s device_id=%s "
        "notification_type=%s",
        record_id,
        payload.alert_id,
        payload.device_id,
        payload.notification_type,
    )
    return HeatAlertWebhookResponse(
        accepted=True,
        id=record_id,
        received_at=stored_received_at or received_at,
    )
