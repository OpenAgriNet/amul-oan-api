"""Correlate Raya voice turns with heat-alert calls and persist booking success."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import or_, select, update

from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent
from helpers.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HeatAlertCallContext:
    event_id: str
    campaign: str = "heat_alert_ai_booking"
    animal_number: str | None = None
    tag_number: str | None = None


def _uuid(value: str | None) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except (ValueError, TypeError, AttributeError):
        return None


async def associate_heat_alert_voice_call(
    *,
    session_id: str,
    process_id: str | None,
    event_id: str | None,
    provider_call_uuid: str | None,
) -> HeatAlertCallContext | None:
    """Resolve explicit event id first, then provider/session identifiers."""
    if not webhook_db_configured():
        return None
    parsed_event_id = _uuid(event_id)
    candidates = {
        value.strip()
        for value in (provider_call_uuid, process_id, session_id)
        if value and value.strip()
    }
    async with get_webhook_session() as session:
        event = None
        if parsed_event_id is not None:
            event = (
                await session.execute(
                    select(HeatAlertWebhookEvent)
                    .where(
                        HeatAlertWebhookEvent.id == parsed_event_id,
                        HeatAlertWebhookEvent.called.is_(True),
                        HeatAlertWebhookEvent.call_status == "initiated",
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
        if event is None and candidates:
            conditions = [
                HeatAlertWebhookEvent.outbound_call_uuid.in_(candidates),
                HeatAlertWebhookEvent.voice_session_id == session_id,
            ]
            if process_id:
                conditions.append(HeatAlertWebhookEvent.voice_process_id == process_id)
            event = (
                await session.execute(
                    select(HeatAlertWebhookEvent)
                    .where(
                        HeatAlertWebhookEvent.called.is_(True),
                        HeatAlertWebhookEvent.call_status == "initiated",
                        or_(*conditions),
                    )
                    .order_by(HeatAlertWebhookEvent.received_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if event is None:
            return None
        event.voice_session_id = session_id
        if process_id:
            event.voice_process_id = process_id
        await session.commit()
        return HeatAlertCallContext(
            event_id=str(event.id),
            animal_number=event.pashuaadhar_no,
            tag_number=event.tag_no,
        )


async def mark_heat_alert_call_booked(event_id: str | None) -> bool:
    parsed = _uuid(event_id)
    if parsed is None or not webhook_db_configured():
        return False
    try:
        async with get_webhook_session() as session:
            result = await session.execute(
                update(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == parsed,
                    HeatAlertWebhookEvent.called.is_(True),
                )
                .values(call_booked=True)
            )
            await session.commit()
        return bool(result.rowcount)
    except Exception:
        logger.exception("Failed to persist heat-alert booking success")
        return False
