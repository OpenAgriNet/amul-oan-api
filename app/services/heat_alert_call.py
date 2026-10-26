"""The collar heat-alert call: one use of app.services.outbound_call.

A set time after an alert is stored (HEAT_ALERT_CALL_DELAY_SECONDS, ten
minutes) Sarlaben rings the farmer with the heat-alert script and offers an AI
booking. The row's ``called`` flag is the claim: it flips to true before the
dial, inside the same statement that checks the alert is still worth a call,
so neither a retry nor a second replica recovering the same row can ring twice.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy import or_, select, update

from app.config import settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent
from app.services import outbound_call
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

SCRIPT_PROMPT = "heat_alert_call_script_gu"
_ATTEMPTS = 3


def enabled() -> bool:
    """No agent id, no calls: alerts are stored and messaged as before."""
    return bool(settings.heat_alert_call_agent_id and settings.raya_call_api_key)


def agent_args(event: HeatAlertWebhookEvent) -> dict[str, str]:
    """What the agent gets: the rendered script plus the identifiers it names."""
    animal_no, tag_no = event.pashuaadhar_no or "", event.tag_no or ""
    return {
        "script": get_prompt(SCRIPT_PROMPT, {"animal_no": animal_no, "tag_no": tag_no}).strip(),
        "animal_no": animal_no,
        "tag_no": tag_no,
        "alert_id": event.alert_id,
    }


def schedule(event_id: uuid.UUID, due_at: datetime) -> None:
    outbound_call.schedule(due_at, lambda: call_farmer(event_id), name=f"heat-alert-call:{event_id}")


async def call_farmer(event_id: uuid.UUID) -> None:
    event = await _claim(event_id)
    if event is None:
        return  # already called, booked another way, or the AI window has closed
    try:
        call_uuid = await _dial(event)
    except outbound_call.OutboundCallError as exc:
        logger.warning("Heat-alert call not placed id=%s reason=%s", event_id, exc)
        await _record(event_id, call_error=str(exc)[:64])
        return
    logger.info("Heat-alert call placed id=%s call_uuid=%s", event_id, call_uuid)
    await _record(event_id, outbound_call_uuid=call_uuid)


async def _dial(event: HeatAlertWebhookEvent) -> str:
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return await outbound_call.initiate_call(
                settings.heat_alert_call_agent_id, event.farmer_contact, agent_args(event)
            )
        except outbound_call.OutboundCallError as exc:
            if exc.retry_after is None or attempt == _ATTEMPTS:
                raise
            await asyncio.sleep(exc.retry_after)
    raise AssertionError("unreachable")


async def _claim(event_id: uuid.UUID) -> HeatAlertWebhookEvent | None:
    now = datetime.now(timezone.utc)
    async with get_webhook_session() as session:
        event = (
            await session.execute(
                update(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == event_id,
                    HeatAlertWebhookEvent.called.is_(False),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                    or_(
                        HeatAlertWebhookEvent.ai_window_end_at.is_(None),
                        HeatAlertWebhookEvent.ai_window_end_at > now,
                    ),
                )
                .values(called=True)
                .returning(HeatAlertWebhookEvent)
            )
        ).scalar_one_or_none()
        await session.commit()
    return event


async def _record(event_id: uuid.UUID, **values) -> None:
    async with get_webhook_session() as session:
        await session.execute(
            update(HeatAlertWebhookEvent).where(HeatAlertWebhookEvent.id == event_id).values(**values)
        )
        await session.commit()


async def recover() -> int:
    """Reschedule every call still due. Runs at startup on each replica; the claim dedupes."""
    if not (webhook_db_configured() and enabled()):
        return 0
    now = datetime.now(timezone.utc)
    async with get_webhook_session() as session:
        rows = (
            await session.execute(
                select(HeatAlertWebhookEvent.id, HeatAlertWebhookEvent.call_due_at).where(
                    HeatAlertWebhookEvent.call_due_at.is_not(None),
                    HeatAlertWebhookEvent.called.is_(False),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                    or_(
                        HeatAlertWebhookEvent.ai_window_end_at.is_(None),
                        HeatAlertWebhookEvent.ai_window_end_at > now,
                    ),
                )
            )
        ).all()
    for event_id, due_at in rows:
        schedule(event_id, due_at)
    logger.info("Heat-alert calls rescheduled count=%s", len(rows))
    return len(rows)
