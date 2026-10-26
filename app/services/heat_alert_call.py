"""Schedule one eligible outbound call for a stored heat-alert webhook."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, update

from agents.tools.beckn.amul import authenticated_accounts, fetch_authenticated_farmers
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.farmer_ticket_summary_backend import get_farmer_ticket_summary_api
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryRequestModel
from app.config import get_config_value, settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent
from app.services import outbound_call
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

SCRIPT_PROMPT = "heat_alert_call_script_gu"
_PROVIDER_ATTEMPTS = 3
_ELIGIBILITY_ATTEMPTS = 3
_ELIGIBILITY_RETRY_BASE_SECONDS = 60


def enabled() -> bool:
    """No agent or API key means alerts are stored and messaged without calls."""
    return bool(settings.heat_alert_call_agent_id and settings.raya_call_api_key)


def agent_args(event: HeatAlertWebhookEvent) -> dict[str, str]:
    """Arguments consumed by the configured Raya heat-alert agent."""
    animal_no, tag_no = event.pashuaadhar_no or "", event.tag_no or ""
    return {
        "script": get_prompt(
            SCRIPT_PROMPT, {"animal_no": animal_no, "tag_no": tag_no}
        ).strip(),
        "animal_no": animal_no,
        "tag_no": tag_no,
        "alert_id": event.alert_id,
        "heat_alert_event_id": str(event.id),
    }


def schedule(event_id: uuid.UUID, due_at: datetime) -> None:
    outbound_call.schedule(
        due_at,
        lambda: call_farmer(event_id),
        name=f"heat-alert-call:{event_id}",
    )


async def _record(event_id: uuid.UUID, **values) -> None:
    values.setdefault("call_updated_at", datetime.now(timezone.utc))
    async with get_webhook_session() as session:
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(HeatAlertWebhookEvent.id == event_id)
            .values(**values)
        )
        await session.commit()


async def _claim(event_id: uuid.UUID) -> HeatAlertWebhookEvent | None:
    now = datetime.now(timezone.utc)
    async with get_webhook_session() as session:
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.id == event_id,
                HeatAlertWebhookEvent.call_status == "scheduled",
                HeatAlertWebhookEvent.ai_window_end_at.is_not(None),
                HeatAlertWebhookEvent.ai_window_end_at <= now,
            )
            .values(call_status="window_closed", call_updated_at=now)
        )
        event = (
            await session.execute(
                update(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == event_id,
                    HeatAlertWebhookEvent.call_status == "scheduled",
                    HeatAlertWebhookEvent.called.is_(False),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                    HeatAlertWebhookEvent.call_due_at <= now,
                    or_(
                        HeatAlertWebhookEvent.ai_window_end_at.is_(None),
                        HeatAlertWebhookEvent.ai_window_end_at > now,
                    ),
                )
                .values(
                    call_status="checking",
                    call_attempts=HeatAlertWebhookEvent.call_attempts + 1,
                    call_updated_at=now,
                )
                .returning(HeatAlertWebhookEvent)
            )
        ).scalar_one_or_none()
        await session.commit()
    return event


async def _resolve_account(event: HeatAlertWebhookEvent):
    mobile = normalize_phone_to_mobile(event.farmer_contact or "")
    if not mobile:
        return "missing_account", None
    try:
        profiles = await fetch_authenticated_farmers(mobile)
    except Exception:
        return "profile_unavailable", None
    matches = authenticated_accounts(profiles or [])
    for field, raw_value in (
        ("union_code", event.union_code),
        ("society_code", event.society_code),
        ("farmer_code", event.farmer_code),
    ):
        value = (raw_value or "").strip()
        if value:
            matches = [account for account in matches if getattr(account, field) == value]
    if len(matches) != 1:
        return "account_not_resolved", None
    return None, matches[0]


async def _retry_eligibility(event: HeatAlertWebhookEvent, reason: str) -> None:
    if event.call_attempts >= _ELIGIBILITY_ATTEMPTS:
        await _record(event.id, call_status="eligibility_failed", call_error=reason)
        return
    due_at = outbound_call.after_quiet_hours(
        datetime.now(timezone.utc)
        + timedelta(
            seconds=_ELIGIBILITY_RETRY_BASE_SECONDS
            * (2 ** max(0, event.call_attempts - 1))
        )
    )
    await _record(
        event.id,
        call_status="scheduled",
        call_due_at=due_at,
        call_error=reason,
    )
    schedule(event.id, due_at)


async def _dial(event: HeatAlertWebhookEvent) -> str:
    for attempt in range(1, _PROVIDER_ATTEMPTS + 1):
        try:
            return await outbound_call.initiate_call(
                settings.heat_alert_call_agent_id,
                event.farmer_contact,
                agent_args(event),
            )
        except outbound_call.OutboundCallError as exc:
            if exc.ambiguous or exc.retry_after is None or attempt == _PROVIDER_ATTEMPTS:
                raise
            await asyncio.sleep(exc.retry_after)
    raise AssertionError("unreachable")


async def _mark_dispatching(event_id: uuid.UUID) -> bool:
    """Recheck booking/window gates immediately before the external write."""
    now = datetime.now(timezone.utc)
    async with get_webhook_session() as session:
        result = await session.execute(
            update(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.id == event_id,
                HeatAlertWebhookEvent.call_status == "checking",
                HeatAlertWebhookEvent.call_booked.is_(False),
                or_(
                    HeatAlertWebhookEvent.ai_window_end_at.is_(None),
                    HeatAlertWebhookEvent.ai_window_end_at > now,
                ),
            )
            .values(call_status="dispatching", call_error=None, call_updated_at=now)
            .returning(HeatAlertWebhookEvent.id)
        )
        await session.commit()
    return result.scalar_one_or_none() is not None


async def call_farmer(event_id: uuid.UUID) -> None:
    event = await _claim(event_id)
    if event is None:
        return

    error, account = await _resolve_account(event)
    if error == "profile_unavailable":
        await _retry_eligibility(event, error)
        return
    if error:
        await _record(event.id, call_status="eligibility_failed", call_error=error)
        return

    token = str(get_config_value("PASHUGPT_TOKEN") or "").strip()
    if not token:
        await _retry_eligibility(event, "ticket_provider_not_configured")
        return
    summary = await get_farmer_ticket_summary_api(
        FarmerTicketSummaryRequestModel(
            unionCode=account.union_code,
            societyCode=account.society_code,
            farmerCode=account.farmer_code,
        ),
        token,
    )
    if summary is None:
        await _retry_eligibility(event, "ticket_summary_unavailable")
        return
    if summary.pending_tickets:
        await _record(event.id, call_status="suppressed_pending", call_error=None)
        return

    if not await _mark_dispatching(event.id):
        return
    try:
        call_uuid = await _dial(event)
    except outbound_call.OutboundCallError as exc:
        await _record(
            event.id,
            call_status="ambiguous" if exc.ambiguous else "dispatch_failed",
            call_error=str(exc)[:64],
        )
        return
    await _record(
        event.id,
        called=True,
        call_status="initiated",
        outbound_call_uuid=call_uuid,
        call_error=None,
    )
    logger.info("Heat-alert outbound call accepted id=%s", event.id)


async def recover() -> int:
    """Recover safe queued work; an interrupted dispatch remains ambiguous."""
    if not (webhook_db_configured() and enabled()):
        return 0
    now = datetime.now(timezone.utc)
    async with get_webhook_session() as session:
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(HeatAlertWebhookEvent.call_status == "checking")
            .values(call_status="scheduled", call_due_at=now, call_updated_at=now)
        )
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(HeatAlertWebhookEvent.call_status == "dispatching")
            .values(
                call_status="ambiguous",
                call_error="interrupted_dispatch",
                call_updated_at=now,
            )
        )
        rows = (
            await session.execute(
                select(HeatAlertWebhookEvent.id, HeatAlertWebhookEvent.call_due_at).where(
                    HeatAlertWebhookEvent.call_status == "scheduled",
                    HeatAlertWebhookEvent.call_due_at.is_not(None),
                    HeatAlertWebhookEvent.called.is_(False),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                )
            )
        ).all()
        await session.commit()
    for event_id, due_at in rows:
        schedule(event_id, due_at)
    logger.info("Heat-alert calls recovered count=%s", len(rows))
    return len(rows)
