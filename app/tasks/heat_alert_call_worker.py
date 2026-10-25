"""Durable eligibility and outbound-call worker for heat-alert events."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from agents.tools.beckn.amul import authenticated_accounts, fetch_authenticated_farmers
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.farmer_ticket_summary_backend import get_farmer_ticket_summary_api
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryRequestModel
from app.config import get_config_value, settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent, OutboundCallDispatchState
from app.services.outbound_call import (
    OutboundCallAmbiguousFailure,
    OutboundCallConfigurationError,
    OutboundCallDefiniteFailure,
    initiate_outbound_call,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

_worker_task: Optional[asyncio.Task] = None
_LEASE = timedelta(minutes=2)

_DIGITS = {
    "0": "ઝીરો",
    "1": "વન",
    "2": "ટુ",
    "3": "થ્રી",
    "4": "ફોર",
    "5": "ફાઈવ",
    "6": "સિક્સ",
    "7": "સેવન",
    "8": "એઈટ",
    "9": "નાઈન",
    "૦": "ઝીરો",
    "૧": "વન",
    "૨": "ટુ",
    "૩": "થ્રી",
    "૪": "ફોર",
    "૫": "ફાઈવ",
    "૬": "સિક્સ",
    "૭": "સેવન",
    "૮": "એઈટ",
    "૯": "નાઈન",
}
_LETTERS = {
    "A": "એ", "B": "બી", "C": "સી", "D": "ડી", "E": "ઈ", "F": "એફ",
    "G": "જી", "H": "એચ", "I": "આઈ", "J": "જે", "K": "કે", "L": "એલ",
    "M": "એમ", "N": "એન", "O": "ઓ", "P": "પી", "Q": "ક્યુ", "R": "આર",
    "S": "એસ", "T": "ટી", "U": "યુ", "V": "વી", "W": "ડબલ્યુ",
    "X": "એક્સ", "Y": "વાય", "Z": "ઝેડ",
}


def speak_identifier(value: str | None) -> str:
    """Spell an identifier deterministically for Gujarati TTS."""
    words: list[str] = []
    for char in str(value or "").strip():
        if char in _DIGITS:
            words.append(_DIGITS[char])
        elif char.upper() in _LETTERS:
            words.append(_LETTERS[char.upper()])
        elif char.isalnum():
            words.append(char)
    return " ".join(words)


def heat_alert_opening_message(animal_number: str | None, tag_number: str | None) -> str:
    return (
        "નમસ્તે!\n"
        "હું અમુલ તરફથી સરલાબેન બોલું છું.\n"
        f"આપના ગાય/ભેંસ નંબર: {speak_identifier(animal_number)}\n"
        f"ટેગ નંબર : {speak_identifier(tag_number)}\n"
        "ગાય / ભેંસમાં ગરમી ના લક્ષણો ડિજિટલ બેલ્ટના માધ્યમિક નોંધાયેલ છે, "
        "તો શું હું આપના ગાય / ભેંસના માટે કૃત્રિમ બીજદાન માટેની ટિકિટ રજીસ્ટર કરું?"
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _set_claimed_event(
    event_id: uuid.UUID,
    lease_token: str,
    expected_status: str,
    **values,
) -> bool:
    """Update an event only while this worker still owns its current lease."""
    values.setdefault("call_updated_at", _now())
    async with get_webhook_session() as session:
        result = await session.execute(
            update(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.id == event_id,
                HeatAlertWebhookEvent.call_lease_token == lease_token,
                HeatAlertWebhookEvent.call_status == expected_status,
            )
            .values(**values)
        )
        await session.commit()
    return bool(result.rowcount)


async def _claim_due_event() -> HeatAlertWebhookEvent | None:
    now = _now()
    async with get_webhook_session() as session:
        # Eligibility work is safe to repeat. A crashed dispatch is not: the
        # provider may have received it before the process died.
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.call_status == "checking",
                HeatAlertWebhookEvent.call_lease_expires_at < now,
            )
            .values(
                call_status="scheduled",
                call_next_attempt_at=now,
                call_last_error="eligibility_lease_expired",
                call_lease_expires_at=None,
                call_lease_token=None,
                call_updated_at=now,
            )
        )
        await session.execute(
            update(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.call_status == "dispatching",
                HeatAlertWebhookEvent.call_lease_expires_at < now,
            )
            .values(
                call_status="ambiguous",
                call_last_error="dispatch_lease_expired",
                call_lease_expires_at=None,
                call_lease_token=None,
                call_updated_at=now,
            )
        )
        stmt = (
            select(HeatAlertWebhookEvent)
            .where(
                HeatAlertWebhookEvent.call_status.in_(("scheduled", "eligibility_retry")),
                HeatAlertWebhookEvent.call_due_at.is_not(None),
                HeatAlertWebhookEvent.call_due_at <= now,
                or_(
                    HeatAlertWebhookEvent.call_next_attempt_at.is_(None),
                    HeatAlertWebhookEvent.call_next_attempt_at <= now,
                ),
            )
            .order_by(
                HeatAlertWebhookEvent.call_next_attempt_at,
                HeatAlertWebhookEvent.received_at,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        event = (await session.execute(stmt)).scalar_one_or_none()
        if event is None:
            await session.commit()
            return None
        lease_token = str(uuid.uuid4())
        event.call_status = "checking"
        event.call_lease_expires_at = now + _LEASE
        event.call_lease_token = lease_token
        event.call_updated_at = now
        await session.commit()
        return event


def _retry_delay(attempt: int) -> timedelta:
    seconds = max(1, settings.heat_alert_call_retry_base_seconds) * (
        2 ** max(0, attempt - 1)
    )
    return timedelta(seconds=seconds)


async def _eligibility_retry(
    event_id: uuid.UUID, lease_token: str, category: str
) -> None:
    """Count only actual eligibility failures, never scheduling deferrals."""
    now = _now()
    async with get_webhook_session() as session:
        event = (
            await session.execute(
                select(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == event_id,
                    HeatAlertWebhookEvent.call_status == "checking",
                    HeatAlertWebhookEvent.call_lease_token == lease_token,
                    HeatAlertWebhookEvent.call_lease_expires_at > now,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if event is None:
            await session.commit()
            return
        event.call_eligibility_attempts += 1
        if event.call_eligibility_attempts >= max(
            1, settings.heat_alert_call_max_eligibility_attempts
        ):
            event.call_status = "eligibility_failed"
        else:
            event.call_status = "eligibility_retry"
            event.call_next_attempt_at = now + _retry_delay(
                event.call_eligibility_attempts
            )
        event.call_last_error = category
        event.call_lease_expires_at = None
        event.call_lease_token = None
        event.call_updated_at = now
        await session.commit()


async def _resolve_account(event: HeatAlertWebhookEvent):
    mobile = normalize_phone_to_mobile(event.farmer_contact or "")
    society = (event.society_code or "").strip()
    farmer = (event.farmer_code or "").strip()
    if not mobile:
        return "missing_account", None
    try:
        profiles = await fetch_authenticated_farmers(mobile)
    except Exception:
        return "profile_unavailable", None
    matches = authenticated_accounts(profiles or [])
    if society:
        matches = [account for account in matches if account.society_code == society]
    if farmer:
        matches = [account for account in matches if account.farmer_code == farmer]
    if len(matches) != 1:
        return "account_not_resolved", None
    return None, matches[0]


async def _reserve_provider_slot(
    event_id: uuid.UUID, lease_token: str, union_code: str
) -> int | None:
    now = _now()
    async with get_webhook_session() as session:
        await session.execute(
            pg_insert(OutboundCallDispatchState)
            .values(id=1, updated_at=now)
            .on_conflict_do_nothing(index_elements=["id"])
        )
        state = (
            await session.execute(
                select(OutboundCallDispatchState)
                .where(OutboundCallDispatchState.id == 1)
                .with_for_update()
            )
        ).scalar_one()
        event = (
            await session.execute(
                select(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == event_id,
                    HeatAlertWebhookEvent.call_status == "checking",
                    HeatAlertWebhookEvent.call_lease_token == lease_token,
                    HeatAlertWebhookEvent.call_lease_expires_at > now,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if event is None:
            await session.commit()
            return None
        if state.next_allowed_at and state.next_allowed_at > now:
            event.call_status = "scheduled"
            event.call_next_attempt_at = state.next_allowed_at
            event.call_lease_expires_at = None
            event.call_lease_token = None
            event.call_updated_at = now
            await session.commit()
            return None
        state.next_allowed_at = now + timedelta(
            seconds=max(1, settings.raya_outbound_call_min_interval_seconds)
        )
        state.updated_at = now
        event.call_status = "dispatching"
        event.call_dispatch_attempts += 1
        event.call_lease_expires_at = now + _LEASE
        event.resolved_union_code = union_code
        event.call_updated_at = now
        await session.commit()
        return event.call_dispatch_attempts


async def _dispatch(
    event: HeatAlertWebhookEvent, lease_token: str, union_code: str
) -> None:
    dispatch_attempt = await _reserve_provider_slot(
        event.id, lease_token, union_code
    )
    if dispatch_attempt is None:
        return
    agent_id = (settings.raya_heat_alert_agent_id or "").strip()
    args = {
        "opening_message": heat_alert_opening_message(event.pashuaadhar_no, event.tag_no),
        "animal_number": event.pashuaadhar_no or "",
        "tag_number": event.tag_no or "",
        "campaign": "heat_alert_ai_booking",
        "heat_alert_event_id": str(event.id),
    }
    try:
        accepted = await initiate_outbound_call(
            event.farmer_contact or "", agent_id, args
        )
    except OutboundCallDefiniteFailure as exc:
        if exc.retryable and dispatch_attempt < max(
            1, settings.heat_alert_call_max_dispatch_attempts
        ):
            await _set_claimed_event(
                event.id,
                lease_token,
                "dispatching",
                call_status="scheduled",
                call_last_error=exc.category,
                call_next_attempt_at=_now() + _retry_delay(dispatch_attempt),
                call_lease_expires_at=None,
                call_lease_token=None,
            )
        else:
            await _set_claimed_event(
                event.id,
                lease_token,
                "dispatching",
                call_status="dispatch_failed",
                call_last_error=exc.category,
                call_lease_expires_at=None,
                call_lease_token=None,
            )
        return
    except OutboundCallConfigurationError:
        await _set_claimed_event(
            event.id,
            lease_token,
            "dispatching",
            call_status="dispatch_failed",
            call_last_error="configuration",
            call_lease_expires_at=None,
            call_lease_token=None,
        )
        return
    except OutboundCallAmbiguousFailure as exc:
        await _set_claimed_event(
            event.id,
            lease_token,
            "dispatching",
            call_status="ambiguous",
            call_last_error=exc.category,
            call_lease_expires_at=None,
            call_lease_token=None,
        )
        return
    await _set_claimed_event(
        event.id,
        lease_token,
        "dispatching",
        called=True,
        call_status="initiated",
        outbound_call_uuid=accepted.call_uuid,
        call_last_error=None,
        call_lease_expires_at=None,
        call_lease_token=None,
    )


async def _process(event: HeatAlertWebhookEvent) -> None:
    lease_token = event.call_lease_token
    if not lease_token:
        return
    error, account = await _resolve_account(event)
    if error == "profile_unavailable":
        await _eligibility_retry(event.id, lease_token, error)
        return
    if error:
        await _set_claimed_event(
            event.id,
            lease_token,
            "checking",
            call_status="eligibility_failed",
            call_last_error=error,
            call_lease_expires_at=None,
            call_lease_token=None,
        )
        return
    token = str(get_config_value("PASHUGPT_TOKEN") or "").strip()
    if not token:
        await _eligibility_retry(
            event.id, lease_token, "ticket_provider_not_configured"
        )
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
        await _eligibility_retry(event.id, lease_token, "ticket_summary_unavailable")
        return
    if summary.pending_tickets:
        await _set_claimed_event(
            event.id,
            lease_token,
            "checking",
            call_status="suppressed_pending",
            call_last_error=None,
            resolved_union_code=account.union_code,
            call_lease_expires_at=None,
            call_lease_token=None,
        )
        return
    await _dispatch(event, lease_token, account.union_code)


async def _run_once() -> bool:
    event = await _claim_due_event()
    if event is None:
        return False
    await _process(event)
    return True


async def _run_loop() -> None:
    interval = max(1, settings.heat_alert_call_worker_interval_seconds)
    logger.info("Heat-alert call worker started interval_seconds=%s", interval)
    while True:
        try:
            worked = await _run_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Heat-alert call worker iteration failed")
            worked = False
        if not worked:
            await asyncio.sleep(interval)


async def start_heat_alert_call_worker() -> None:
    global _worker_task
    if not webhook_db_configured():
        logger.info("Heat-alert call worker not started: WEBHOOK_DB_URL not configured")
        return
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_run_loop())


async def stop_heat_alert_call_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    finally:
        _worker_task = None
        logger.info("Heat-alert call worker stopped")
