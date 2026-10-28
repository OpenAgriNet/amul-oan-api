"""The collar heat-alert call: one use of app.services.outbound_call.

A set time after an alert is stored (HEAT_ALERT_CALL_DELAY_SECONDS, ten
minutes) Sarlaben rings the farmer with the heat-alert script and offers an AI
booking. The row's ``called`` flag is the claim: it flips to true before the
dial, inside the same statement that checks the alert is still worth a call,
so neither a retry nor a second replica recovering the same row can ring twice.
``call_error`` says why a claimed row was not rung, or how the dial failed.

When the farmer answers, Raya sends the voice turn with ``session_id`` equal to
the call UUID it returned to us and ``call_type=outbound`` (seen 2026-10-26);
``context_for_call`` matches that to the call we placed, so the voice agent
knows which animal it is about and books for that account.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import func, or_, select, update

from agents.tools.beckn.amul import (
    AuthenticatedFarmerAccount,
    authenticated_accounts,
    fetch_animal_profile,
    fetch_authenticated_farmers,
)
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.farmer_cache import get_farmer_data_cached_only
from app.config import get_config_value, settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent
from app.services import outbound_call
from app.turn.types import HeatAlertCall
from app.utils import get_cache, set_cache
from app.voice.outbound import is_outbound
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

SCRIPT_PROMPT = "heat_alert_call_script_gu"
_ATTEMPTS = 3
# A farmer answers within minutes of the dial; the row itself lives 24 h.
_ANSWER_WINDOW = timedelta(hours=3)
_SESSION_TTL = 4 * 60 * 60
# The lookup runs on the farmer's first reply: past this the agent asks instead.
_PROFILE_TIMEOUT_S = 3

# Identifiers are read out character by character, as Amul's script does
# ("S1325" -> "એસ વન થ્રી ટુ ફાઈવ"); the TTS would otherwise guess.
_SPOKEN = {
    **dict(zip("0123456789", ["ઝીરો", "વન", "ટુ", "થ્રી", "ફોર", "ફાઈવ", "સિક્સ", "સેવન", "એઈટ", "નાઈન"])),
    **dict(zip("૦૧૨૩૪૫૬૭૮૯", ["ઝીરો", "વન", "ટુ", "થ્રી", "ફોર", "ફાઈવ", "સિક્સ", "સેવન", "એઈટ", "નાઈન"])),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
        "એ", "બી", "સી", "ડી", "ઈ", "એફ", "જી", "એચ", "આઈ", "જે", "કે", "એલ", "એમ",
        "એન", "ઓ", "પી", "ક્યુ", "આર", "એસ", "ટી", "યુ", "વી", "ડબલ્યુ", "એક્સ", "વાય", "ઝેડ",
    ])),
}


def spoken(identifier: str | None) -> str:
    return " ".join(_SPOKEN.get(c.upper(), c) for c in (identifier or "").strip() if not c.isspace())


def enabled() -> bool:
    """No agent id, no calls: alerts are stored and messaged as before."""
    return bool(settings.raya_heat_alert_agent_id and settings.raya_outbound_call_api_key)


def agent_args(event: HeatAlertWebhookEvent) -> dict[str, str]:
    """What the agent gets: the rendered script plus the identifiers it names."""
    animal_no, tag_no = event.pashuaadhar_no or "", event.tag_no or ""
    return {
        # One line: Raya drops the line breaks when it speaks the arg, running the words together.
        "script": " ".join(get_prompt(SCRIPT_PROMPT, {"animal_no": spoken(animal_no), "tag_no": spoken(tag_no)}).split()),
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
    skip = await _reason_not_to_call(event)
    if skip:
        logger.info("Heat-alert call skipped id=%s reason=%s", event_id, skip)
        await _record(event_id, call_error=skip)
        return
    try:
        call_uuid = await _dial(event)
    except outbound_call.OutboundCallError as exc:
        logger.warning("Heat-alert call not placed id=%s reason=%s", event_id, exc)
        # A definite refusal releases the claim; an ambiguous one keeps it, the call may exist.
        await _record(event_id, called=exc.ambiguous, call_error=str(exc)[:64])
        return
    logger.info("Heat-alert call placed id=%s call_uuid=%s", event_id, call_uuid)
    await _record(event_id, outbound_call_uuid=call_uuid)


async def _reason_not_to_call(event: HeatAlertWebhookEvent) -> str | None:
    only = _only_numbers()
    if only and outbound_call.mobile_number(event.farmer_contact) not in only:
        return "not_in_only_numbers"
    if await _has_pending_ticket(event):
        return "pending_ticket"
    return None


def _only_numbers() -> set[str]:
    """HEAT_ALERT_CALL_ONLY_NUMBERS, for a test phase: when set, no one else is rung."""
    return {n.strip() for n in settings.heat_alert_call_only_numbers.split(",") if n.strip()}


async def _has_pending_ticket(event: HeatAlertWebhookEvent) -> bool:
    """Amul Pashudhan lists a pending ticket for the alert's account.

    Unknown counts as none: the call goes ahead and the booking flow handles
    an already-booked animal.
    """
    token = str(get_config_value("PASHUGPT_TOKEN") or "").strip()
    try:
        account = await _account(event)
        if account is None or not token:
            logger.warning("Heat-alert ticket check skipped id=%s account=%s", event.id, account is not None)
            return False
        async with httpx.AsyncClient(timeout=settings.farmer_backend_http_timeout_seconds) as client:
            response = await client.get(
                f"{settings.amulpashudhan_base_url}/GetFarmerTicketSummary",
                params={
                    "unionCode": account.union_code,
                    "societyCode": account.society_code,
                    "farmerCode": account.farmer_code,
                },
                headers={"Authorization": f"Bearer {token}"},
            )
        data = response.json() if response.status_code == 200 else None
        if not isinstance(data, dict):
            logger.warning("Heat-alert ticket check unusable id=%s status=%s", event.id, response.status_code)
            return False
        return bool(data.get("pendingTickets"))
    except Exception:
        logger.exception("Heat-alert ticket check failed id=%s", event.id)
        return False


async def _account(event: HeatAlertWebhookEvent) -> AuthenticatedFarmerAccount | None:
    """The alert's account: its own codes, with the union filled from the farmer cache when the partner omits it."""
    union = (event.union_code or "").strip()
    society = (event.society_code or "").strip()
    farmer = (event.farmer_code or "").strip()
    if union and society and farmer:
        return AuthenticatedFarmerAccount(union_code=union, society_code=society, farmer_code=farmer)
    mobile = normalize_phone_to_mobile(event.farmer_contact or "")
    accounts = authenticated_accounts(await fetch_authenticated_farmers(mobile)) if mobile else []
    matches = [
        a for a in accounts
        if (not society or _same_code(a.society_code, society)) and (not farmer or _same_code(a.farmer_code, farmer))
    ]
    return matches[0] if len(matches) == 1 else None


def _same_code(a: str | None, b: str | None) -> bool:
    """Cowfit zero-pads codes PashuGPT does not (society "0958" vs "958")."""
    return (a or "").strip().lstrip("0") == (b or "").strip().lstrip("0")


async def _dial(event: HeatAlertWebhookEvent) -> str:
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return await outbound_call.initiate_call(
                settings.raya_heat_alert_agent_id, event.farmer_contact, agent_args(event)
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


def _session_key(session_id: str) -> str:
    return f"{session_id}_heat_alert_call"


async def context_for_call(session_id: str, mobile: str | None, call_type: str | None) -> HeatAlertCall | None:
    """The heat alert this voice session answers, if it is one of our calls.

    Raya's session id for a call we placed is the call UUID it returned, so the
    match is exact; the phone we rang in the last few hours is the fallback on an
    outbound turn. Kept on the session once found.
    """
    cached = await get_cache(_session_key(session_id))
    if isinstance(cached, dict) and cached.get("event_id"):
        return HeatAlertCall(**cached)
    if not webhook_db_configured():
        return None
    event = await _placed_call(session_id, mobile if is_outbound(call_type) else None)
    if event is None:
        return None
    account = await _account(event)
    call = HeatAlertCall(
        event_id=str(event.id),
        alert_id=event.alert_id,
        animal_no=event.pashuaadhar_no or "",
        tag_no=event.tag_no or "",
        union_code=account.union_code if account else "",
        society_code=account.society_code if account else "",
        farmer_code=account.farmer_code if account else "",
        script=agent_args(event)["script"],
        species=await _species(event, account),
    )
    await set_cache(_session_key(session_id), asdict(call), ttl=_SESSION_TTL)
    logger.info("Voice session is a heat-alert call session_id=%s event_id=%s", session_id, call.event_id)
    return call


async def _species(event: HeatAlertWebhookEvent, account: AuthenticatedFarmerAccount | None) -> str:
    """The alert animal's species, so the agent need not ask; "" when unknown.

    The webhook never carries it. The farmer cache has it only for farmers who
    have used chat or voice, so the animal profile is the fallback.
    """
    ids = [i.strip() for i in (event.pashuaadhar_no, event.tag_no) if i and i.strip()]
    if not ids:
        return ""
    try:
        mobile = normalize_phone_to_mobile(event.farmer_contact or "")
        envelope = await get_farmer_data_cached_only(mobile) if mobile else None
        for record in (envelope.farmers if envelope else None) or []:
            if account and record.farmerCode and not _same_code(record.farmerCode, account.farmer_code):
                continue
            for animal in record.animals or []:
                if str(animal.tagNumber or "").strip() in ids:
                    return _species_name(animal.animalType)
        # The Pashu Aadhaar only: a short collar tag ("1") could name another farmer's animal.
        if account and event.pashuaadhar_no:
            profile = await asyncio.wait_for(
                fetch_animal_profile(event.pashuaadhar_no.strip(), union_code=account.union_code),
                _PROFILE_TIMEOUT_S,
            )
            return _species_name(profile.animal_type if profile else None)
    except asyncio.TimeoutError:
        logger.warning("Heat-alert species lookup timed out id=%s", event.id)
    except Exception:
        logger.exception("Heat-alert species lookup failed id=%s", event.id)
    return ""


def _species_name(value: str | None) -> str:
    species = (value or "").strip().lower()
    return species if species in ("cow", "buffalo") else ""


async def _placed_call(session_id: str, mobile: str | None) -> HeatAlertWebhookEvent | None:
    placed = HeatAlertWebhookEvent.outbound_call_uuid.is_not(None)
    async with get_webhook_session() as session:
        event = (
            await session.execute(
                select(HeatAlertWebhookEvent).where(placed, HeatAlertWebhookEvent.outbound_call_uuid == session_id).limit(1)
            )
        ).scalar_one_or_none()
        if event is None and mobile:
            since = datetime.now(timezone.utc) - _ANSWER_WINDOW
            event = (
                await session.execute(
                    select(HeatAlertWebhookEvent)
                    .where(
                        placed,
                        func.right(func.regexp_replace(HeatAlertWebhookEvent.farmer_contact, r"\D", "", "g"), 10) == mobile,
                        HeatAlertWebhookEvent.call_due_at > since,
                    )
                    .order_by(HeatAlertWebhookEvent.call_due_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
    return event


async def mark_booked(event_id: str) -> None:
    """The farmer booked the AI visit on our call."""
    try:
        await _record(uuid.UUID(event_id), call_booked=True)
    except Exception:
        logger.exception("Could not mark heat-alert call booked event_id=%s", event_id)


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
