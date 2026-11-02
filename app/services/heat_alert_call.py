"""The collar heat-alert call: one use of app.services.outbound_call.

A set time after an alert is stored (HEAT_ALERT_CALL_DELAY_SECONDS, ten
minutes) Sarlaben rings the farmer with the heat-alert script and offers an AI
booking. The row's ``called`` flag is the claim: it flips to true before the
dial, inside the same statement that checks the alert is still worth a call,
so neither a retry nor a second replica recovering the same row can ring twice.
``call_error`` says why a claimed row was not rung, or how the dial failed.

Dials go out one at a time on the Raya key, RAYA_OUTBOUND_CALL_GAP_SECONDS
apart across all replicas, so the 06:00 batch of night alerts does not trip
Raya's rate limit. A dial Raya refuses is tried again later; an unanswered call
(Raya outcome Failure) is rung again HEAT_ALERT_CALL_RETRIES times,
HEAT_ALERT_CALL_RETRY_AFTER_SECONDS apart, while the AI window is open. A row
whose call is settled keeps ``called`` true, so a restart never rings it again.

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
from helpers.gujarati_numbers import tag_to_gujarati
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

SCRIPT_PROMPT = "heat_alert_call_script_gu"
# Tries of one dial while Raya refuses it (429, unreachable), waiting retry_after, then twice that, ...
_RATE_LIMIT_TRIES = 4
# A dial still refused after that is tried again this much later.
_REFUSED_RETRY = timedelta(minutes=10)
# Postgres advisory lock held by the replica whose turn it is to dial.
_DIAL_LOCK = 4_204_861_330
_DIAL_LOCK_POLL_S = 2.0
# Raya's outcome is read this long after the dial, then every _OUTCOME_RECHECK while Pending.
_OUTCOME_FIRST_CHECK = timedelta(minutes=3)
_OUTCOME_RECHECK = timedelta(minutes=2)
_OUTCOME_CHECKS = 8
# A farmer answers within minutes of the dial; the row itself lives 24 h.
_ANSWER_WINDOW = timedelta(hours=3)
_SESSION_TTL = 4 * 60 * 60
# The lookup runs on the farmer's first reply: past this the agent asks instead.
_PROFILE_TIMEOUT_S = 3

# Identifiers are read out character by character, digits as Gujarati words
# ("S123" -> "એસ એક બે ત્રણ", never "એકસો ત્રેવીસ"); the TTS would otherwise guess.
_SPOKEN = {
    **{d: tag_to_gujarati(d) for d in "0123456789"},
    **{g: tag_to_gujarati(d) for g, d in zip("૦૧૨૩૪૫૬૭૮૯", "0123456789")},
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
        "એ", "બી", "સી", "ડી", "ઈ", "એફ", "જી", "એચ", "આઈ", "જે", "કે", "એલ", "એમ",
        "એન", "ઓ", "પી", "ક્યુ", "આર", "એસ", "ટી", "યુ", "વી", "ડબલ્યુ", "એક્સ", "વાય", "ઝેડ",
    ])),
}


_SPECIES_GU = {"cow": "ગાય", "buffalo": "ભેંસ"}


def spoken(identifier: str | None) -> str:
    return " ".join(_SPOKEN.get(c.upper(), c) for c in (identifier or "").strip() if not c.isspace())


def enabled() -> bool:
    """No agent id, no calls: alerts are stored and messaged as before."""
    return bool(settings.raya_heat_alert_agent_id and settings.raya_outbound_call_api_key)


def agent_args(event: HeatAlertWebhookEvent, species: str = "") -> dict[str, str]:
    """What the agent gets: the rendered script plus the identifiers it names.

    The script names the animal "ગાય" or "ભેંસ" when its species is known, else "પશુ" (animal).
    """
    animal_no, tag_no = event.pashuaadhar_no or "", event.tag_no or ""
    context = {"animal_no": spoken(animal_no), "tag_no": spoken(tag_no), "species": _SPECIES_GU.get(species, "પશુ")}
    return {
        # One line: Raya drops the line breaks when it speaks the arg, running the words together.
        "script": " ".join(get_prompt(SCRIPT_PROMPT, context).split()),
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
        species = await _species(event, await _account(event))
    except Exception:
        logger.exception("Heat-alert species lookup failed id=%s", event_id)
        species = ""  # the script then says "પશુ"
    try:
        call_uuid = await _dial(event, species)
    except outbound_call.OutboundCallError as exc:
        reason = str(exc)[:64]
        logger.warning("Heat-alert call not placed id=%s reason=%s", event_id, reason)
        now = datetime.now(timezone.utc)
        if reason == "quiet_hours":
            await _ring_later(event_id, now, reason)
        elif exc.retry_after is not None:
            # Refused before any call existed (rate limit, unreachable): safe to dial again.
            await _ring_later(event_id, now + _REFUSED_RETRY, reason)
        else:
            # Ambiguous (the call may exist) or refused for good (bad number, 4xx):
            # the claim stays, so nothing rings this row again.
            await _record(event_id, call_error=reason)
        return
    logger.info("Heat-alert call placed id=%s call_uuid=%s attempt=%s", event_id, call_uuid, event.call_attempts + 1)
    if event.call_attempts + 1 <= settings.heat_alert_call_retries:
        _follow_up(event_id, call_uuid, _OUTCOME_FIRST_CHECK)


async def _ring_later(event_id: uuid.UUID, at: datetime, reason: str | None) -> None:
    """Release the claim with a new due time; recovery picks it up if this process ends first."""
    at = outbound_call.after_quiet_hours(at)
    await _record(event_id, called=False, call_due_at=at, call_error=reason)
    schedule(event_id, at)


def _follow_up(event_id: uuid.UUID, call_uuid: str, delay: timedelta, checks_left: int = _OUTCOME_CHECKS) -> None:
    outbound_call.schedule(
        datetime.now(timezone.utc) + delay,
        lambda: _check_outcome(event_id, call_uuid, checks_left),
        name=f"heat-alert-outcome:{event_id}",
    )


async def _check_outcome(event_id: uuid.UUID, call_uuid: str, checks_left: int) -> None:
    """Read how the call ended; ring again if nobody answered and a retry is left."""
    outcome = await outbound_call.call_outcome(call_uuid)
    if outcome in (None, "Pending"):
        if checks_left > 1:
            _follow_up(event_id, call_uuid, _OUTCOME_RECHECK, checks_left - 1)
        else:
            logger.warning("Heat-alert call outcome unknown id=%s call_uuid=%s", event_id, call_uuid)
        return
    async with get_webhook_session() as session:
        # Once per call: a second replica reading the same outcome later must not
        # release the claim of a retry that is already being dialled.
        event = (
            await session.execute(
                update(HeatAlertWebhookEvent)
                .where(
                    HeatAlertWebhookEvent.id == event_id,
                    HeatAlertWebhookEvent.outbound_call_uuid == call_uuid,
                    HeatAlertWebhookEvent.call_outcome.is_(None),
                )
                .values(call_outcome=outcome)
                .returning(HeatAlertWebhookEvent)
            )
        ).scalar_one_or_none()
        await session.commit()
    if event is None or outcome != "Failure":
        return
    retry_at = outbound_call.after_quiet_hours(
        datetime.now(timezone.utc) + timedelta(seconds=settings.heat_alert_call_retry_after_seconds)
    )
    if event.call_booked:
        return
    if event.ai_window_end_at is not None and retry_at >= event.ai_window_end_at:
        logger.info("Heat-alert call unanswered, AI window closes before a retry id=%s", event_id)
        return
    logger.info("Heat-alert call unanswered, ringing again id=%s at=%s", event_id, retry_at.isoformat())
    await _ring_later(event_id, retry_at, None)


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


async def _dial(event: HeatAlertWebhookEvent, species: str = "") -> str:
    """Place the call when it is this replica's turn, and record it in the same transaction.

    The advisory lock makes dials on the Raya key go one at a time across
    replicas; inside it, the call waits until RAYA_OUTBOUND_CALL_GAP_SECONDS
    have passed since the last dial anywhere. The UUID is committed as the lock
    is released, before the farmer can answer.
    """
    args = agent_args(event, species)
    async with get_webhook_session() as session:
        # A try-lock loop: a blocking lock wait would outlast the 10 s statement timeout.
        while not (await session.execute(select(func.pg_try_advisory_xact_lock(_DIAL_LOCK)))).scalar():
            await asyncio.sleep(_DIAL_LOCK_POLL_S)
        dial = update(HeatAlertWebhookEvent).where(HeatAlertWebhookEvent.id == event.id)
        try:
            for attempt in range(1, _RATE_LIMIT_TRIES + 1):
                last = (await session.execute(select(func.max(HeatAlertWebhookEvent.call_dialled_at)))).scalar()
                if last is not None:
                    wait = settings.raya_outbound_call_gap_seconds - (datetime.now(timezone.utc) - last).total_seconds()
                    await asyncio.sleep(max(0.0, wait))
                try:
                    call_uuid = await outbound_call.initiate_call(
                        settings.raya_heat_alert_agent_id, event.farmer_contact, args
                    )
                except outbound_call.OutboundCallError as exc:
                    if exc.retry_after is None or attempt == _RATE_LIMIT_TRIES:
                        raise
                    await asyncio.sleep(exc.retry_after * 2 ** (attempt - 1))
                    continue
                await session.execute(
                    dial.values(
                        outbound_call_uuid=call_uuid,
                        call_attempts=event.call_attempts + 1,
                        call_outcome=None,
                        call_error=None,
                    )
                )
                return call_uuid
            raise AssertionError("unreachable")
        finally:
            # Stamped after the request, refused or not: the gap counts from when Raya last heard from us.
            await session.execute(dial.values(call_dialled_at=func.clock_timestamp()))
            await session.commit()


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
    species = await _species(event, account)
    call = HeatAlertCall(
        event_id=str(event.id),
        alert_id=event.alert_id,
        animal_no=event.pashuaadhar_no or "",
        tag_no=event.tag_no or "",
        union_code=account.union_code if account else "",
        society_code=account.society_code if account else "",
        farmer_code=account.farmer_code if account else "",
        script=agent_args(event, species)["script"],
        species=species,
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


async def owed() -> tuple[list[HeatAlertWebhookEvent], list[HeatAlertWebhookEvent]]:
    """Rows still owed a call, and placed calls whose outcome may lead to a retry.

    Exactly what ``recover`` schedules; scripts/heat_alert_call_dry_run.py
    prints it to show what a restart would ring.
    """
    now = datetime.now(timezone.utc)
    window_open = or_(HeatAlertWebhookEvent.ai_window_end_at.is_(None), HeatAlertWebhookEvent.ai_window_end_at > now)
    async with get_webhook_session() as session:
        due = (
            await session.execute(
                select(HeatAlertWebhookEvent).where(
                    HeatAlertWebhookEvent.call_due_at.is_not(None),
                    HeatAlertWebhookEvent.called.is_(False),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                    window_open,
                )
            )
        ).scalars().all()
        awaiting = (
            await session.execute(
                select(HeatAlertWebhookEvent).where(
                    HeatAlertWebhookEvent.called.is_(True),
                    HeatAlertWebhookEvent.call_booked.is_(False),
                    HeatAlertWebhookEvent.outbound_call_uuid.is_not(None),
                    HeatAlertWebhookEvent.call_outcome.is_(None),
                    HeatAlertWebhookEvent.call_attempts <= settings.heat_alert_call_retries,
                    window_open,
                )
            )
        ).scalars().all()
    return list(due), list(awaiting)


async def recover() -> int:
    """Reschedule every call still due, and the outcome reads of calls that may need a retry.

    Runs at startup on each replica; the claim dedupes. Rows whose call is
    settled keep ``called`` true and are not touched.
    """
    if not (webhook_db_configured() and enabled()):
        return 0
    due, awaiting = await owed()
    for event in due:
        schedule(event.id, event.call_due_at)
    for event in awaiting:
        _follow_up(event.id, event.outbound_call_uuid, timedelta(0))
    logger.info("Heat-alert calls rescheduled count=%s outcome_checks=%s", len(due), len(awaiting))
    return len(due)
