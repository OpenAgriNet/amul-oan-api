"""Outbound-call consent handling (milk-details readout) for voice calls.

Raya stamps ``call_type=outbound`` on requests for calls WE placed, and Raya —
not this service — speaks the opening line ("may I read out your last 7 days of
milk?"). Our work starts at the farmer's ANSWER: the first outbound turn already
carries it, and it is classified by :mod:`app.voice.outbound_consent` into
affirmative / negative / other. This module never emits an intro; doing so would
give the farmer two.

Everything caller-facing here is pinned Gujarati, served verbatim. It never goes
through TranslateGemma: these lines carry a phone number, the brand name, and the
farmer-facing framing of a data readout, all of which the translation sandwich has
historically drifted on.

State lives in Redis under the session (not in message history) so the stage
survives history trimming and so an inbound call can never accidentally read it.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pytz

from agents.deps import FarmerAccount
from app.config import settings
from app.utils import get_cache, set_cache
from helpers.utils import get_logger

logger = get_logger(__name__)

CALL_TYPE_INBOUND = "inbound"
CALL_TYPE_OUTBOUND = "outbound"

# Session stages for the outbound opener.
# Legacy: written only by the removed in-app intro. Sessions carrying it were
# mid-call across that deploy; kept so their next turn still routes to the
# consent gate. Nothing sets it any more.
STAGE_INTRO_SENT = "intro_sent"
STAGE_RESOLVED = "resolved"          # consent already handled; call is a normal conversation

_STATE_SUFFIX = "outbound_state"
_MILK_SUFFIX = "outbound_milk_summary"
_STATE_TTL = 60 * 60 * 4  # a call never outlives this; keeps stale state out of Redis


# ── Scripted lines (verbatim, never translated) ───────────────────────────────
# Source: "Outbound call script for previously interacted (non-returning
# registered callers)". The English variants exist only for message history and
# for en-target test calls — the caller hears the Gujarati.
#
# NOTE: the script's opening line lives in RAYA, not here. Everything below is
# said only AFTER the farmer has replied to it.
#
# Digits are spelled as Gujarati words on purpose: clean_output_by_language()
# strips every non-Gujarati-block character for gu, so an ASCII digit would be
# deleted and the caller would hear a gap where the number should be.

OUTBOUND_DECLINE_FAREWELL = {
    "gu": (
        "કંઈ વાંધો નહીં, હું આ કોલ ડિસ્કનેક્ટ કરું છું. "
        "તમે પશુપાલન વિષે કોઈ પણ માહિતી માટે ૦૮૦૩૫૪૫૩૫૪૫ પર કોલ કે વૉટ્સએપ કરી શકો છો. "
        "તમારો દિવસ શુભ રહે!"
    ),
    "en": (
        "That is no problem, I will disconnect this call. "
        "For any information about animal husbandry you can call or WhatsApp 080-35453545. "
        "Have a good day!"
    ),
}

# Said when the farmer agrees but we have nothing to read out (no accounts on the
# number, or the upstream lookup failed). Better than reading a failure message
# aloud, and it leaves the call open for whatever they actually need.
OUTBOUND_NO_DATA = {
    "gu": (
        "માફ કરશો, અત્યારે તમારા દૂધની વિગતો મળી શકતી નથી. "
        "પશુપાલન વિષે તમારે કંઈ પૂછવું હોય તો જણાવો."
    ),
    "en": (
        "Sorry, I am not able to fetch your milk details right now. "
        "If you have any question about animal husbandry, please tell me."
    ),
}


def is_outbound(call_type: Optional[str]) -> bool:
    return (call_type or CALL_TYPE_INBOUND).strip().lower() == CALL_TYPE_OUTBOUND


def normalize_call_type(call_type: Optional[str]) -> str:
    return CALL_TYPE_OUTBOUND if is_outbound(call_type) else CALL_TYPE_INBOUND


def milk_window(days: int = 7) -> tuple[str, str]:
    """(fromdate, todate) ISO strings for the trailing ``days``-day window, IST.

    Inclusive of today, so 7 days is today plus the previous six.
    """
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    start = today - timedelta(days=max(1, days) - 1)
    return start.isoformat(), today.isoformat()


# ── Session state ─────────────────────────────────────────────────────────────

def _state_key(session_id: str) -> str:
    return f"{session_id}_{_STATE_SUFFIX}"


def _milk_key(session_id: str) -> str:
    return f"{session_id}_{_MILK_SUFFIX}"


async def set_stage(session_id: str, stage: str) -> None:
    """Record the outbound stage for this session. Best-effort."""
    try:
        await set_cache(_state_key(session_id), {"stage": stage}, ttl=_STATE_TTL)
    except Exception as e:  # pragma: no cover - state is an optimisation, never fatal
        logger.warning("Outbound stage write failed - session_id=%s stage=%s error=%s", session_id, stage, e)


async def get_stage(session_id: str) -> Optional[str]:
    """Return the recorded stage, or None when this is not a staged outbound call."""
    try:
        state = await get_cache(_state_key(session_id))
    except Exception as e:  # pragma: no cover
        logger.warning("Outbound stage read failed - session_id=%s error=%s", session_id, e)
        return None
    if isinstance(state, dict):
        stage = state.get("stage")
        return stage if isinstance(stage, str) else None
    return None


# ── Milk prefetch ─────────────────────────────────────────────────────────────
# Strong refs to in-flight prefetch tasks: asyncio only holds a weak reference, so
# without this a prefetch can be garbage-collected mid-flight.
_prefetch_tasks: set[asyncio.Task] = set()


async def prefetch_milk_summary(session_id: str, accounts: list[FarmerAccount]) -> None:
    """Fetch and cache the 7-day milk summary for this session. Never raises.

    Runs while the caller is still hearing the intro line. This is the slowest
    tool in the stack (20s upstream timeouts); doing it during the ~4s the farmer
    takes to answer "હા" is what keeps the reply off the 60s nginx cut that drops
    live calls.
    """
    from agents.voice.tools.milk_collection import fetch_milk_summary_for_accounts

    if not accounts:
        logger.info("Outbound milk prefetch skipped (no accounts) - session_id=%s", session_id)
        return

    fromdate, todate = milk_window(settings.outbound_milk_window_days)
    try:
        summary = await asyncio.wait_for(
            fetch_milk_summary_for_accounts(accounts, fromdate, todate, session_id=session_id),
            timeout=settings.outbound_milk_prefetch_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Outbound milk prefetch timed out - session_id=%s timeout=%.1fs",
            session_id,
            settings.outbound_milk_prefetch_timeout_seconds,
        )
        return
    except Exception as e:
        logger.warning("Outbound milk prefetch failed - session_id=%s error=%s", session_id, e)
        return

    try:
        await set_cache(_milk_key(session_id), {"summary": summary}, ttl=_STATE_TTL)
        logger.info(
            "Outbound milk prefetch cached - session_id=%s accounts=%s chars=%s window=%s..%s",
            session_id, len(accounts), len(summary), fromdate, todate,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Outbound milk prefetch cache write failed - session_id=%s error=%s", session_id, e)


def spawn(coro, *, label: str) -> None:
    """Run a background coroutine that must outlive the current streaming turn.

    The intro turn returns as soon as the line is spoken, so the prefetch has to
    survive it. A bare ``create_task`` is not enough — asyncio holds only a weak
    reference, so the task can be collected mid-flight.
    """
    async def _guarded():
        try:
            await coro
        except Exception as e:  # pragma: no cover - background work is best-effort
            logger.warning("Outbound background task failed - label=%s error=%s", label, e)

    task = asyncio.create_task(_guarded())
    _prefetch_tasks.add(task)
    task.add_done_callback(_prefetch_tasks.discard)


async def get_prefetched_milk_summary(session_id: str) -> Optional[str]:
    try:
        cached = await get_cache(_milk_key(session_id))
    except Exception as e:  # pragma: no cover
        logger.warning("Outbound milk summary read failed - session_id=%s error=%s", session_id, e)
        return None
    if isinstance(cached, dict):
        summary = cached.get("summary")
        return summary if isinstance(summary, str) and summary.strip() else None
    return None


def milk_fetch_hint(fromdate: str, todate: str) -> str:
    """Hint used when the prefetch has not landed (cold cache, slow upstream).

    The agent fetches the window itself with the tool. Slower than the prefetch
    path, but the dates are pinned here so it cannot pick the wrong window.
    """
    return (
        "- Outbound call: the farmer has just agreed to hear their milk deposit details. "
        f"Call get_farmer_milk_collection_details for fromdate {fromdate} to todate {todate}, "
        "then read the totals out conversationally — total quantity and total amount first, "
        "then anything notable. Do not list every record one by one. End by asking if they "
        "need anything else about animal husbandry."
    )


def milk_answer_hint(summary: str) -> str:
    """Per-turn hint appended to the agent's input on an affirmative reply.

    The data is handed to the agent rather than re-fetched by it: the farmer has
    already consented to exactly this readout, so a tool round-trip here would add
    latency and give the model a chance to call the wrong window.
    """
    return (
        "- Outbound call: the farmer has just agreed to hear their milk deposit details "
        f"for the last {settings.outbound_milk_window_days} days. Read out the summary below "
        "conversationally: total quantity and total amount first, then anything notable. "
        "Do not list every record one by one. Do not call any tool to fetch this — it is "
        "already fetched. End by asking if they need anything else about animal husbandry.\n"
        f"Milk deposit summary:\n{summary}"
    )
