"""What a caller waiting on the model hears, and whether their request is still live.

From voice-oan-api at amul-dev ``3b19835``. The nudge helpers are
``agents/tools/common.py`` unchanged; the nudge task and the staleness check are
the ones in ``app/services/voice.py``.

The nudge fires on whichever comes first, the configured timeout since the
request started or the agent's first tool call, and is sent through the side
channel. ``run_turn`` stops it before the first thing the caller hears.
"""
from __future__ import annotations

import asyncio
import contextvars
import random
import time
from typing import Any, Dict, Optional

import httpx

from app.config import settings
from app.observability import start_observation
from app.turn.types import SideChannelEmission, SideChannelSender, StalenessCheck, Turn
from app.utils import is_session_request_owner, refresh_session_request_ownership
from helpers.utils import get_logger

logger = get_logger(__name__)

# ── Tool-call nudge signaling ───────────────────────────────────────────
# A per-request asyncio.Event stored in a ContextVar.  Any tool wrapper
# can call fire_tool_call_nudge() to tell the nudge task "a tool was
# invoked – send the hold message now instead of waiting for the timer".
_tool_call_nudge_event: contextvars.ContextVar[Optional[asyncio.Event]] = contextvars.ContextVar(
    "_tool_call_nudge_event", default=None
)


def set_tool_call_nudge_event(event: asyncio.Event) -> contextvars.Token:
    """Set the nudge event for the current async context (call once per request)."""
    return _tool_call_nudge_event.set(event)


def fire_tool_call_nudge() -> None:
    """Signal that a tool call has started – the nudge task should fire immediately."""
    event = _tool_call_nudge_event.get(None)
    if event is not None and not event.is_set():
        event.set()


_TIMEOUT_NUDGE_MESSAGES: dict[str, list[str]] = {
    "gu": [
        "હું જવાબ લઈને પાછી આવી રહી છું, કૃપા કરીને થોડી રાહ જુઓ.",
        "કૃપા કરીને થોડી રાહ જુઓ, હું ચકાસી રહી છું.",
    ],
    "en": [
        "I'm getting back to you, please wait.",
        "Please wait a moment while I check.",
    ],
}

_TOOL_NUDGE_MESSAGES: dict[str, list[str]] = {
    "gu": [
        "હું ચકાસી રહી છું, કૃપા કરીને થોડી રાહ જુઓ.",
        "કૃપા કરીને થોડી રાહ જુઓ, હું તપાસી રહી છું.",
    ],
    "en": [
        "I'm checking that now, please wait.",
        "One moment, please wait.",
    ],
}


def get_timeout_nudge_message(lang_code: str = "en") -> str:
    messages = _TIMEOUT_NUDGE_MESSAGES.get(lang_code, _TIMEOUT_NUDGE_MESSAGES["en"])
    return random.choice(messages)


def get_tool_nudge_message(lang_code: str = "en") -> str:
    messages = _TOOL_NUDGE_MESSAGES.get(lang_code, _TOOL_NUDGE_MESSAGES["en"])
    return random.choice(messages)


async def send_nudge_message_raya(message: str, session_id: str, process_id: str = None) -> None:
    """Send nudge message via RAYA API (async, non-blocking)."""
    try:
        nudge_url = settings.nudge_api_url
        payload: Dict[str, Any] = {
            "message": message,
            "session_id": session_id,
        }
        if process_id:
            payload["process_id"] = process_id

        logger.info(
            "Nudge API request sent; session_id=%s process_id=%s url=%s payload=%s",
            session_id,
            process_id,
            nudge_url,
            payload,
        )
        with start_observation(
            "send_nudge_message_raya",
            input={"session_id": session_id, "process_id": process_id, "message": message},
            metadata={"url": nudge_url, "component": "nudge_api"},
        ) as observation:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    nudge_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            if observation is not None:
                observation.update(
                    output={"status_code": response.status_code},
                    metadata={"url": nudge_url, "component": "nudge_api"},
                )
        response_body = response.text
        logger.info(
            "Nudge API response; session_id=%s process_id=%s status=%s body=%s",
            session_id,
            process_id,
            response.status_code,
            response_body,
        )
        if response.status_code == 200:
            logger.info(
                "Nudge message sent; session_id=%s process_id=%s api_status=%s response=%s",
                session_id,
                process_id,
                response.status_code,
                response_body,
            )
        else:
            logger.warning(
                "Nudge message API failed; session_id=%s process_id=%s api_status=%s response=%s",
                session_id,
                process_id,
                response.status_code,
                response_body,
            )
    except httpx.HTTPError as e:
        logger.error(
            "Error sending nudge message; session_id=%s process_id=%s error=%s",
            session_id,
            process_id,
            e,
        )
    except Exception as e:
        logger.error(
            "Unexpected error sending nudge message; session_id=%s process_id=%s error=%s",
            session_id,
            process_id,
            e,
        )


class RayaNudgeSender:
    """The telephony provider's nudge endpoint, as a ``SideChannelSender``."""

    def __init__(self, *, session_id: str, process_id: Optional[str] = None) -> None:
        self._session_id = session_id
        self._process_id = process_id

    async def send(self, emission: SideChannelEmission) -> None:
        await send_nudge_message_raya(emission.text, self._session_id, self._process_id)


class CallStaleness:
    """Voice's ``StalenessCheck``: the caller hung up, or a newer request now
    owns the session.

    Session ownership is refreshed at most every
    ``session_owner_refresh_interval_seconds``; whether this request still owns
    the session is read on every check.
    """

    def __init__(self, *, http_request, owner, session_id: str, process_id: Optional[str]) -> None:
        self._http_request = http_request
        self._owner = owner
        self._session_id = session_id
        self._process_id = process_id
        self._last_owner_refresh_at = 0.0

    async def __call__(self, reason: str) -> Optional[str]:
        if self._http_request is not None and await self._http_request.is_disconnected():
            logger.info(
                "Stopping request due to client disconnect - session_id=%s process_id=%s reason=%s",
                self._session_id,
                self._process_id,
                reason,
            )
            return "client_disconnected"

        owner = self._owner
        now = time.monotonic()
        if owner is not None and (
            self._last_owner_refresh_at == 0.0
            or now - self._last_owner_refresh_at >= settings.session_owner_refresh_interval_seconds
        ):
            refreshed = await refresh_session_request_ownership(owner)
            self._last_owner_refresh_at = now
            if not refreshed:
                logger.info(
                    "Stopping stale request after ownership lost during refresh - session_id=%s process_id=%s epoch=%s reason=%s",
                    self._session_id,
                    self._process_id,
                    owner.epoch,
                    reason,
                )
                return "stale_request"

        if owner is not None and not await is_session_request_owner(owner):
            logger.info(
                "Stopping stale request because a newer request owns the session - session_id=%s process_id=%s epoch=%s reason=%s",
                self._session_id,
                self._process_id,
                owner.epoch,
                reason,
            )
            return "stale_request"

        return None


class VoiceLiveness:
    """The telephony nudge for one call turn, as ``SurfaceProfile.liveness``.

    Building it arms the nudge, unless ``ENABLE_VOICE_NUDGES`` is off.
    """

    def __init__(
        self,
        turn: Turn,
        *,
        started_at: float,
        send: SideChannelSender,
        is_stale: Optional[StalenessCheck],
    ) -> None:
        self._session_id = turn.session_id
        self._process_id = turn.call.process_id if turn.call is not None else None
        self._task: Optional[asyncio.Task] = None
        if not settings.enable_voice_nudges:
            logger.info(
                "Voice nudges disabled by config; session_id=%s process_id=%s",
                self._session_id,
                self._process_id,
            )
            return
        self._tool_call = asyncio.Event()
        set_tool_call_nudge_event(self._tool_call)
        lang = (turn.target_lang or "gu").strip().lower()
        self._task = asyncio.create_task(self._nudge(lang, started_at, send, is_stale))
        logger.info(
            "Nudge initiated; session_id=%s process_id=%s",
            self._session_id,
            self._process_id,
        )

    async def stop(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _nudge(
        self,
        lang: str,
        started_at: float,
        send: SideChannelSender,
        is_stale: Optional[StalenessCheck],
    ) -> None:
        waits: tuple[asyncio.Task, ...] = ()
        try:
            elapsed = max(0.0, time.monotonic() - started_at)
            remaining = max(0.0, float(settings.nudge_timeout_seconds) - elapsed)
            logger.info(
                "Nudge armed; session_id=%s process_id=%s elapsed=%.3fs remaining=%.3fs timeout=%.3fs",
                self._session_id,
                self._process_id,
                elapsed,
                remaining,
                settings.nudge_timeout_seconds,
            )

            # Wait for EITHER the timer OR a tool-call signal
            timer_task = asyncio.create_task(asyncio.sleep(remaining))
            event_task = asyncio.create_task(self._tool_call.wait())
            waits = (timer_task, event_task)
            done, _pending = await asyncio.wait(
                {timer_task, event_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            trigger_reason = "tool_call" if event_task in done else "timeout"
            if is_stale is not None and await is_stale("before_nudge_send") is not None:
                return
            nudge_msg = (
                get_tool_nudge_message(lang)
                if trigger_reason == "tool_call"
                else get_timeout_nudge_message(lang)
            )
            await send.send(SideChannelEmission(nudge_msg))
            logger.info(
                "Nudge sent (%s); session_id=%s process_id=%s total_elapsed=%.3fs",
                trigger_reason,
                self._session_id,
                self._process_id,
                max(0.0, time.monotonic() - started_at),
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                "Nudge task failed; session_id=%s process_id=%s error=%s",
                self._session_id,
                self._process_id,
                e,
            )
        finally:
            # Also when stopped mid-wait: voice left these two running.
            for wait in waits:
                if not wait.done():
                    wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
