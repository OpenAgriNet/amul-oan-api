import asyncio
import contextvars
import random
import httpx
from typing import Dict, Any, Optional
from voice.app.observability import start_observation
from voice.helpers.utils import get_logger
from voice.app.config import settings

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
        "હું જવાબ લઈને પાછી આવું છું, કૃપા કરીને થોડી રાહ જુઓ.",
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
