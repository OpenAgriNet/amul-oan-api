import asyncio
import contextvars
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

_tool_call_nudge_event: contextvars.ContextVar[Optional[asyncio.Event]] = contextvars.ContextVar(
    "_tool_call_nudge_event",
    default=None,
)


def set_tool_call_nudge_event(event: asyncio.Event) -> contextvars.Token:
    return _tool_call_nudge_event.set(event)


def fire_tool_call_nudge() -> None:
    event = _tool_call_nudge_event.get(None)
    if event is not None and not event.is_set():
        event.set()


_NUDGE_MESSAGES_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "nudge_messages.json"
with open(_NUDGE_MESSAGES_PATH, "r", encoding="utf-8") as _f:
    _NUDGE_MESSAGES_DATA: dict = json.load(_f)


def get_random_nudge_message(lang_code: str = "en") -> str:
    hold = _NUDGE_MESSAGES_DATA.get("hold_messages", {})
    messages = hold.get(lang_code) or hold.get("en") or []
    if not messages:
        return "Please hold."
    return random.choice(messages)


async def send_nudge_message_raya(message: str, session_id: str, process_id: str | None = None) -> None:
    try:
        payload: Dict[str, Any] = {"message": message, "session_id": session_id}
        if process_id:
            payload["process_id"] = process_id
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                settings.nudge_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if response.status_code != 200:
            logger.warning(
                "Nudge message API failed; session_id=%s process_id=%s status=%s body=%s",
                session_id,
                process_id,
                response.status_code,
                response.text,
            )
    except Exception as e:
        logger.error(
            "Error sending nudge message; session_id=%s process_id=%s error=%s",
            session_id,
            process_id,
            e,
        )

