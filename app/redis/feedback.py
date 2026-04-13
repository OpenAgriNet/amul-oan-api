"""Redis-backed feedback state helpers for voice flow."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.redis.cache import get_cache, set_cache
from app.redis.config import key as redis_key

FEEDBACK_STATE_NAMESPACE = "feedback_state"


def _feedback_key(session_id: str) -> str:
    return redis_key(FEEDBACK_STATE_NAMESPACE, session_id)


async def get_feedback_state(session_id: str) -> dict:
    state = await get_cache(_feedback_key(session_id))
    if state:
        return state
    return {"initiated": False, "rating_received": False, "trigger": None}


async def set_feedback_initiated(session_id: str, trigger: str) -> None:
    await set_cache(
        _feedback_key(session_id),
        {"initiated": True, "rating_received": False, "trigger": trigger},
        ttl=settings.feedback_state_ttl,
    )


async def clear_feedback_initiated(session_id: str) -> None:
    await set_cache(
        _feedback_key(session_id),
        {"initiated": False, "rating_received": False, "trigger": None},
        ttl=settings.feedback_state_ttl,
    )


async def set_feedback_rating_received(session_id: str) -> None:
    state = await get_feedback_state(session_id)
    state = {**state, "rating_received": True}
    await set_cache(_feedback_key(session_id), state, ttl=settings.feedback_state_ttl)


def extract_conversation_events_from_messages(messages: list[Any]) -> list[str]:
    events: list[str] = []
    for msg in messages or []:
        for part in getattr(msg, "parts", []) or []:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            tool_name = getattr(part, "tool_name", None) or getattr(part, "name", None)
            if tool_name != "signal_conversation_state":
                continue
            raw_args = getattr(part, "args", None)
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    args = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
            else:
                args = {}
            event = args.get("event") if isinstance(args, dict) else None
            if event in ("conversation_closing", "user_frustration", "in_progress"):
                events.append(event)
    return events

