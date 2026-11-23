"""Phone numbers a chat answer may give: only ones a tool returned or the farmer wrote.

In a session that used FodderInfo, chat drops an answer line carrying a phone number
that no tool returned and the farmer did not write: the model once answered "with
contact numbers" without calling the tool and invented sellers with numbers. Earlier
turns' tool returns come from the saved history (the model's copy is trimmed, the
saved one is not); this turn's reach the tracker through ``reporting``
(agents/tools/telemetry.py), which every registered tool runs inside.
"""
from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from typing import Any, AsyncIterator, Iterable, Optional

from pydantic_ai.messages import ToolCallPart, ToolReturnPart, UserPromptPart

logger = logging.getLogger(__name__)

FODDER_TOOLS = frozenset({"search_fodder_availability", "lookup_fodder_locations", "get_fodder_listing"})
# An Indian mobile, digits optionally split by spaces or hyphens, optionally after +91.
_PHONE = re.compile(r"(?<![\d+])(?:\+?91[ -]?)?([6-9](?:[ -]?\d){9})(?!\d)")
DROPPED_NOTE = "I left out seller details I could not confirm. Ask me to search again for current sellers."


def phones(text: str) -> set[str]:
    return {re.sub(r"\D", "", match) for match in _PHONE.findall(text or "")}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class Contacts:
    """The numbers this chat session may repeat, and whether it used FodderInfo."""

    def __init__(self, history: Optional[Iterable[Any]] = None) -> None:
        self.allowed: set[str] = set()
        self.fodder = False
        for message in history or ():
            for part in getattr(message, "parts", ()):
                if isinstance(part, (ToolCallPart, ToolReturnPart)) and part.tool_name in FODDER_TOOLS:
                    self.fodder = True
                if isinstance(part, (ToolReturnPart, UserPromptPart)):
                    self.allowed |= phones(_text(part.content))

    def note(self, tool_name: str, result: Any) -> None:
        if tool_name in FODDER_TOOLS:
            self.fodder = True
        self.allowed |= phones(_text(result))

    def unconfirmed(self, line: str) -> bool:
        return self.fodder and bool(phones(line) - self.allowed)

    async def filter(self, source: AsyncIterator[str]) -> AsyncIterator[str]:
        """Pass the answer through; once the session has used FodderInfo, line by line,
        without the lines that carry an unconfirmed number."""
        buffer, dropped = "", 0
        async for chunk in source:
            if not self.fodder and not buffer:
                yield chunk
                continue
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if self.unconfirmed(line):
                    dropped += 1
                else:
                    yield line + "\n"
        if buffer:
            if self.unconfirmed(buffer):
                dropped += 1
            else:
                yield buffer
        if dropped:
            logger.warning("contacts: dropped %d answer line(s) with an unconfirmed phone number", dropped)
            yield "\n\n" + DROPPED_NOTE


_current: ContextVar[Optional[Contacts]] = ContextVar("chat_contacts", default=None)


def track(history: Optional[Iterable[Any]]) -> Contacts:
    """Start tracking this turn's tool results; tools run after this see the tracker."""
    contacts = Contacts(history)
    _current.set(contacts)
    return contacts


def note_tool_result(tool_name: str, result: Any) -> None:
    contacts = _current.get()
    if contacts is not None:
        contacts.note(tool_name, result)
