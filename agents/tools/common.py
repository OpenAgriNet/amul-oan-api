import asyncio
import contextvars
import random
import httpx
from typing import Dict, Any, Optional
from app.observability import start_observation
from helpers.utils import get_logger
from app.config import settings

logger = get_logger(__name__)

# ── Tool-call nudge signaling ───────────────────────────────────────────
# A per-request asyncio.Event stored in a ContextVar.  Any tool wrapper
# can call fire_tool_call_nudge() to tell the nudge task "a tool was
# invoked – send the hold message now instead of waiting for the timer".
_tool_call_nudge_event: contextvars.ContextVar[Optional[asyncio.Event]] = contextvars.ContextVar(
    "_tool_call_nudge_event", default=None
)


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


