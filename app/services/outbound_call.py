"""Outbound calls through the Raya (LitWiz) call API, now or at a given time.

One POST per call: https://docs.litwizlabs.com/api-reference/calls/initiate-a-call.
``agent_args`` is how per-call values (the script, identifiers) reach the agent;
what the agent does with them is configured in the Raya console, not here.
This module knows nothing about why a call is made. A use (the collar heat
alert, app/services/heat_alert_call.py) picks the agent, builds the args and
decides, when its time comes, whether the call still makes sense.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping
from zoneinfo import ZoneInfo

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
# Farmers are not rung at night (decision of 2026-10-09, amul-oan-api#384): a
# call due inside this span is moved to its end.
QUIET_FROM, QUIET_UNTIL = time(22, 0), time(6, 0)
_TIMEOUT_SECONDS = 15.0


class OutboundCallError(RuntimeError):
    """The provider did not accept the call.

    ``retry_after`` is set only when retrying cannot place a second call: the
    provider answered 429 (and said when), or the request never reached it.
    ``ambiguous`` means the request was sent and never answered properly, so
    the call may exist; such a failure must not be dialled again blindly.
    """

    def __init__(self, reason: str, retry_after: float | None = None, *, ambiguous: bool = False):
        super().__init__(reason)
        self.retry_after = retry_after
        self.ambiguous = ambiguous


def mobile_number(value: str | None) -> str:
    """The 10-digit form the API takes; contacts arrive with or without +91 / 0."""
    digits = re.sub(r"\D", "", value or "")[-10:]
    if len(digits) != 10:
        raise OutboundCallError("invalid_number")
    return digits


async def initiate_call(
    agent_id: str, to_number: str, agent_args: Mapping[str, Any] | None = None
) -> str:
    """Place one call now and return the provider's call UUID."""
    api_key = (settings.raya_outbound_call_api_key or "").strip()
    if not api_key:
        raise OutboundCallError("not_configured")
    body = {
        "agent_id": agent_id,
        "to_number": mobile_number(to_number),
        "agent_args": dict(agent_args or {}),
        "country_code": "91",
        "timezone": "Asia/Kolkata",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.raya_outbound_call_api_url, headers={"X-API-Key": api_key}, json=body
            )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise OutboundCallError("unreachable", retry_after=5) from exc
    except httpx.HTTPError as exc:
        raise OutboundCallError(type(exc).__name__, ambiguous=True) from exc

    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code == 429:
        # Rate-limit bodies carry retry_after; concurrency-limit bodies do not.
        raise OutboundCallError("rate_limited", retry_after=float(data.get("retry_after") or 20))
    if response.status_code >= 500 or (response.status_code == 200 and not data.get("uuid")):
        raise OutboundCallError(f"http_{response.status_code}", ambiguous=True)
    if response.status_code != 200:
        raise OutboundCallError(f"http_{response.status_code}")
    return str(data["uuid"])


def after_quiet_hours(at: datetime) -> datetime:
    """``at``, or the end of quiet hours if it falls inside them."""
    local = at.astimezone(IST)
    if local.time() >= QUIET_FROM:
        local += timedelta(days=1)
    elif local.time() >= QUIET_UNTIL:
        return at
    return local.replace(hour=QUIET_UNTIL.hour, minute=0, second=0, microsecond=0)


_scheduled: set[asyncio.Task] = set()


def schedule(at: datetime, work: Callable[[], Awaitable[None]], *, name: str) -> asyncio.Task:
    """Run ``work`` at ``at`` (moved past quiet hours) on this process's event loop.

    A sleeping task is a coroutine frame and a timer handle, a few KB, so
    thousands pending cost nothing. What it does not survive is the process
    ending: a use persists its due time and reschedules on startup, and makes
    ``work`` safe to run twice, since two replicas may both recover it.
    """
    at = after_quiet_hours(at)

    async def _run() -> None:
        await asyncio.sleep(max(0.0, (at - datetime.now(timezone.utc)).total_seconds()))
        try:
            await work()
        except Exception:
            logger.exception("Scheduled outbound call failed name=%s", name)

    task = asyncio.create_task(_run(), name=name)
    _scheduled.add(task)
    task.add_done_callback(_scheduled.discard)
    return task


async def cancel_scheduled() -> None:
    """Drop pending work at shutdown; the next process recovers it from storage."""
    for task in list(_scheduled):
        task.cancel()
    await asyncio.gather(*_scheduled, return_exceptions=True)
