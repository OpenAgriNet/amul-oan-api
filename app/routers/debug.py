"""GET /api/debug/turns/{trace_id}: what a chat turn did, for the dev UI's Debug panel.

Mounted only when the debug view is on (DEBUG_VIEW_ENABLED on a development deployment, see
app/debug_view.py). A turn is readable by the user who asked it alone, for a day; any other
trace id, someone else's included, is a 404.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app import debug_view
from app.auth.jwt_auth import get_chat_user
from app.observability import get_langfuse_client, is_trace_id
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])

NOT_FOUND = "No debug record for this turn: it is more than a day old, someone else's, or from before the debug view was on."


async def _langfuse_url(trace_id: str) -> str | None:
    """The trace in Langfuse (its own sign-in), or None. The first call looks up the project id."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return await asyncio.wait_for(asyncio.to_thread(client.get_trace_url, trace_id=trace_id), timeout=3)
    except Exception as error:
        logger.info("debug view: no Langfuse link for %s: %s", trace_id, error)
        return None


@router.get("/turns/{trace_id}")
async def debug_turn(trace_id: str, user_info: dict = Depends(get_chat_user)) -> dict:
    if not is_trace_id(trace_id):
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    try:
        turn = await debug_view.read_turn(trace_id, debug_view.owner_of(user_info))
    except Exception as error:
        logger.warning("debug view: turn %s could not be read: %s", trace_id, error)
        raise HTTPException(status_code=503, detail="The debug store (Redis) did not answer; try again.") from None
    if turn is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    turn["langfuse_url"] = await _langfuse_url(trace_id)
    return turn
