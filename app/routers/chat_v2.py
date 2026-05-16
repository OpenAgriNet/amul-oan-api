"""
Unified /v2/chat endpoint.

Merges the old /chat (web) and planned /voice paths into one POST endpoint
with a `channel` parameter ("web" | "voice" | "whatsapp").

Key improvements over v1:
  - POST with JSON body (better for larger payloads, voice metadata, etc.)
  - Proper SSE formatting (data: {...}\n\n) so Postman / EventSource / FE can parse it.
  - Channel-aware pipeline: switches prompt and translation behaviour per channel.
"""

import json
import uuid

from fastapi import APIRouter, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.auth.jwt_auth import get_chat_user
from app.services.chat import stream_chat_messages
from app.utils import _get_message_history
from app.models.requests import ChatRequest
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v2/chat", tags=["chat-v2"])


async def _sse_stream(message_stream):
    """Wrap raw text chunks in SSE `data:` envelope so standard clients can parse them.

    Each event is a JSON object: {"text": "<chunk>"}.
    The final event is: data: [DONE]
    """
    async for chunk in message_stream:
        payload = json.dumps({"text": chunk}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/")
async def chat_v2_endpoint(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(get_chat_user),
):
    """
    Unified chat endpoint (v2) that supports web, voice, and whatsapp channels.

    **Improvements over v1:**
    - Uses POST with JSON body
    - Proper SSE `data: {...}\\n\\n` framing for streaming mode
    - Channel-aware: `web`, `voice`, `whatsapp`

    **Auth:** Bearer JWT or X-API-Key + X-User-Phone (same as v1).
    """
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        f"[v2] Chat request - session_id: {session_id}, user_id: {body.user_id}, "
        f"channel: {body.channel}, "
        f"authenticated_user: {user_info}, source_lang: {body.source_lang}, "
        f"target_lang: {body.target_lang}, "
        f"use_translation_pipeline: {body.use_translation_pipeline}, "
        f"query: {body.query}"
    )

    history = await _get_message_history(session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    message_stream = stream_chat_messages(
        query=body.query,
        session_id=session_id,
        source_lang=body.source_lang,
        target_lang=body.target_lang,
        channel=body.channel,
        user_id=body.user_id,
        history=history,
        user_info=user_info,
        background_tasks=background_tasks,
        use_translation_pipeline=body.use_translation_pipeline or False,
    )

    # Non-streaming mode: collect full response and return JSON
    if body.stream is False:
        full_response = "".join([chunk async for chunk in message_stream])
        return JSONResponse(
            content={
                "session_id": session_id,
                "response": full_response,
                "stream": False,
                "channel": body.channel,
            }
        )

    # Streaming mode: wrap in proper SSE envelope
    return StreamingResponse(
        _sse_stream(message_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
