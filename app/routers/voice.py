import json
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.auth.jwt_auth import get_current_user
from app.config import settings
from app.models.requests import VoiceChatRequest
from app.redis.locks import claim_session_request_ownership
from app.services.conversation import VOICE_CONFIG, process_conversation
from app.utils import _get_message_history
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def sse_wrapper(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """
    Transport layer: wrap raw async text (or dict) chunks as SSE frames.
    Each chunk is one `data:` field; stream ends with an explicit `end` event.
    """
    first_logged = False
    async for chunk in stream:
        if not first_logged:
            logger.info(
                "voice sse_wrapper: first chunk received (type=%s)",
                type(chunk).__name__,
            )
            first_logged = True
        if isinstance(chunk, dict):
            payload = json.dumps(chunk, ensure_ascii=False)
        else:
            payload = str(chunk)
        yield f"data: {payload}\n\n"

    logger.info("voice sse_wrapper: stream ended")
    yield "event: end\ndata: done\n\n"


@router.get("/")
async def voice_endpoint(
    http_request: Request,
    request: VoiceChatRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Voice pipeline: same control layer as chat (`process_conversation` + VOICE_CONFIG).
    Requires JWT (Authorization: Bearer <token>).
    """
    session_id = request.session_id or str(uuid.uuid4())
    use_translation_pipeline = settings.enable_translation_pipeline

    logger.info(
        "Voice request received - session_id: %s, user_id: %s, source_lang: %s, "
        "target_lang: %s, provider: %s, process_id: %s, use_translation_pipeline: %s, query: %s",
        session_id,
        request.user_id,
        request.source_lang,
        request.target_lang,
        request.provider,
        request.process_id,
        use_translation_pipeline,
        request.query,
    )

    owner = await claim_session_request_ownership(session_id)
    logger.info(
        "Session ownership claimed - session_id=%s epoch=%s token=%s process_id=%s",
        session_id,
        owner.epoch,
        owner.request_token,
        request.process_id,
    )

    history = await _get_message_history(session_id)
    logger.debug(
        "Retrieved message history for session %s - length: %s",
        session_id,
        len(history),
    )

    voice_opts = {
        **VOICE_CONFIG,
        "use_translation": use_translation_pipeline,
    }

    message_stream = process_conversation(
        input_data={
            "query": request.query,
            "session_id": session_id,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "user_id": request.user_id,
            "history": history,
            "provider": request.provider,
            "process_id": request.process_id,
            "user_info": user_info,
            "use_translation_pipeline": use_translation_pipeline,
            "owner": owner,
            "http_request": http_request,
        },
        **voice_opts,
    )

    return StreamingResponse(
        sse_wrapper(message_stream),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

#this is created for minimal testing of the SSE connection
# @router.get("/test-stream")
# async def voice_test_stream(_user: dict = Depends(get_current_user)):
#     """
#     Temporary: isolated SSE sanity check (dummy chunks, no LLM/voice business logic).
#     Same auth as /voice/ for consistency.
#     """

#     async def dummy_stream():
#         for i in range(5):
#             await asyncio.sleep(0.3)
#             yield f"voice test chunk {i + 1}"

#     return StreamingResponse(
#         sse_wrapper(dummy_stream()),
#         media_type="text/event-stream",
#         headers=SSE_HEADERS,
#     )
