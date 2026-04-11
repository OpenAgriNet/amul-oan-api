import asyncio
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from app.auth.jwt_auth import get_current_user
from app.models.requests import ChatRequest
from app.services.conversation import CHAT_CONFIG, process_conversation
from app.utils import _get_message_history
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache", #revalidate the cache
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no", #for nginx
}


async def sse_chat_stream(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Wrap a raw text-chunk async generator as SSE.

    Each chunk from the business stream becomes one `message` event.
    After the stream completes, emit a terminal `end` event.

    SSE format (per event):
      event: message
      data: <json payload>
      <blank line>
    """
    async for chunk in stream:
        payload = {"text": chunk}
        yield (
            "event: message\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        )
    yield "event: end\ndata: done\n\n"


@router.get("/")
async def chat_endpoint(
    request: ChatRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Chat endpoint. When stream=True (default), returns proper SSE so clients
    (browser EventSource, curl -N, Postman) can display incremental output.
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        f"Chat request received - session_id: {session_id}, user_id: {request.user_id}, "
        f"authenticated_user: {user_info}, source_lang: {request.source_lang}, "
        f"target_lang: {request.target_lang}, use_translation_pipeline: {request.use_translation_pipeline}, query: {request.query}"
    )

    history = await _get_message_history(session_id)
    logger.debug(
        f"Retrieved message history for session {session_id} - length: {len(history)}"
    )

    use_translation = bool(request.use_translation_pipeline)
    # Moderation is always on for /chat (control-layer contract; not yet wired into stream_chat_messages).
    chat_config = {
        **CHAT_CONFIG,
        "use_translation": use_translation,
        "use_moderation": True,
    }
    message_stream = await process_conversation(
        input_data={
            "query": request.query,
            "session_id": session_id,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "user_id": request.user_id,
            "history": history,
            "user_info": user_info,
            "use_translation_pipeline": use_translation,
        },
        **chat_config,
    )

    if request.stream is False:
        full_response = "".join([chunk async for chunk in message_stream])
        return JSONResponse(
            content={
                "session_id": session_id,
                "response": full_response,
                "stream": False,
            }
        )

    return StreamingResponse(
        sse_chat_stream(message_stream),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


#this was created for minimal testing of streaming
# @router.get("/sse-test")
# async def chat_sse_test():
#     """
#     Minimal SSE sanity check independent of LLM/chat logic.
#     Emits 5 message events then end.
#     """
#     async def raw_chunks():
#         for i in range(5):
#             await asyncio.sleep(0.3)
#             yield f"sse-test line {i + 1}"

#     return StreamingResponse(
#         sse_chat_stream(raw_chunks()),
#         media_type="text/event-stream",
#         headers=SSE_HEADERS,
#     )
