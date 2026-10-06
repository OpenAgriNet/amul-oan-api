from contextlib import aclosing

import anyio
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Send
from app.auth.jwt_auth import get_chat_user
from app.services.chat import stream_chat_messages
from app.utils import _get_message_history
from app.models.requests import ChatRequest
from app.personas import history_session_id_for_persona, resolve_chat_persona
from helpers.utils import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that always closes its body iterator.

    Starlette leaves the iterator suspended when the client goes away: on ASGI
    2.4 ``send`` raises out of the streaming loop, and on older servers the
    stream task is cancelled mid-``send``. The chat turn would then stay open —
    root span not exited, no outcome recorded — until the event loop finalised
    the generator, after the response's background tasks had already run inside
    that span. Closing it here records the hang-up as "cancelled" immediately.
    """

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                # Shielded: on the cancellation path this runs inside a cancelled
                # scope, and the close must still complete.
                with anyio.CancelScope(shield=True):
                    await aclose()


@router.get("/")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    request: ChatRequest = Depends(),
    user_info: dict = Depends(get_chat_user)
):
    """
    Chat endpoint that streams responses back to the client.
    Requires JWT authentication.
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(
        f"Chat request received - session_id: {session_id}, user_id: {request.user_id}, "
        f"channel: {request.channel}, "
        f"authenticated_user: {user_info}, source_lang: {request.source_lang}, "
        f"target_lang: {request.target_lang}, "
        f"requested_persona: {request.persona}, query: {request.query}"
    )
    
    resolved_persona = resolve_chat_persona(user_info, request.persona)
    history_session_id = history_session_id_for_persona(session_id, resolved_persona)
    history = await _get_message_history(history_session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    artifacts: list[dict] = []
    message_stream = stream_chat_messages(
        query=request.query,
        session_id=session_id,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        channel=request.channel,
        user_id=request.user_id,
        history=history,
        user_info=user_info,
        background_tasks=background_tasks,
        persona=resolved_persona,
        history_session_id=history_session_id,
        artifact_sink=artifacts,
        emit_artifact_frames=request.stream is not False,
    )

    if request.stream is False:
        async with aclosing(message_stream) as stream:
            full_response = "".join([chunk async for chunk in stream])
        return JSONResponse(
            content={
                "session_id": session_id,
                "response": full_response,
                "artifacts": artifacts,
                "stream": False,
            }
        )

    return ClosingStreamingResponse(message_stream, media_type='text/event-stream')
