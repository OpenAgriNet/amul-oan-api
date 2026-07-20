from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from app.auth.jwt_auth import get_chat_user
from app.services.chat import stream_chat_messages
from app.llm_core import split as _llm_split
from app.config import settings
from app.utils import _get_message_history
from app.models.requests import ChatRequest
from helpers.utils import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

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
        f"use_translation_pipeline: {request.use_translation_pipeline}, query: {request.query}"
    )
    
    history = await _get_message_history(session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    # Sticky per-session routing via the unified weighted named-profile split
    # (the only path). The resolved profile is mapped back to the legacy
    # "oss"/"legacy" variant string the downstream chat pipeline branches on. With
    # the env-synthesized config (OSS_PIPELINE_PCT -> profile weights) this is the
    # same bit-compatible sha256 bucket + Redis-sticky assignment as before, so it
    # is distribution-identical to the removed pipeline_router.
    pipeline_variant = await _llm_split.resolve_variant(session_id)

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
        use_translation_pipeline=request.use_translation_pipeline if request.use_translation_pipeline is not None else True,
        pipeline_variant=pipeline_variant,
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

    return StreamingResponse(message_stream, media_type='text/event-stream')
