from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from app.auth.jwt_auth import get_chat_user
from app.services.chat import stream_chat_messages
from app.llm_core import split as _llm_split
from app.config import settings
from app.utils import _get_message_history
from app.models.requests import ChatRequest
from app.personas import history_session_id_for_persona, resolve_chat_persona
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
        f"requested_persona: {request.persona}, "
        f"use_translation_pipeline: {request.use_translation_pipeline}, query: {request.query}"
    )
    
    resolved_persona = resolve_chat_persona(user_info, request.persona)
    history_session_id = history_session_id_for_persona(session_id, resolved_persona)
    history = await _get_message_history(history_session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    # Sticky per-session routing via the unified weighted named-profile split
    # (the only path). The routing token is the actual profile NAME (N-way), threaded
    # downstream and served DIRECTLY (no oss/legacy collapse). With the env-synthesized
    # config (OSS_PIPELINE_PCT -> profile weights) the profile is named oss/managed and
    # this is the same bit-compatible sha256 bucket assignment as before.
    pipeline_profile = await _llm_split.resolve_profile(session_id)

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
        use_translation_pipeline=request.use_translation_pipeline if request.use_translation_pipeline is not None else True,
        pipeline_profile=pipeline_profile,
        requested_persona=request.persona,
        history_session_id=history_session_id,
        artifact_sink=artifacts,
        emit_artifact_frames=request.stream is not False,
    )

    if request.stream is False:
        full_response = "".join([chunk async for chunk in message_stream])
        return JSONResponse(
            content={
                "session_id": session_id,
                "response": full_response,
                "artifacts": artifacts,
                "stream": False,
            }
        )

    return StreamingResponse(message_stream, media_type='text/event-stream')
