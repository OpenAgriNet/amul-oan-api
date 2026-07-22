from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from app.auth.jwt_auth import get_current_user
from app.config import settings
from app.services.voice_trace import create_voice_trace
from app.services.voice import stream_voice_message
from app.llm_core import split as _llm_split
from app.utils import _get_message_history, claim_session_request_ownership
from app.models.requests import VoiceRequest
from helpers.utils import get_logger
import time
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

@router.get("/")
async def voice_endpoint(
    http_request: Request,
    request: VoiceRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Voice endpoint that streams responses back to the client.
    Requires JWT authentication (Authorization: Bearer <token>).
    JWT is validated using the public key from JWT_PUBLIC_KEY env or JWT_PUBLIC_KEY_PATH file.
    session_id is used for message history and Langfuse Sessions: same ID groups all agent runs for one conversation.
    """
    session_id = request.session_id or str(uuid.uuid4())
    trace = create_voice_trace(
        session_id=session_id,
        user_id=request.user_id,
        query=request.query,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        provider=request.provider,
        process_id=request.process_id,
    )
    logger.info(
        f"Voice request received - session_id: {session_id}, user_id: {request.user_id}, "
        f"source_lang: {request.source_lang}, "
        f"target_lang: {request.target_lang}, provider: {request.provider}, process_id: {request.process_id}, "
        f"query: {request.query}"
    )
    # These two steps happen before StreamingResponse starts iterating the
    # generator, so the router attaches their timings to the request trace.
    owner_started_at = time.perf_counter()
    owner = await claim_session_request_ownership(session_id)
    trace.attach_stage_timing("ownership_claim", (time.perf_counter() - owner_started_at) * 1000.0)
    logger.info(
        "Session ownership claimed - session_id=%s epoch=%s token=%s process_id=%s",
        session_id,
        owner.epoch,
        owner.request_token,
        request.process_id,
    )

    history_started_at = time.perf_counter()
    history = await _get_message_history(session_id)
    trace.attach_stage_timing(
        "history_load",
        (time.perf_counter() - history_started_at) * 1000.0,
        history_messages=len(history),
    )
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    # Sticky per-session routing via the unified weighted named-profile split
    # (the only path). Mapped back to the "oss"/"legacy" variant string the
    # downstream voice pipeline branches on. Distribution-identical to the removed
    # pipeline_router (same sha256 bucket + Redis-sticky assignment).
    pipeline_variant = await _llm_split.resolve_variant(session_id)

    return StreamingResponse(
        stream_voice_message(
            query=request.query,
            session_id=session_id,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=request.user_id,
            history=history,
            provider=request.provider,
            process_id=request.process_id,
            user_info=user_info,
            owner=owner,
            http_request=http_request,
            trace=trace,
            pipeline_variant=pipeline_variant,
        ),
        media_type='text/event-stream'
    )
