"""The voice endpoint, as voice-oan-api serves it (``app/routers/voice.py`` at
amul-dev ``3b19835``). Registered only when ``VOICE_ROUTE_ENABLED`` is on."""
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.auth.jwt_auth import get_current_user
from app.models.requests import VoiceRequest
from app.routers.chat import ClosingStreamingResponse
from app.services.voice import stream_voice_message
from app.utils import _get_message_history, claim_session_request_ownership
from app.voice.telemetry import RouteTimings
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.get("/")
async def voice_endpoint(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request: VoiceRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Voice endpoint that streams responses back to the client.
    Requires JWT authentication (Authorization: Bearer <token>).
    session_id is used for message history and Langfuse Sessions: same ID groups all agent runs for one conversation.
    """
    started_at = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    # No caller phone or words in the log: only whether there is a caller, and the query's length.
    logger.info(
        "Voice request received - session_id: %s, has_user_id: %s, source_lang: %s, target_lang: %s, "
        "provider: %s, process_id: %s, call_type: %s, query_chars: %s",
        session_id,
        bool(request.user_id) and request.user_id != "anonymous",
        request.source_lang,
        request.target_lang,
        request.provider,
        request.process_id,
        request.call_type,
        len(request.query or ""),
    )
    owner_started_at = time.perf_counter()
    owner = await claim_session_request_ownership(session_id)
    ownership_claim_ms = (time.perf_counter() - owner_started_at) * 1000.0
    logger.info(
        "Session ownership claimed - session_id=%s epoch=%s process_id=%s",
        session_id,
        owner.epoch,
        request.process_id,
    )

    history_started_at = time.perf_counter()
    history = await _get_message_history(session_id)
    history_load_ms = (time.perf_counter() - history_started_at) * 1000.0
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")

    return ClosingStreamingResponse(
        stream_voice_message(
            query=request.query,
            session_id=session_id,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=request.user_id,
            history=history,
            user_info=user_info,
            provider=request.provider,
            process_id=request.process_id,
            call_type=request.call_type,
            owner=owner,
            http_request=http_request,
            background_tasks=background_tasks,
            route=RouteTimings(
                started_at=started_at,
                ownership_claim_ms=ownership_claim_ms,
                history_load_ms=history_load_ms,
                history_messages=len(history),
            ),
        ),
        media_type="text/event-stream",
    )
