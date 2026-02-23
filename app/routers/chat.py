from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.auth.jwt_auth import get_current_user
from app.services.chat import stream_chat_messages
from app.utils import _get_message_history
from app.tasks.suggestions import create_suggestions
from app.models.requests import ChatRequest
from helpers.utils import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.get("/")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    request: ChatRequest = Depends(),
    user_info: dict = Depends(get_current_user)  # Authentication required
):
    """
    Chat endpoint that streams responses back to the client.
    Requires JWT authentication.
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(
        f"Chat request received - session_id: {session_id}, user_id: {request.user_id}, "
        f"authenticated_user: {user_info}, source_lang: {request.source_lang}, "
        f"target_lang: {request.target_lang}, use_translation_pipeline: {request.use_translation_pipeline}, query: {request.query}"
    )
    
    history = await _get_message_history(session_id)
    logger.debug(f"Retrieved message history for session {session_id} - length: {len(history)}")
        
    return StreamingResponse(
        stream_chat_messages(
            query=request.query,
            session_id=session_id,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=request.user_id,
            history=history,
            user_info=user_info,
            background_tasks=background_tasks,
            use_translation_pipeline=request.use_translation_pipeline or False,
        ),
        media_type='text/event-stream'
    ) 