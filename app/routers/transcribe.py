import uuid
import time
from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from app.models.requests import TranscribeRequest
from helpers.transcription import transcribe_bhashini
from helpers.utils import get_logger
from app.auth.jwt_auth import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

@router.post("/")
async def transcribe(request: TranscribeRequest = Body(...), user_info: dict = Depends(get_current_user)):
    """
    Transcribe audio content using the specified service.
    """
    session_id = request.session_id or str(uuid.uuid4())
        
    source_lang = request.source_lang or 'gu'
    if request.service_type == 'bhashini':
        transcription = transcribe_bhashini(request.audio_content, source_lang)
        logger.info(f"Transcription: {transcription}")
    else:
        return JSONResponse({
            'status': 'error',
            'message': 'Invalid service type'
        }, status_code=400)
        
    return JSONResponse({
        'status': 'success',
        'text': transcription,
        'session_id': session_id
    }, status_code=200)