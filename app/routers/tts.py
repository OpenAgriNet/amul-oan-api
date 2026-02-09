from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from app.models.requests import TTSRequest
from helpers.tts import text_to_speech_bhashini_async
import uuid
import base64
from helpers.utils import get_logger
from app.auth.jwt_auth import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

@router.post("/")
async def tts(request: TTSRequest = Body(...), user_info: dict = Depends(get_current_user)):
    """
    Convert text to speech using the specified service.
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    # For now, only bhashini is implemented (async for non-blocking behaviour)
    audio_data = await text_to_speech_bhashini_async(
        request.text,
        request.target_lang,
        gender="female",
        sampling_rate=8000,
    )
    
    if isinstance(audio_data, bytes):
        audio_data = base64.b64encode(audio_data).decode('utf-8')
        
    return JSONResponse({
        'status': 'success',
        'audio_data': audio_data,
        'session_id': session_id
    }, status_code=200)