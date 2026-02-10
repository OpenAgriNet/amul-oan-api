from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from app.models.requests import TTSRequest
from helpers.tts import text_to_speech_bhashini_async, text_to_speech_raya_async
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
    Currently supports:
      - bhashini
      - raya
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        if request.service_type == "bhashini":
            audio_bytes = await text_to_speech_bhashini_async(
                request.text,
                request.target_lang,
                gender="female",
                sampling_rate=8000,
            )
        elif request.service_type == "raya":
            # Raya helper internally uses env-driven config.
            audio_bytes = await text_to_speech_raya_async(
                request.text,
                source_lang=request.target_lang,
                sampling_rate=8000,
            )
        else:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Unsupported TTS service_type '{request.service_type}'",
                },
                status_code=400,
            )
    except Exception as e:
        logger.exception("TTS failed: session_id=%s, service_type=%s, error=%s", session_id, request.service_type, str(e))
        return JSONResponse(
            {
                "status": "error",
                "message": "TTS failed",
                "detail": str(e),
                "session_id": session_id,
            },
            status_code=500,
        )

    if isinstance(audio_bytes, bytes):
        audio_data = base64.b64encode(audio_bytes).decode("utf-8")
    else:
        audio_data = audio_bytes

    return JSONResponse(
        {
            "status": "success",
            "audio_data": audio_data,
            "session_id": session_id,
        },
        status_code=200,
    )