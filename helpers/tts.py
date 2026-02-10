import os
import base64
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

TTS_BHASHINI_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
TTS_TIMEOUT = 60.0

# Raya configuration is intentionally env-driven so that the actual endpoint
# and auth scheme can be wired without code changes.
RAYA_TTS_URL = os.getenv("RAYA_TTS_URL")
RAYA_TTS_API_KEY = os.getenv("RAYA_TTS_API_KEY")


def text_to_speech_bhashini(
    text: str,
    source_lang: str = "mr",
    gender: str = "female",
    sampling_rate: int = 8000,
) -> bytes:
    """Synchronous TTS via Bhashini. Prefer the async version in async paths."""
    import requests

    url = TTS_BHASHINI_URL
    headers = {
        "Accept": "*/*",
        "Authorization": os.getenv("MEITY_API_KEY_VALUE"),
        "Content-Type": "application/json",
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": source_lang},
                    "serviceId": "",
                    "gender": gender,
                    "samplingRate": sampling_rate,
                },
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }
    response = requests.post(url, headers=headers, json=data, timeout=TTS_TIMEOUT)
    response.raise_for_status()
    response_json = response.json()
    audio_content = response_json["pipelineResponse"][0]["audio"][0]["audioContent"]
    return base64.b64decode(audio_content)


async def text_to_speech_bhashini_async(
    text: str,
    source_lang: str = "mr",
    gender: str = "female",
    sampling_rate: int = 8000,
) -> bytes:
    """
    Async TTS via Bhashini. Non-blocking; use this from async endpoints.
    """
    headers = {
        "Accept": "*/*",
        "Authorization": os.getenv("MEITY_API_KEY_VALUE"),
        "Content-Type": "application/json",
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": source_lang},
                    "serviceId": "",
                    "gender": gender,
                    "samplingRate": sampling_rate,
                },
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }
    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        response = await client.post(TTS_BHASHINI_URL, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    audio_content = response_json["pipelineResponse"][0]["audio"][0]["audioContent"]
    return base64.b64decode(audio_content)


async def text_to_speech_raya_async(
    text: str,
    source_lang: str = "mr",
    sampling_rate: int = 8000,
    voice_id: Optional[str] = None,
) -> bytes:
    """
    Async TTS via Raya.

    NOTE: The exact Raya API contract is environment-specific. This helper assumes:
      - `RAYA_TTS_URL` points to a POST endpoint that accepts JSON.
      - Auth is provided via `RAYA_TTS_API_KEY` as a Bearer token.
      - Response either:
          a) returns JSON with base64 `audioContent`, or
          b) returns raw audio bytes.

    Adjust payload / parsing as needed to match the actual Raya API.
    """
    if not RAYA_TTS_URL:
        raise RuntimeError("RAYA_TTS_URL is not configured")

    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
    }
    if RAYA_TTS_API_KEY:
        headers["Authorization"] = f"Bearer {RAYA_TTS_API_KEY}"

    payload = {
        "text": text,
        "language": source_lang,
        "samplingRate": sampling_rate,
    }
    if voice_id:
        payload["voiceId"] = voice_id

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        response = await client.post(RAYA_TTS_URL, headers=headers, json=payload)

    response.raise_for_status()

    # Try JSON with base64 first; if that fails, treat as raw audio bytes.
    try:
        data = response.json()
        audio_b64 = (
            data.get("audioContent")
            or data.get("audio_data")
            or data.get("audio")
        )
        if not audio_b64:
            raise ValueError("No audio content field in Raya response")
        return base64.b64decode(audio_b64)
    except Exception:
        # Fallback: assume the body is already raw audio bytes.
        return response.content