import os
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()

TTS_BHASHINI_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
TTS_TIMEOUT = 60.0


def text_to_speech_bhashini(text, source_lang='mr', gender='female', sampling_rate=8000):
    """Synchronous version; prefer text_to_speech_bhashini_async in async code."""
    import requests
    url = TTS_BHASHINI_URL
    headers = {
        'Accept': '*/*',
        'Authorization': os.getenv('MEITY_API_KEY_VALUE'),
        'Content-Type': 'application/json',
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": source_lang},
                    "serviceId": "",
                    "gender": gender,
                    "samplingRate": sampling_rate
                }
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }
    response = requests.post(url, headers=headers, json=data, timeout=TTS_TIMEOUT)
    assert response.status_code == 200, f"Error: {response.status_code} {response.text}"
    response_json = response.json()
    audio_content = response_json['pipelineResponse'][0]['audio'][0]['audioContent']
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