import asyncio
import os
import base64
import requests
import json
import httpx
import logging
from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from typing import Dict
from langcodes import Language
from openai import OpenAI
from io import BytesIO
from pydub import AudioSegment

load_dotenv()

_transcription_logger = logging.getLogger(__name__)
BHASHINI_SAMPLE_RATE = 16000
BHASHINI_CHANNELS = 1

def base64_to_audio_file(base64_string: str, filename: str = "audio.wav") -> BytesIO:
    """
    Convert a base64 encoded string to a file-like object for Whisper.
    
    Args:
        base64_string (str): The base64 encoded string
        filename (str): Name of the file with extension (e.g. "audio.wav")
        
    Returns:
        BytesIO: A file-like object that can be used with Whisper
    """
    audio_bytes = base64.b64decode(base64_string)
    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename  # This is important for Whisper to recognize the file format
    return audio_file

def convert_audio_to_base64(filepath: str) -> str:
    """
    Convert a .wav file to base64 encoded string for ai4bharat
    """
    with open(filepath, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string

# @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def transcribe_whisper(audio_base64: str):
    """
    Transcribes an audio file using the Whisper service.

    Parameters:
    audio_base64 (str): The base64 encoded audio content
    """
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=base64_to_audio_file(audio_base64),
        response_format="verbose_json"
    )
    lang_code = Language.find(response.language).language
    text      = response.text
    return lang_code, text
    


def _detect_audio_format_from_base64(audio_base64: str) -> str:
    """Detect likely audio format from base64 content (first few decoded bytes)."""
    try:
        raw = base64.b64decode(audio_base64[:64], validate=False)
        if raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
            return 'wav'
        if raw[:4] == b'OggS':
            return 'ogg'
        if raw[:4] == b'fLaC':
            return 'flac'
        if len(raw) >= 4 and raw[:4] == bytes([0x1a, 0x45, 0xdf, 0xa3]):
            return 'webm'  # EBML/Matroska magic
        if raw[:3] == b'ID3' or (len(raw) >= 3 and raw[:3] == b'\xff\xfb\x90'):
            return 'mp3'
        return 'unknown'
    except Exception:
        return 'unknown'


def _convert_audio_to_wav_base64(audio_base64: str, detected_format: str) -> str:
    """
    Convert non-WAV audio (WebM, OGG, MP3, FLAC) to WAV 16kHz mono base64.
    Bhashini expects WAV at 16kHz; Chrome MediaRecorder sends WebM by default.
    Requires ffmpeg (pydub uses it for WebM/Opus decoding).
    """
    _transcription_logger.info(
        "transcribe: audio format '%s' detected, converting to WAV (%dHz, mono)",
        detected_format, BHASHINI_SAMPLE_RATE
    )
    raw_bytes = base64.b64decode(audio_base64)
    audio_buffer = BytesIO(raw_bytes)
    audio_buffer.name = f"audio.{detected_format}"

    audio = AudioSegment.from_file(audio_buffer, format=detected_format)
    # Resample to 16kHz, mono
    audio = audio.set_frame_rate(BHASHINI_SAMPLE_RATE).set_channels(BHASHINI_CHANNELS)
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_base64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

    _transcription_logger.info(
        "transcribe: conversion complete, input_len=%d -> wav_len=%d",
        len(audio_base64), len(wav_base64)
    )
    return wav_base64


def transcribe_bhashini(audio_base64: str, source_lang='mr'):
    """
    Transcribes an audio file using the Bhashini service.

    Parameters:
    source_lang (str): The language code of the audio file's language. Default is 'gu' (Gujarati).

    Returns:
    str: The transcribed text if the request is successful.

    Raises:
    requests.HTTPError: If Bhashini returns a non-2xx status.
    """
    url = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
    api_key = os.getenv('MEITY_API_KEY_VALUE')
    
    # Logging and format handling
    detected_format = _detect_audio_format_from_base64(audio_base64)
    _transcription_logger.info(
        "transcribe_bhashini: source_lang=%s, audio_len=%d, detected_format=%s, api_key_present=%s",
        source_lang, len(audio_base64), detected_format, api_key is not None
    )

    # Bhashini expects WAV 16kHz mono. Chrome MediaRecorder sends WebM (often Opus).
    if detected_format != 'wav':
        convert_format = detected_format if detected_format != 'unknown' else 'webm'
        if detected_format == 'unknown':
            _transcription_logger.info(
                "transcribe_bhashini: format unknown, assuming webm (common for browser recordings)"
            )
        audio_base64 = _convert_audio_to_wav_base64(audio_base64, convert_format)
    else:
        _transcription_logger.info("transcribe_bhashini: audio already WAV, no conversion needed")
    
    headers = {
        'Accept': '*/*',
        'User-Agent': 'Thunder Client (https://www.thunderclient.com)',
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    # "serviceId": "bhashini/ai4bharat/conformer-multilingual-asr",
                    "language": {
                        "sourceLanguage": source_lang,
                    },
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                    "preProcessors": ["vad"],
                }
            }
        ],
        "inputData": {
            "audio": [
                {
                    "audioContent": audio_base64
                }
            ]
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        
        if response.status_code != 200:
            _transcription_logger.error(
                "transcribe_bhashini: Bhashini returned status=%d, body=%s",
                response.status_code, response.text[:1000] if response.text else "(empty)"
            )
        response.raise_for_status()
        
        response_json = response.json()
        return response_json['pipelineResponse'][0]['output'][0]['source']
        
    except requests.exceptions.RequestException as e:
        _transcription_logger.exception(
            "transcribe_bhashini: Request failed: %s", str(e)
        )
        raise


BHASHINI_PIPELINE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
TRANSCRIBE_TIMEOUT = 30.0


async def transcribe_bhashini_async(audio_base64: str, source_lang: str = "mr") -> str:
    """
    Async version of transcribe_bhashini. Uses httpx for non-blocking HTTP and
    runs CPU-bound pydub conversion in a thread so the event loop is not blocked.

    Returns:
        str: The transcribed text.

    Raises:
        httpx.HTTPStatusError: If Bhashini returns a non-2xx status.
    """
    detected_format = _detect_audio_format_from_base64(audio_base64)
    _transcription_logger.info(
        "transcribe_bhashini_async: source_lang=%s, audio_len=%d, detected_format=%s",
        source_lang, len(audio_base64), detected_format,
    )

    if detected_format != "wav":
        convert_format = detected_format if detected_format != "unknown" else "webm"
        if detected_format == "unknown":
            _transcription_logger.info(
                "transcribe_bhashini_async: format unknown, assuming webm (common for browser recordings)"
            )
        audio_base64 = await asyncio.to_thread(
            _convert_audio_to_wav_base64, audio_base64, convert_format
        )
    else:
        _transcription_logger.info("transcribe_bhashini_async: audio already WAV, no conversion needed")

    api_key = os.getenv("MEITY_API_KEY_VALUE")
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": source_lang},
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                    "preProcessors": ["vad"],
                },
            }
        ],
        "inputData": {"audio": [{"audioContent": audio_base64}]},
    }

    async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT) as client:
        response = await client.post(
            BHASHINI_PIPELINE_URL,
            headers=headers,
            data=json.dumps(data),
        )
    if response.status_code != 200:
        _transcription_logger.error(
            "transcribe_bhashini_async: Bhashini returned status=%d, body=%s",
            response.status_code, (response.text[:1000] if response.text else "(empty)"),
        )
    response.raise_for_status()
    response_json = response.json()
    return response_json["pipelineResponse"][0]["output"][0]["source"]


def detect_audio_language_bhashini(audio_base64: str):
    """
    Detects the language of an audio file using the Bhashini API.
    
    Returns:
    str: The detected language code if the request is successful.
    str: An error message if the request fails.
    """
    url = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
    headers = {
        'Accept': '*/*',
        'Authorization': os.getenv('MEITY_API_KEY_VALUE'),
    }
    data = {
        "pipelineTasks": [
            {
                "taskType": "audio-lang-detection",
                "config": {
                    "serviceId": "bhashini/iitmandi/audio-lang-detection/gpu",
                    "language": {
                        "sourceLanguage": "auto"
                    },
                    "audioFormat": "wav",
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_base64}]
        }
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    detected_language_code = response_json['pipelineResponse'][0]['output'][0]['langPrediction'][0]['langCode']

    # NOTE: Keeping only English and Gujarati for now
    return 'en' if detected_language_code == 'en' else 'mr'