import os
import base64
import requests
import json
from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from typing import Dict
from langcodes import Language
from openai import OpenAI
from io import BytesIO
load_dotenv()

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
    


def transcribe_bhashini(audio_base64: str, source_lang='mr'):
    """
    Transcribes an audio file using the Bhashini service.

    Parameters:
    source_lang (str): The language code of the audio file's language. Default is 'gu' (Gujarati).

    Returns:
    str: The transcribed text if the request is successful.
    None: If the request fails, it returns None and prints an error message.
    """   
    url = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
    headers = {
        'Accept': '*/*',
        'User-Agent': 'Thunder Client (https://www.thunderclient.com)',
        'Authorization': os.getenv('MEITY_API_KEY_VALUE'),
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
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    response_json = response.json()
    return response_json['pipelineResponse'][0]['output'][0]['source']

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