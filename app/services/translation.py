"""
Translation service using TranslateGemma models.

Provides translation between Indian languages and English using
TranslateGemma model sizes (4B, 12B, 27B) deployed on vLLM.
"""

import os
import json
import aiohttp
from typing import Literal, Optional
from helpers.utils import get_logger
from dotenv import load_dotenv
from agents.tools.terms import get_mini_glossary_for_text

load_dotenv()

logger = get_logger(__name__)


def _fix_dandas(text: str) -> str:
    """Replace Devanagari dandas (।) with periods in TranslateGemma output."""
    return text.replace("।", ".")


TRANSLATION_ENDPOINTS = {
    "4b": os.getenv("TRANSLATEGEMMA_4B_ENDPOINT", "http://10.128.170.2:8081/v1"),
    "12b": os.getenv("TRANSLATEGEMMA_12B_ENDPOINT", "http://10.128.170.2:8082/v1"),
    "27b": os.getenv("TRANSLATEGEMMA_27B_ENDPOINT", "http://localhost:8085/v1"),
    "27b-base": os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1"),
}

DEFAULT_TRANSLATION_MODEL = os.getenv("DEFAULT_TRANSLATION_MODEL", "4b")

TRANSLATION_MODEL_IDS = {
    "4b": os.getenv("TRANSLATEGEMMA_4B_MODEL", "translategemma-4b"),
    "12b": os.getenv("TRANSLATEGEMMA_12B_MODEL", "translategemma-12b"),
    "27b": os.getenv("TRANSLATEGEMMA_27B_MODEL", "marathi-translategemma-27b-2250"),
    "27b-base": os.getenv("TRANSLATEGEMMA_27B_BASE_MODEL", "translategemma-27b-base"),
}

LANG_NAMES = {
    "marathi": "Marathi", "english": "English", "hindi": "Hindi",
    "gujarati": "Gujarati", "tamil": "Tamil", "kannada": "Kannada",
    "odia": "Oriya", "telugu": "Telugu", "punjabi": "Punjabi",
    "malayalam": "Malayalam", "bengali": "Bengali", "urdu": "Urdu",
    "assamese": "Assamese",
    "mr": "Marathi", "en": "English", "hi": "Hindi", "gu": "Gujarati",
    "ta": "Tamil", "kn": "Kannada", "or": "Oriya", "te": "Telugu",
    "pa": "Punjabi", "ml": "Malayalam", "bn": "Bengali", "ur": "Urdu",
    "as": "Assamese"
}

LANG_CODES = {
    "marathi": "mr", "english": "en", "hindi": "hi", "gujarati": "gu",
    "tamil": "ta", "kannada": "kn", "odia": "or", "telugu": "te",
    "punjabi": "pa", "malayalam": "ml", "bengali": "bn", "urdu": "ur",
    "assamese": "as",
    "mr": "mr", "en": "en", "hi": "hi", "gu": "gu", "ta": "ta",
    "kn": "kn", "or": "or", "te": "te", "pa": "pa", "ml": "ml",
    "bn": "bn", "ur": "ur", "as": "as"
}

INDIAN_LANGUAGES = [
    "marathi", "mr", "hindi", "hi", "gujarati", "gu", "tamil", "ta",
    "kannada", "kn", "odia", "or", "telugu", "te", "punjabi", "pa",
    "malayalam", "ml", "bengali", "bn", "urdu", "ur", "assamese", "as"
]


def _format_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    mini_glossary: Optional[str] = None,
) -> str:
    """Format the translation prompt using TranslateGemma's official chat template.
    When target is Gujarati and mini_glossary is provided, injects a dynamic term list
    so the model uses consistent domain terminology."""
    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    target_name = LANG_NAMES.get(target_lang.lower(), target_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())
    target_code = LANG_CODES.get(target_lang.lower(), target_lang.lower())

    instruction = (
        f"You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text "
        f"while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_name} translation, without any additional explanations or commentary."
    )
    if mini_glossary and mini_glossary.strip():
        instruction += f"\n\nSpecific Translations for Reference:\n{mini_glossary.strip()}\n"
    instruction += f"\n\nPlease translate the following {source_name} text into {target_name}:\n\n\n{text.strip()}"

    prompt = (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return prompt


def _resolve_model(model_size: Optional[str], target_lang: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Pick the right model/endpoint based on translation direction.
    For *→English use base model; for en→* use finetuned.
    Returns (model_size, endpoint, model_id).
    """
    target_code = LANG_CODES.get(target_lang.lower(), target_lang.lower())
    if target_code == "en" and model_size in ("27b", None):
        model_size = "27b-base"
    model_size = model_size or DEFAULT_TRANSLATION_MODEL
    endpoint = TRANSLATION_ENDPOINTS.get(model_size)
    model_id = TRANSLATION_MODEL_IDS.get(model_size)
    return model_size, endpoint, model_id


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048
) -> str:
    """Translate text using TranslateGemma."""
    if not text or not text.strip():
        return text

    if source_lang.lower() == target_lang.lower():
        logger.info("Source and target languages are the same, skipping translation")
        return text

    model_size, endpoint, model_id = _resolve_model(model_size, target_lang)
    if not endpoint or not model_id:
        raise ValueError(f"Invalid translation model size: {model_size}")

    mini_glossary = ""
    if target_lang.lower() in ("gujarati", "gu"):
        mini_glossary = get_mini_glossary_for_text(text, threshold=0.95)
        if mini_glossary:
            logger.info(f"Translation prompt: injected mini glossary ({len(mini_glossary.splitlines())} terms)")
    prompt = _format_translation_prompt(text, source_lang, target_lang, mini_glossary=mini_glossary)
    logger.info(f"Translating {source_lang} -> {target_lang} using {model_size} model")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/completions",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise Exception(f"Translation failed with status {response.status}")

                result = await response.json()
                translated_text = result["choices"][0]["text"].strip()
                translated_text = _fix_dandas(translated_text)
                logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
                return translated_text

    except aiohttp.ClientError as e:
        logger.error(f"Translation API connection error: {str(e)}")
        raise Exception(f"Failed to connect to translation service: {str(e)}")


async def translate_text_stream_fast(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048
):
    """Stream translated text token by token (no artificial delay)."""
    if not text or not text.strip():
        return

    if source_lang.lower() == target_lang.lower():
        yield text
        return

    model_size, endpoint, model_id = _resolve_model(model_size, target_lang)
    if not endpoint or not model_id:
        raise ValueError(f"Invalid translation model size: {model_size}")

    mini_glossary = ""
    if target_lang.lower() in ("gujarati", "gu"):
        mini_glossary = get_mini_glossary_for_text(text, threshold=0.95)
        if mini_glossary:
            logger.info(f"Translation prompt: injected mini glossary ({len(mini_glossary.splitlines())} terms)")
    prompt = _format_translation_prompt(text, source_lang, target_lang, mini_glossary=mini_glossary)
    logger.info(f"Fast streaming translation {source_lang} -> {target_lang} using {model_size} model")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/completions",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise Exception(f"Translation failed with status {response.status}")

                buffer = b''
                async for chunk in response.content.iter_chunked(64):
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data)
                                content = chunk_data['choices'][0].get('text', '')
                                if content:
                                    yield _fix_dandas(content)
                            except json.JSONDecodeError:
                                continue

    except Exception as e:
        logger.error(f"Translation streaming error: {str(e)}")
        raise
