"""
Translation service using TranslateGemma models.

Provides translation between Indian languages and English using
TranslateGemma 27B base model deployed on vLLM.
"""

import os
import json
import re
import aiohttp
from pathlib import Path
from typing import Literal, Optional
from helpers.utils import get_logger
from dotenv import load_dotenv
from agents.tools.terms import get_mini_glossary_for_text

load_dotenv()

logger = get_logger(__name__)


GU_PREFERRED_TRANSLATION_RULES = [
    "Use farmer-preferred Gujarati livestock terms.",
    "Prefer 'બાવલું' over 'પાહો' for udder context.",
    "Prefer 'ધાર' over 'ટીપાં' for milk streams.",
    "Use 'ગાભણ' for pregnant livestock context.",
    "Do not output editorial markers like 'red colour' or formatting instructions.",
]


def _load_gu_term_policy() -> dict:
    candidates = [
        Path.cwd() / "assets/gu_term_policy.json",
        Path(__file__).resolve().parents[2] / "assets/gu_term_policy.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed loading Gujarati term policy at %s: %s", path, e)
                return {}
    return {}


def _build_gu_policy_replacements(policy: dict) -> list[tuple[str, str]]:
    forbidden = policy.get("forbidden", {}) if isinstance(policy, dict) else {}
    if not isinstance(forbidden, dict):
        return []
    # Longer keys first so phrase-level replacements win before single-word ones.
    items = sorted(
        [(str(k).strip(), str(v).strip()) for k, v in forbidden.items() if str(k).strip() and str(v).strip()],
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    out: list[tuple[str, str]] = []
    for src, dst in items:
        pattern = re.escape(src)
        out.append((pattern, dst))
    return out


GU_POST_REPLACEMENTS_BASE = [
    (r"(?i)red\s*colour\s*-?\s*delete", ""),
    (r"(?i)red\s*colour", ""),
    (r"(?i)\bpaho\b", "બાવલું"),
    (r"ગર્ભવતી", "ગાભણ"),
    (r"પાહો", "બાવલું"),
    (r"ટીપાં", "ધાર"),
    (r"શિશુ\s*પશુ", "નાના બચ્ચા/વાછરડી"),
    (r"\bટોળા\b", "ધણ"),
    (r"સંતુલિત\s*પશુ\s*ચારો", "પશુદાણ"),
    (r"ગર્ભાધાન", "બીજદાન"),
    (r"ટિક્કી", "ઇતરડી"),
    (r"સ્તનના\s*નિપલ્સ", "આંચળ"),
    (r"સ્તન\s*પ્રદેશ", "બાવલા ના ભાગ"),
    (r"\bકીડા\b", "કૃમિ"),
    (r"ઇતરડીનાશક\s*દવાઓ", "કીટનાશક દવાનો ઉપયોગ"),
    (r"\bપીપી\b", "પોટેશિયમ પરમેંગેનેટના દ્રાવણ"),
    (r"નાની\s*ઈજા", "નાની તિરાડો"),
    (r"\bપાડુ\b", "બચ્ચું"),
    (r"તાવના\s*તબક્કામાં", "તાવના શરૂઆત ના તબક્કામાં"),
]
GU_TERM_POLICY = _load_gu_term_policy()
GU_POLICY_REPLACEMENTS = _build_gu_policy_replacements(GU_TERM_POLICY)
GU_POST_REPLACEMENTS = GU_POST_REPLACEMENTS_BASE + GU_POLICY_REPLACEMENTS


def _fix_dandas(text: str) -> str:
    """Replace Devanagari dandas (।) with periods in TranslateGemma output."""
    return text.replace("।", ".")


def _post_normalize_gu_translation(
    text: str,
    target_lang: str,
    *,
    strip_outer: bool = False,
) -> str:
    if target_lang.lower() not in ("gujarati", "gu"):
        return text
    out = text
    for pat, repl in GU_POST_REPLACEMENTS:
        out = re.sub(pat, repl, out)
    # collapse extra spaces introduced by removals
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() if strip_outer else out


TRANSLATION_ENDPOINTS = {
    "4b": os.getenv("TRANSLATEGEMMA_4B_ENDPOINT", "http://10.128.170.2:8081/v1"),
    "12b": os.getenv("TRANSLATEGEMMA_12B_ENDPOINT", "http://10.128.170.2:8082/v1"),
    "27b": os.getenv("TRANSLATEGEMMA_27B_ENDPOINT", "http://localhost:8085/v1"),
    "27b-base": os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1"),
}

DEFAULT_TRANSLATION_MODEL = os.getenv("DEFAULT_TRANSLATION_MODEL", "27b-base")

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
        f"Produce only the {target_name} translation, without any additional explanations or commentary.\n"
        f"Preserve newlines, paragraph breaks, and list structure (bullets, numbered items, markdown) exactly as in the source."
    )
    if mini_glossary and mini_glossary.strip():
        lines = mini_glossary.strip().splitlines()
        rules = []
        for line in lines:
            if " -> " in line:
                en_term, gu_term = line.split(" -> ", 1)
                rules.append(f"Rule: '{en_term.strip()}' must be translated as '{gu_term.strip()}'.")
        if rules:
            instruction += "\n\n**Terminology Rules (mandatory):**\n" + "\n".join(rules) + "\n"
    if target_code == "gu":
        instruction += (
            "\n\n**Gujarati Livestock Style Rules (mandatory):**\n- "
            + "\n- ".join(GU_PREFERRED_TRANSLATION_RULES)
            + "\n"
        )
    instruction += f"\n\nPlease translate the following {source_name} text into {target_name}:\n\n\n{text.strip()}"

    prompt = (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return prompt


def _resolve_model(model_size: Optional[str], target_lang: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Resolve to 27b-base model/endpoint for all translations.
    Returns (model_size, endpoint, model_id).
    """
    model_size = "27b-base"
    endpoint = TRANSLATION_ENDPOINTS.get(model_size)
    model_id = TRANSLATION_MODEL_IDS.get(model_size)
    return model_size, endpoint, model_id


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
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
        mini_glossary = get_mini_glossary_for_text(text, threshold=0.90, max_terms=40)
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
                translated_text = _post_normalize_gu_translation(
                    translated_text,
                    target_lang,
                )
                logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
                return translated_text

    except aiohttp.ClientError as e:
        logger.error(f"Translation API connection error: {str(e)}")
        raise Exception(f"Failed to connect to translation service: {str(e)}")


async def translate_text_stream_fast(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
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
        mini_glossary = get_mini_glossary_for_text(text, threshold=0.90, max_terms=40)
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
                                    content = _fix_dandas(content)
                                    content = _post_normalize_gu_translation(
                                        content,
                                        target_lang,
                                        strip_outer=False,
                                    )
                                    yield content
                            except json.JSONDecodeError:
                                continue

    except Exception as e:
        logger.error(f"Translation streaming error: {str(e)}")
        raise
