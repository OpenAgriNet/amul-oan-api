"""
Translation service using TranslateGemma models.

Provides translation between Indian languages and English using
TranslateGemma 27B base model deployed on vLLM.
"""

import os
import json
import re
import random
import aiohttp
from pathlib import Path
from typing import Literal, Optional
from anthropic import AsyncAnthropic
from helpers.utils import get_logger
from dotenv import load_dotenv
from agents.tools.terms import get_mini_glossary_for_text
from agents.models import LLM_MODEL_NAME

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

load_dotenv()

logger = get_logger(__name__)


ANTHROPIC_PRETRANSLATION_MODEL = os.getenv(
    "ANTHROPIC_PRETRANSLATION_MODEL",
    "claude-haiku-4-5",
)
_anthropic_client: Optional[AsyncAnthropic] = None


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
    # Keep only script/format cleanup and a couple of safe transliteration fixes here.
    # Terminology ownership should live in the glossary/policy layers.
    (r"(?i)\bpaho\b", "બાવલું"),
    (r"ગર્ભવતી", "ગાભણ"),
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

# Multi-endpoint support: comma-separated list for load-balanced 27b-base
_27b_base_ep_raw = os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINTS", "").strip()
TRANSLATION_ENDPOINTS_27B_BASE: list[str] = (
    [e.strip() for e in _27b_base_ep_raw.split(",") if e.strip()]
    if _27b_base_ep_raw
    else [TRANSLATION_ENDPOINTS["27b-base"]]
)

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
    endpoint = random.choice(TRANSLATION_ENDPOINTS_27B_BASE)
    model_id = TRANSLATION_MODEL_IDS.get(model_size)
    return model_size, endpoint, model_id


def _get_langfuse():
    if not get_langfuse_client:
        return None
    try:
        return get_langfuse_client()
    except Exception:
        return None


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

    langfuse = _get_langfuse()

    try:
        if not langfuse:
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

        with langfuse.start_as_current_observation(
            name="text_translation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text,
            },
            model=model_id,
            metadata={
                "translation_provider": "translategemma",
                "model_size": model_size,
                "pipeline_stage": "text_translation",
            },
        ) as observation:
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
                    observation.update(output=translated_text)
                    logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
                    return translated_text

    except aiohttp.ClientError as e:
        logger.error(f"Translation API connection error: {str(e)}")
        raise Exception(f"Failed to connect to translation service: {str(e)}")


def _openai_chat_message_content(data: dict) -> str:
    """Extract assistant text from an OpenAI-compatible chat/completions JSON body."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return ""


def _inference_chat_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = (os.getenv("INFERENCE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _resolve_pretranslation_backend() -> str:
    """
    Choose query→English pre-translation backend.
    Default: Gemma on vLLM when LLM_PROVIDER=vllm; otherwise Anthropic Haiku (with TG fallback inside Haiku path).
    Override with PRETRANSLATION_BACKEND=gemma_vllm|anthropic|translategemma
    """
    explicit = (os.getenv("PRETRANSLATION_BACKEND") or "").strip().lower()
    if explicit:
        aliases = {
            "gemma_vllm": "gemma_vllm",
            "vllm": "gemma_vllm",
            "gemma": "gemma_vllm",
            "gemma4": "gemma_vllm",
            "anthropic": "anthropic",
            "haiku": "anthropic",
            "claude": "anthropic",
            "translategemma": "translategemma",
            "tg": "translategemma",
        }
        mapped = aliases.get(explicit, explicit)
        if mapped in ("gemma_vllm", "anthropic", "translategemma"):
            return mapped
        logger.warning("Unknown PRETRANSLATION_BACKEND=%r; using auto selection", explicit)
    if (os.getenv("LLM_PROVIDER") or "").lower() == "vllm":
        return "gemma_vllm"
    return "anthropic"


async def translate_to_english_pretranslation(
    text: str,
    source_lang: str,
    *,
    max_tokens: int = 1024,
) -> str:
    """GU/indic→English pre-translation for the translation pipeline (backend from env)."""
    backend = _resolve_pretranslation_backend()
    if backend == "gemma_vllm":
        return await translate_to_english_with_vllm_gemma(text, source_lang, max_tokens=max_tokens)
    if backend == "anthropic":
        return await translate_to_english_with_haiku(text, source_lang, max_tokens=max_tokens)
    return await translate_text(text, source_lang, "english")


async def translate_to_english_with_vllm_gemma(
    text: str,
    source_lang: str,
    *,
    max_tokens: int = 1024,
) -> str:
    """Translate to English via the chat LLM on OpenAI-compatible vLLM (e.g. Gemma 4 IT)."""
    if not text or not text.strip():
        return text
    if source_lang.lower() in {"english", "en"}:
        return text

    base_url = (os.getenv("INFERENCE_ENDPOINT_URL") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("INFERENCE_ENDPOINT_URL is required for Gemma vLLM pre-translation")

    model = (os.getenv("PRETRANSLATION_GEMMA_MODEL") or LLM_MODEL_NAME or "").strip()
    if not model:
        raise ValueError(
            "No model for Gemma pre-translation: set PRETRANSLATION_GEMMA_MODEL or LLM_MODEL_NAME"
        )

    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())
    system_prompt = (
        "You are a precise agricultural translation engine. "
        "Translate the user's message into natural English only. "
        "Preserve meaning, livestock terminology, and formatting. "
        "Do not answer the question. Do not add commentary."
    )
    user_content = (
        f"Translate this {source_name} ({source_code}) text to English.\n\n"
        f"{text.strip()}"
    )
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = _inference_chat_headers()
    langfuse = _get_langfuse()

    async def _call() -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    err_body = await response.text()
                    logger.error(
                        "Gemma pre-translation API error %s: %s",
                        response.status,
                        err_body[:800],
                    )
                    raise Exception(f"Gemma pre-translation failed with status {response.status}")
                result = await response.json()
                out = _openai_chat_message_content(result)
                if not out:
                    raise ValueError("Gemma vLLM pre-translation returned empty output")
                return out

    if not langfuse:
        translated = await _call()
        logger.info(
            "Gemma vLLM pre-translation ok (%s chars -> %s chars)",
            len(text),
            len(translated),
        )
        return translated

    with langfuse.start_as_current_observation(
        name="query_pretranslation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": "english",
            "text": text,
        },
        model=model,
        metadata={
            "translation_provider": "gemma_vllm",
            "pipeline_stage": "query_pretranslation",
        },
    ) as observation:
        translated_text = await _call()
        observation.update(output=translated_text)
        logger.info(
            "Gemma vLLM pre-translation ok (%s chars -> %s chars)",
            len(text),
            len(translated_text),
        )
        return translated_text


def _get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic pre-translation")
        _anthropic_client = AsyncAnthropic(api_key=api_key)
    return _anthropic_client


async def translate_to_english_with_haiku(
    text: str,
    source_lang: str,
    *,
    max_tokens: int = 1024,
) -> str:
    """Translate input text to English using Anthropic Haiku for pipeline pre-translation."""
    if not text or not text.strip():
        return text

    if source_lang.lower() in {"english", "en"}:
        return text

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not str(api_key).strip():
        logger.info(
            "ANTHROPIC_API_KEY missing; falling back to TranslateGemma for %s->English pre-translation",
            source_lang,
        )
        return await translate_text(
            text=text,
            source_lang=source_lang,
            target_lang="english",
        )

    client = _get_anthropic_client()
    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())

    langfuse = _get_langfuse()

    if not langfuse:
        response = await client.messages.create(
            model=ANTHROPIC_PRETRANSLATION_MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            system=(
                "You are a precise agricultural translation engine. "
                "Translate the user's message into natural English only. "
                "Preserve meaning, livestock terminology, and formatting. "
                "Do not answer the question. Do not add commentary."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Translate this {source_name} ({source_code}) text to English.\n\n"
                        f"{text.strip()}"
                    ),
                }
            ],
        )

        translated_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                translated_parts.append(block.text)

        translated_text = "".join(translated_parts).strip()
        if not translated_text:
            raise ValueError("Anthropic pre-translation returned empty output")
        return translated_text

    with langfuse.start_as_current_observation(
        name="query_pretranslation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": "english",
            "text": text,
        },
        model=ANTHROPIC_PRETRANSLATION_MODEL,
        metadata={
            "translation_provider": "anthropic",
            "pipeline_stage": "query_pretranslation",
        },
    ) as observation:
        response = await client.messages.create(
            model=ANTHROPIC_PRETRANSLATION_MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            system=(
                "You are a precise agricultural translation engine. "
                "Translate the user's message into natural English only. "
                "Preserve meaning, livestock terminology, and formatting. "
                "Do not answer the question. Do not add commentary."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Translate this {source_name} ({source_code}) text to English.\n\n"
                        f"{text.strip()}"
                    ),
                }
            ],
        )

        translated_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                translated_parts.append(block.text)

        translated_text = "".join(translated_parts).strip()
        if not translated_text:
            raise ValueError("Anthropic pre-translation returned empty output")
        observation.update(output=translated_text)
        return translated_text


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

    translated_parts: list[str] = []
    langfuse = _get_langfuse()

    try:
        if not langfuse:
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
                                        translated_parts.append(content)
                                        yield content
                                except json.JSONDecodeError:
                                    continue
            return

        with langfuse.start_as_current_observation(
            name="stream_translation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text,
            },
            model=model_id,
            metadata={
                "translation_provider": "translategemma",
                "model_size": model_size,
                "stream": "true",
                "pipeline_stage": "stream_translation",
            },
        ) as observation:
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
                                        translated_parts.append(content)
                                        yield content
                                except json.JSONDecodeError:
                                    continue
            observation.update(output="".join(translated_parts))

    except Exception as e:
        logger.error(f"Translation streaming error: {str(e)}")
        raise
