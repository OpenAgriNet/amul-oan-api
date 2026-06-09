import asyncio
from contextlib import nullcontext
from functools import lru_cache
import time
from typing import AsyncGenerator, Optional, Literal
import re
from fastapi import Request

import regex
# from fastapi import BackgroundTasks
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart, SystemPromptPart

from pydantic_ai.usage import UsageLimits

from agents.voice import voice_agent, voice_agent_signed_in, STATIC_VOICE_SYSTEM_PROMPT
from agents.tools.farmer import normalize_phone_to_mobile
from agents.services.farmer_cache import (
    get_farmer_data_cached_only,
    refresh_farmer_data_bounded,
    enqueue_farmer_refresh,
    should_refresh_farmer_data,
    exceeds_max_serve_stale,
)
from app.models.union import UnionName
from app.services.scheme_ingestion import (
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from agents.tools.common import (
    get_timeout_nudge_message,
    get_tool_nudge_message,
    send_nudge_message_raya,
    set_tool_call_nudge_event,
)
from agents.tools.conversation_state import set_conversation_closing_flag
from agents.tools.terms import get_ambiguity_hints_for_query
from helpers.gujarati_numbers import mask_tag_identifier
from helpers.utils import get_logger, clean_output_by_language, get_today_date_str
from app.config import settings
from app.utils import (
    update_message_history,
    trim_history,
    format_message_pairs,
    clean_message_history_for_openai,
    SessionRequestOwner,
    is_session_request_owner,
    refresh_session_request_ownership,
    release_session_request_ownership,
)
from app.model_boundary_capture import boundary_capture_context
from app.services.stt_signals import (
    detect_stt_signal,
    generate_stt_signal_response,
    count_consecutive_stt_signals,
)
from app.services.moderation import ModerationVerdict, check_moderation
from app.services.translation import (
    INDIAN_LANGUAGES,
    OPENAI_PRETRANSLATION_MODEL,
    OSS_PRETRANSLATION_MODEL,
    translate_text,
    translate_text_stream_fast,
    translate_to_english_with_gpt5_mini,
    translate_to_english_with_oss_vllm,
    translate_to_english_with_structured_fallback,
    translation_channel,
)
from app.services.voice_trace import VoiceTrace, create_voice_trace
# NOTE: Removing telemetry for now.
# from app.tasks.telemetry import send_telemetry
from agents.deps import FarmerContext
from agents.models import (
    LLM_MODEL_NAME,
    OSS_LLM_MODEL_NAME,
    get_model_for_variant,
    provider_for_variant,
)
from agents.models.farmer import FarmerDataEnvelope, FarmerRecord
try:  # Langfuse is optional at import time
    from langfuse import get_client as _get_langfuse_client
except ImportError:  # pragma: no cover
    _get_langfuse_client = None

logger = get_logger(__name__)


class SentenceSegmenter:
    sep = 'ŽžŽžSentenceSeparatorŽžŽž'
    latin_terminals = '!?.:;'
    jap_zh_terminals = '。！？'
    terminals = latin_terminals + jap_zh_terminals

    def __init__(self):
        terminals = self.terminals
        self._re = [
            (regex.compile(r'(\P{N})([' + terminals + r'])(\p{Z}*)'), r'\1\2\3' + self.sep),
            (regex.compile(r'(' + terminals + r')(\P{N})'), r'\1' + self.sep + r'\2'),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [t for t in line.split(self.sep) if t != '']


sentence_segmenter = SentenceSegmenter()
VOICE_TRANSLATION_BATCH_CHAR_LIMIT = 600
VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS = 180


def extract_complete_sentences(text: str):
    if not text:
        return [], ""
    inline_structural_match = re.search(r"(?=\s#{1,6}\s)|(?=\n#{1,6}\s)|(?=\n\d+\.\s)|(?=\n[-*•]\s)", text)
    if inline_structural_match and inline_structural_match.start() > 0:
        split_at = inline_structural_match.start()
        head = text[:split_at]
        tail = text[split_at:].lstrip("\n")
        if head:
            return [head], tail
    structural_match = re.search(r"\n(?=(?:#{1,6}\s|[-*•]\s|\d+\.\s))", text)
    if structural_match:
        split_at = structural_match.start()
        head = text[:split_at]
        tail = text[split_at:].lstrip("\n")
        if head:
            return [head], tail
    sentences = sentence_segmenter(text)
    if len(sentences) <= 1:
        return [], text
    return sentences[:-1], sentences[-1]


def _split_voice_batch_text(text: str, max_chars: int = VOICE_TRANSLATION_BATCH_CHAR_LIMIT) -> tuple[str, str]:
    if len(text) <= max_chars:
        return text, ""

    window = text[:max_chars]
    split_at = -1
    for pattern in ("\n\n", "\n", ". ", "? ", "! ", ": ", "; ", "। ", "。 ", "### ", "## ", "# "):
        idx = window.rfind(pattern)
        if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
            split_at = idx + len(pattern.rstrip())
            break

    if split_at < 0:
        structural_markers = (
            r"\n(?=#{1,6}\s)",
            r"\n(?=\d+\.\s)",
            r"\n(?=[-*•]\s)",
            r"(?<=:)\s+",
            r"(?<=;)\s+",
        )
        for pattern in structural_markers:
            matches = list(re.finditer(pattern, window))
            if matches:
                idx = matches[-1].start()
                if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
                    split_at = idx
                    break

    if split_at < 0:
        # Last resort: split at the latest word boundary so an unpunctuated
        # run-on still flushes for voice delivery instead of stalling until
        # the stream ends.
        idx = window.rfind(" ")
        if idx >= VOICE_TRANSLATION_SOFT_SPLIT_MIN_CHARS:
            split_at = idx

    if split_at < 0:
        return text, ""

    return text[:split_at], text[split_at:]


def extract_translation_units(text: str):
    if not text:
        return [], ""

    ready_sentences, remaining = extract_complete_sentences(text)
    ready_units = [unit for unit in ready_sentences if unit and unit.strip()]

    while remaining and len(remaining) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
        head, tail = _split_voice_batch_text(remaining)
        if not head or head == remaining:
            break
        ready_units.append(head)
        remaining = tail

    return ready_units, remaining


def _batch_starts_new_line_or_list(text: str) -> bool:
    if not text or not text.strip():
        return False
    stripped = text.lstrip()
    if text != stripped:
        return True
    if stripped.startswith(("-", "•")) and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    if stripped.startswith("*") and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    return bool(re.match(r"^\d+\.\s", stripped))


def _prepare_text_for_voice_translation(text: str) -> str:
    """Make English text more translation-safe for voice rendering."""
    if not text:
        return text

    out = text
    # Flatten markdown list structure into spoken separators before translation.
    out = re.sub(r"\s*\n\s*[-*•]\s*", ", ", out)
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r":,\s*", ": ", out)
    return out.strip()


# ── Greeting short-circuit helpers ─────────────────────────────────────
_GREETING_TOKENS = {
    # English
    "hello", "hi", "hey", "hlo",
    # Gujarati
    "હલો", "હેલો", "નમસ્તે", "નમસ્કાર",
    # Hindi
    "नमस्ते", "हेलो", "हलो",
    # Transliteration
    "namaste", "halo", "helo",
    # Multi-word greeting combos
    "ha hello", "હા હલો", "ji", "જી", "bolo", "બોલો",
    "ha bolo", "હા બોલો", "ji bolo", "જી બોલો",
}


def _is_bare_greeting(query: str) -> bool:
    """Return True if the query is just a greeting with no real content."""
    cleaned = re.sub(r"[*\s]+", " ", query).strip().lower()
    if not cleaned:
        return False
    # Strip punctuation for matching
    cleaned = re.sub(r"[.,!?।]+$", "", cleaned).strip()
    if cleaned in _GREETING_TOKENS:
        return True
    # Collapse repeated words: "hello hello" → "hello"
    words = cleaned.split()
    if len(words) <= 4:
        deduped = " ".join(dict.fromkeys(words))
        if deduped in _GREETING_TOKENS:
            return True
    return False


_GREETING_RESPONSES = {
    "gu": "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો.",
    "en": "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.",
}

# ── Fragment detection (garbled / too-short input) ────────────────────────
_FRAGMENT_RESPONSES = {
    "gu": "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો.",
    "en": "I could not understand your question. Please ask your question again.",
}

_HISTORY_MARKERS = {
    "greeting": "hello",
    "fragment": "[fragment]",
    "low_confidence": "[unclear-user-input]",
    "pretranslation_failed": "[pretranslation-failed]",
    "stt_no_audio": "[stt:no-audio]",
    "stt_unclear": "[stt:unclear-speech]",
    "moderation_reject": "[moderation-rejected]",
}


def _is_fragment_query(query: str) -> bool:
    """Return True if query is too short/garbled to be a real question."""
    cleaned = re.sub(r"[*\s.,!?।]+", " ", query).strip()
    if not cleaned:
        return True
    # Single character or very short (≤3 chars) — likely noise
    if len(cleaned) <= 3:
        return True
    return False


# ── Hold message detection ─────────────────────────────────────────────
# Carrier IVR "your call is on hold" messages get picked up by STT and
# sent as user input, creating runaway loops. Detect them and respond
# with "goodbye" so the STT provider cuts the call.
_HOLD_MSG_PATTERNS_GU = [
    "હોલ્ડ પર",            # "on hold" in Gujarati
    "લાઇન પર રહો",        # "stay on the line"
    "લાઈન પર રહો",        # variant spelling
]
_HOLD_MSG_PATTERNS_EN = [
    "put your call on hold",
    "call has been put on hold",
    "call on hold",
    "please stay on the line",
    "please remain on the line",
]
TELEPHONY_TERMINATE_CALL_TOKEN = {
    "gu": "Goodbye.",
    "en": "Goodbye.",
}

TRANSLATION_TROUBLE_MESSAGE = {
    "gu": "માફ કરશો, હાલમાં તમારા સવાલનો જવાબ આપવામાં તકલીફ થઈ રહી છે. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    "en": "I'm having some trouble answering your question right now, please call in some time.",
}


def _has_meaningful_history(history: list) -> bool:
    """Return True when the session already contains non-trivial conversation."""
    for msg in reversed(history or []):
        for part in getattr(msg, "parts", []) or []:
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            if detect_stt_signal(text) is not None:
                continue
            return True
    return False


def _is_hold_message(query: str) -> bool:
    """Return True if the query looks like a carrier hold/IVR message."""
    lower = query.lower()
    for pat in _HOLD_MSG_PATTERNS_GU:
        if pat in lower:
            return True
    for pat in _HOLD_MSG_PATTERNS_EN:
        if pat in lower:
            return True
    return False


def _greeting_response(target_lang: str) -> str:
    return _GREETING_RESPONSES.get(target_lang, _GREETING_RESPONSES["gu"])


# ── Identity fast-path ────────────────────────────────────────────────────
_IDENTITY_PHRASES_GU = {
    "તમારું નામ શું છે", "તારું નામ શું છે", "તમે કોણ છો", "આ સેવા શું છે",
    "આ કઈ સેવા છે", "તમે ક્યાંથી બોલો છો", "ક્યાંથી બોલો",
}
_IDENTITY_PHRASES_EN = {
    "what is your name", "who are you", "what is this service", "what service is this",
    "where are you calling from",
}

_IDENTITY_RESPONSE_EN = (
    "I am Sarlaben, your Amul AI assistant for dairy farming and animal husbandry. "
    "Please tell me, how can I help you today?"
)

_WAIT_MESSAGES = {
    "gu": "રાહ જુઓ, હું તમારો જવાબ શોધી રહી છું.",
    "en": "Please wait a moment while I find the answer for you.",
}

_COMPARISON_PATTERNS = (
    r"\bdifference between\b",
    r"\bwhat is the difference\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.\b",
    r"\bdifference\b",
)

_EXPLAINER_PATTERNS = (
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\btell me about\b",
    r"\bexplain\b",
    r"\bmeaning of\b",
)

_SYMPTOM_PATTERNS = (
    r"\bfever\b",
    r"\bnot eating\b",
    r"\bnot come in heat\b",
    r"\bnot coming in heat\b",
    r"\bbleeding\b",
    r"\bdiarrhea\b",
    r"\bloose motion\b",
    r"\bcough\b",
    r"\bbloat\b",
    r"\bmastitis\b",
    r"\bpregnant\b",
    r"\bcalving\b",
    r"\bsick\b",
)

_IDENTITY_DRIFT_PATTERN = re.compile(
    r"\b(?:OpenAI|ChatGPT|GPT|Claude|Anthropic|large language model|"
    r"I am an AI assistant made by|I am an AI made by|created by OpenAI|"
    r"made by Anthropic)\b",
    re.IGNORECASE,
)


def _fast_path_kind_for_query(text: str) -> Optional[Literal["identity"]]:
    """Return 'identity' if the query is an identity or social-greeting query, else None."""
    cleaned = re.sub(r"[.,!?।\s]+", " ", text).strip().lower()
    if not cleaned:
        return None
    if cleaned in _IDENTITY_PHRASES_GU or cleaned in _IDENTITY_PHRASES_EN:
        return "identity"
    return None


def render_in_flight_wait_message(lang: str) -> str:
    """Return the localized in-flight wait message for the given language code."""
    key = (lang or "en").strip().lower()
    return _WAIT_MESSAGES.get(key, _WAIT_MESSAGES["en"])


def _guard_identity_drift(text: str) -> str:
    """Replace any sentence that leaks a non-Sarlaben AI identity with the canonical line."""
    if not _IDENTITY_DRIFT_PATTERN.search(text):
        return text
    sentences = sentence_segmenter(text.strip())
    fixed = []
    replaced = False
    for s in sentences:
        if _IDENTITY_DRIFT_PATTERN.search(s):
            if not replaced:
                fixed.append(_IDENTITY_RESPONSE_EN)
                replaced = True
        else:
            fixed.append(s)
    return " ".join(fixed).strip()


def _voice_answer_mode_for_query(text: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip().lower()
    if not cleaned:
        return None
    if any(re.search(pattern, cleaned) for pattern in _COMPARISON_PATTERNS):
        return "compact_comparison"
    if any(re.search(pattern, cleaned) for pattern in _EXPLAINER_PATTERNS):
        return "compact_explainer"
    if any(re.search(pattern, cleaned) for pattern in _SYMPTOM_PATTERNS):
        return "action_first_symptom"
    return None


def _prepare_voice_output(text: str, lang_code: str) -> str:
    """Normalize model output for voice delivery."""
    return clean_output_by_language(text, lang_code)


def _canonical_history_user_text(kind: str, fallback: str = "") -> str:
    return _HISTORY_MARKERS.get(kind, fallback or kind)


async def _voice_output_stream(text: str, target_lang: str):
    """Stream output translation on the voice channel (§14 channel-aware). The
    `with` lives inside this generator, so the voice channel stays active for the
    whole iteration without re-indenting the caller's loop body."""
    with translation_channel("voice"):
        async for chunk in translate_text_stream_fast(
            text=text,
            source_lang="english",
            target_lang=target_lang,
        ):
            yield chunk


async def _render_text_for_caller(text_en: str, target_lang: str) -> str:
    """Render English loop text for the caller's language outside the agent loop."""
    normalized_target = (target_lang or "en").strip().lower()
    if normalized_target in {"en", "english"}:
        return _prepare_voice_output(text_en, "en")

    try:
        with translation_channel("voice"):
            translated = await translate_text(
                text=text_en,
                source_lang="english",
                target_lang=normalized_target,
            )
        return _prepare_voice_output(translated, normalized_target)
    except Exception as e:
        logger.error(
            "Caller render translation failed; target_lang=%s text=%r error=%s",
            normalized_target,
            text_en[:120],
            e,
        )
        return TRANSLATION_TROUBLE_MESSAGE.get(
            normalized_target,
            TRANSLATION_TROUBLE_MESSAGE["en"],
        )


async def _canned_for_caller(text_en: str, target_lang: str, canned: dict[str, str]) -> str:
    """Return a pre-written caller string for the target language when one exists,
    skipping the TranslateGemma round-trip on fixed fast-path replies. Falls back to
    live translation for languages that have no canned variant."""
    key = (target_lang or "en").strip().lower()
    if key in canned:
        return _prepare_voice_output(canned[key], key)
    return await _render_text_for_caller(text_en, target_lang)


def _history_pair(user_text: str, assistant_text: str) -> tuple[ModelRequest, ModelResponse]:
    return (
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=assistant_text)]),
    )


def _is_signed_in_session(user_info: Optional[dict], user_id: str) -> bool:
    if user_id and user_id != "anonymous":
        return True
    return bool(user_info)


async def get_or_fetch_farmer_data(mobile: str):
    """Voice read policy (stale-while-revalidate).

    Serve the cached envelope immediately when present (fresh or stale — the
    caller enqueues a background refresh for stale records). On a cold/never-
    cached (or hard-expired/deleted) miss, do a bounded blocking fetch so the
    first turn has data, capped by FARMER_COLD_FETCH_TIMEOUT so a slow upstream
    never hangs the call.

    Still patchable by tests that stub this symbol.
    """
    cached = await get_farmer_data_cached_only(mobile)
    if cached is not None:
        if exceeds_max_serve_stale(cached):
            # Too stale to serve (e.g. background refresh has been failing):
            # block on a bounded API call, falling back to the stale record
            # only if the API also fails.
            fresh = await refresh_farmer_data_bounded(mobile)
            return fresh if fresh is not None else cached
        return cached
    return await refresh_farmer_data_bounded(mobile)


def _build_runtime_context_request(deps: FarmerContext) -> ModelRequest:
    """Stable per-call context (constant across a call's turns: date, farmer
    profile, signed-in state, tool groups). Placed BEFORE history so the token
    sequence [system][stable-context][history] stays a single growing prefix that
    vLLM prefix caching can reuse across turns. Per-query content that changes
    every turn lives in _build_query_hints_request() and is appended AFTER history
    so it never breaks this prefix."""
    tool_groups = ["retrieval", "booking"]
    if deps.signed_in and deps.mobile:
        tool_groups.append("signed-in-farmer-data")
    runtime_context = deps.get_runtime_context_message()
    context_lines = [
        "Runtime context for this turn:",
        f"- Today date: {get_today_date_str()}",
        runtime_context.replace("Runtime context for this turn:\n", "", 1),
        f"- Tool groups in this run: {', '.join(tool_groups)}",
    ]
    return ModelRequest(parts=[UserPromptPart(content="\n".join(context_lines))])


def _build_query_hints_request(deps: FarmerContext) -> Optional[ModelRequest]:
    """Per-query hints derived from the caller's current utterance: disambiguation
    rules (for ambiguous terms) and the voice answer mode. Kept OUT of the stable
    pre-history context — these change every turn, so placing them before history
    would break the cacheable prefix. Appended right before the user message
    instead, where the instructions also sit closest to the query they describe.
    Returns None when the query triggers neither hint."""
    hint_lines: list[str] = []
    # Inject ambiguity hints for the agent so it can decide to clarify vs. answer
    ambiguity_hints = get_ambiguity_hints_for_query(
        deps.query or "",
        threshold=settings.ambiguity_match_threshold,
    )
    if ambiguity_hints:
        hint_lines.append(f"- Disambiguation rules for terms in this query:\n{ambiguity_hints}")
    answer_mode = _voice_answer_mode_for_query(deps.query or "")
    if answer_mode == "compact_comparison":
        hint_lines.append(
            "- Voice answer mode: compact comparison. Give one short contrast sentence, then at most one short practical takeaway. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "compact_explainer":
        hint_lines.append(
            "- Voice answer mode: compact explainer. Give one short plain-language definition or explanation, then at most one short practical takeaway. Do not teach the full topic. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "action_first_symptom":
        hint_lines.append(
            "- Voice answer mode: action-first symptom response. Start with the most useful immediate action in one short sentence. Add at most one short safety or escalation sentence. Do not give long background, multiple causes, or a symptom checklist unless asked."
        )
    if not hint_lines:
        return None
    return ModelRequest(parts=[UserPromptPart(content="\n".join(["Hints for the current user query:", *hint_lines]))])


def _extract_farmer_tags(records: list[FarmerRecord]) -> list[str]:
    tags: list[str] = []
    for record in records:
        raw = record.tagNumbers or record.tagNo or ""
        if not raw:
            continue
        for tag in str(raw).split(","):
            cleaned = tag.strip()
            masked = mask_tag_identifier(cleaned)
            if masked and masked not in tags:
                tags.append(masked)
    return tags


def _build_compact_farmer_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    if envelope is None or not envelope.farmers:
        return ""

    first = envelope.farmers[0]
    tags = _extract_farmer_tags(envelope.farmers)
    societies = sorted({r.societyName for r in envelope.farmers if r.societyName})

    lines = [
        f"- Farmer records matched: {len(envelope.farmers)}",
        f"- Farmer data source: {envelope.source or 'unknown'}",
        f"- Farmer cache state: {'stale' if envelope.stale else 'fresh'}",
    ]
    if envelope.refreshAfter:
        lines.append(f"- Farmer refresh after: {envelope.refreshAfter}")
    if first.farmerName:
        lines.append(f"- Farmer name: {first.farmerName}")
    if societies:
        lines.append(f"- Societies: {', '.join(societies[:3])}")
    if first.farmerCode:
        lines.append(f"- Farmer code available: yes")
    union_code = first.model_dump().get("unionCode") or first.model_dump().get("union_code")
    society_code = first.model_dump().get("societyCode") or first.model_dump().get("society_code")
    if union_code:
        lines.append(f"- Union code: {union_code}")
    if society_code:
        lines.append(f"- Society code: {society_code}")
    if first.farmerCode:
        lines.append(f"- Farmer code: {first.farmerCode}")
    # Herd counts: always surface what we have. The agent answers from this
    # context (the brittle get_herd_summary / list_animal_tags / get_farmer_profile
    # tools were dropped — they read the same cache and returned "not available"
    # when the upstream record omitted totalAnimals even though tags were present).
    first_data = first.model_dump()
    total_animals = first.totalAnimals
    if total_animals is None and tags:
        total_animals = len(tags)  # fallback when upstream omits the count
    if total_animals is not None:
        lines.append(f"- Total animals: {total_animals}")
    cow = first_data.get("cow") or first_data.get("Cow")
    if cow is not None:
        lines.append(f"- Cows: {cow}")
    buffalo = first_data.get("buffalo") or first_data.get("Buffalo")
    if buffalo is not None:
        lines.append(f"- Buffaloes: {buffalo}")
    milking = first_data.get("totalMilkingAnimals") or first_data.get("Milking Animal")
    if milking is not None:
        lines.append(f"- Milking animals: {milking}")
    if tags:
        # All tags inline — no truncation, since the list-tags tool was dropped.
        lines.append(f"- Known animal tags: {', '.join(tags)}")
    if len(envelope.farmers) > 1:
        lines.append("- Multiple farmer records are registered on this mobile number.")
        lines.append("- For AI booking, first ask which farmer name the caller wants to use.")
        lines.append("- Use the selected farmer's society and codes only after the farmer is identified.")

    for index, record in enumerate(envelope.farmers[:5], start=1):
        record_data = record.model_dump()
        farmer_name = record_data.get("farmerName") or "Unknown farmer"
        society_name = record_data.get("societyName") or "Unknown society"
        farmer_code = record_data.get("farmerCode")
        union_code = record_data.get("unionCode") or record_data.get("union_code")
        society_code = record_data.get("societyCode") or record_data.get("society_code")
        lines.append(
            f"- Farmer option {index}: name={farmer_name}, society_name={society_name}, "
            f"farmer_code={farmer_code}, union_code={union_code}, society_code={society_code}"
        )

    return "\n".join(lines)


SUPPORTED_SCHEME_CONTEXT_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
}


def _collect_farmer_unions(envelope: Optional[FarmerDataEnvelope]) -> list[str]:
    if envelope is None:
        return []

    seen: set[str] = set()
    unions: list[str] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        raw_union = record.get("unionName") or record.get("union_name")
        normalized_union = str(raw_union or "").strip().lower()
        if not normalized_union or normalized_union in seen:
            continue
        seen.add(normalized_union)
        unions.append(normalized_union)
    return unions


async def _build_union_scheme_summary(farmer_unions: list[str]) -> str:
    scheme_unions = [union_name for union_name in farmer_unions if union_name in SUPPORTED_SCHEME_CONTEXT_UNIONS]
    if not scheme_unions:
        return ""

    lines = [
        "",
        "## Union schemes available",
        "- The following scheme titles are available from the union scheme cache. Use these titles and links for scheme-related questions. Retrieve full cached scheme details when the user asks about a specific scheme.",
    ]
    for union_name in scheme_unions:
        try:
            records = await get_cached_scheme_records_for_union(union_name)
        except SchemeDependencyError:
            logger.warning("Union scheme summary skipped because Redis dependency is unavailable union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache dependency is unavailable.")
            continue
        except SchemeCacheError:
            logger.warning("Union scheme summary skipped because scheme cache could not be read union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache could not be read.")
            continue
        except Exception as exc:
            logger.warning("Union scheme summary skipped because of unexpected error union=%s error=%s", union_name, exc)
            lines.append(f"- **{union_name.title()}**: Scheme list is temporarily unavailable.")
            continue

        if not records:
            lines.append(f"- **{union_name.title()}**: No cached scheme list is available yet.")
            continue

        lines.append(f"- **{union_name.title()} union schemes:**")
        seen_links: set[tuple[str, str]] = set()
        for record in records:
            title = record.get("scheme_title")
            link = record.get("scheme_url")
            if not title or not link:
                continue
            dedupe_key = (str(title).casefold(), str(link))
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            lines.append(f"  - {title}: {link}")
    return "\n".join(lines)


def _build_ai_technician_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    if envelope is None:
        return ""

    technician_groups = envelope.aiTechnicians or []
    lines: list[str] = []
    if technician_groups:
        lines.append("- AI technician options for booking are internal context, not user-provided information.")
        lines.append("- The caller does not know which AI technicians are available unless you tell them by technician name.")
        lines.append("- AI technician options for booking are grouped by farmer and society.")
        lines.append("- Each technician option only has these fields: id, full_name, mobile_number.")
        lines.append("- When asking the farmer to choose a technician, use the technician full name in natural spoken form.")
        lines.append("- Do not ask by technician position, number, option index, or ordinal words such as first, second, or third.")
        lines.append("- Mention phone only if a disambiguating mobile number is needed.")
        for group in technician_groups[:5]:
            farmer_name = group.get("farmerName") or "Unknown farmer"
            society_name = group.get("societyName") or "Unknown society"
            society_code = group.get("societyCode")
            union_code = group.get("unionCode")
            lines.append(
                f"- Technician group: farmer_name={farmer_name}, society_name={society_name}, "
                f"union_code={union_code}, society_code={society_code}"
            )
            technicians = group.get("technicians") or []
            if not technicians:
                lines.append("- AI technician option: none available for this farmer group.")
                continue
            for technician in technicians[:5]:
                name = technician.get("fullName")
                mobile = technician.get("mobileNumber")
                user_id = technician.get("userId")
                option = "- AI technician option:"
                if user_id:
                    option += f" id={user_id}"
                if name:
                    option += f" full_name={name}"
                if mobile:
                    option += f", mobile_number={mobile}"
                lines.append(option)
    else:
        lines.append("- AI technician options for booking are not available in the current signed-in context.")
    return "\n".join(lines)


def should_translate_batch(
    batch_text: str,
    word_count: int,
    is_first_batch: bool = False,
) -> bool:
    """Decide whether the accumulated batch should be flushed for translation."""
    text_end = batch_text.rstrip()
    ends_sentence = text_end.endswith(('.', '!', '?', ':'))

    # Phase 1: first batch — get first audio to the caller fast.
    if is_first_batch:
        return ends_sentence and word_count >= 3

    # Phase 2: subsequent batches — balance quality vs latency.
    if len(batch_text) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
        return True
    if word_count >= 40:
        return True  # force flush, don't hoard

    if word_count < 8:
        return ends_sentence and word_count >= 5

    # 8-40 words: flush on any natural boundary.
    if ends_sentence:
        return True
    if text_end.endswith('\n\n'):
        return True
    if text_end.endswith('\n') and len(batch_text.split('\n')) > 1:
        last_line = batch_text.rstrip('\n').split('\n')[-1].strip()
        if last_line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', last_line):
            return True
    return False

# Langfuse Sessions: same session_id groups all traces for one conversation (session replay, session-level metrics).
def _langfuse_session_context(session_id: str, user_id: str, process_id: Optional[str] = None):
    """Set Langfuse session_id so all agent runs for this conversation appear under one Session."""
    try:
        from app.observability import langfuse_client
        from langfuse import propagate_attributes
        if langfuse_client is None:
            return nullcontext()
        # Langfuse Sessions: session_id ≤200 chars (US-ASCII); same ID = one Session in Langfuse UI
        safe_session_id = (session_id or "").strip()[:200]
        kwargs = dict(
            session_id=safe_session_id or None,
            user_id=(user_id or "anonymous")[:200],
        )
        if process_id:
            kwargs["metadata"] = {"process_id": str(process_id)[:200]}
        return propagate_attributes(**kwargs)
    except Exception:
        return nullcontext()


async def stream_voice_message(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    provider: Optional[Literal['RAYA']] = None,
    process_id: Optional[str] = None,
    user_info: dict = None,
    owner: Optional[SessionRequestOwner] = None,
    http_request: Optional[Request] = None,
    trace: Optional[VoiceTrace] = None,
    pipeline_variant: str = "legacy",
#    background_tasks: BackgroundTasks,

) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    request_started_at = time.monotonic()
    # OSS sticky variant => run the dev OSS path (vLLM gemma agent + vLLM
    # gemma pretranslation). 'legacy' keeps current prod behaviour byte-for-byte;
    # with OSS_PIPELINE_PCT=0 (or OSS endpoint unset) every session is 'legacy'.
    is_oss = pipeline_variant == "oss"
    request_model = get_model_for_variant(pipeline_variant)
    request_provider = provider_for_variant(pipeline_variant)
    request_model_name = OSS_LLM_MODEL_NAME if is_oss else LLM_MODEL_NAME
    last_owner_refresh_at = 0.0
    last_emitted_sig_char: str | None = None
    trace = trace or create_voice_trace(
        session_id=session_id,
        user_id=user_id,
        query=query,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider,
        process_id=process_id,
    )
    # Tag the trace with the resolved pipeline variant so Langfuse dashboards
    # can filter sessions by variant. The categorical *score* is emitted
    # below from inside `trace.request_context()`, where a Langfuse trace
    # context is active — emitting it here would silently no-op
    # (Langfuse v4: "Operations that depend on an active span will be
    # skipped"; mirror of amul-oan-api#70).
    try:
        trace.metadata["pipeline_variant"] = pipeline_variant
        trace.metadata["request_model"] = request_model_name
        trace.metadata["request_provider"] = request_provider
    except Exception:  # pragma: no cover - never break the call
        pass
    logger.info(
        "voice request_variant session_id=%s variant=%s model=%s provider=%s",
        session_id,
        pipeline_variant,
        request_model_name,
        request_provider,
    )

    async def _request_is_stale(reason: str) -> bool:
        nonlocal last_owner_refresh_at
        if http_request is not None and await http_request.is_disconnected():
            trace.set_outcome("client_disconnected")
            logger.info(
                "Stopping request due to client disconnect - session_id=%s process_id=%s reason=%s",
                session_id,
                process_id,
                reason,
            )
            return True

        now = time.monotonic()
        if owner is not None and (
            last_owner_refresh_at == 0.0
            or now - last_owner_refresh_at >= settings.session_owner_refresh_interval_seconds
        ):
            refreshed = await refresh_session_request_ownership(owner)
            last_owner_refresh_at = now
            if not refreshed:
                trace.set_outcome("stale_request")
                logger.info(
                    "Stopping stale request after ownership lost during refresh - session_id=%s process_id=%s epoch=%s reason=%s",
                    session_id,
                    process_id,
                    owner.epoch,
                    reason,
                )
                return True

        if owner is not None and not await is_session_request_owner(owner):
            trace.set_outcome("stale_request")
            logger.info(
                "Stopping stale request because a newer request owns the session - session_id=%s process_id=%s epoch=%s reason=%s",
                session_id,
                process_id,
                owner.epoch,
                reason,
            )
            return True

        return False

    def _first_sig_char(text: str) -> str | None:
        for ch in text or "":
            if not ch.isspace():
                return ch
        return None

    def _last_sig_char(text: str) -> str | None:
        for ch in reversed(text or ""):
            if not ch.isspace():
                return ch
        return None

    def _prepare_translated_emit(text: str) -> str:
        nonlocal last_emitted_sig_char
        if not isinstance(text, str) or not text:
            return text

        first_sig = _first_sig_char(text)
        if (
            last_emitted_sig_char in {".", "!", "?", "।"}
            and first_sig is not None
            and re.match(r"[A-Za-z\u0A80-\u0AFF]", first_sig)
            and not text[0].isspace()
        ):
            text = " " + text

        last_sig = _last_sig_char(text)
        if last_sig is not None:
            last_emitted_sig_char = last_sig
        return text

    def _emit(text: str, *, kind: str = "assistant") -> str:
        trace.record_emit(text, kind=kind)
        return text

    try:
        # Keep the Langfuse root observation open for the full streaming
        # generator so downstream model calls and pydantic-ai spans nest under
        # this voice_request.
        with trace.request_context():
            # Emit the per-session pipeline_variant categorical score from
            # *inside* the trace context (chat #70 fix). score_id is
            # deterministic per session so subsequent voice turns in the
            # same session upsert the same score (no duplicates).
            if _get_langfuse_client is not None:
                try:
                    _lf = _get_langfuse_client()
                    _lf.score_current_trace(
                        name="pipeline_variant",
                        value=pipeline_variant,
                        data_type="CATEGORICAL",
                        score_id=f"voice-variant-{(session_id or '')[:180]}",
                        comment="Sticky pipeline variant for this voice session",
                    )
                except Exception as e:  # pragma: no cover
                    logger.debug("Langfuse: voice pipeline_variant score failed: %s", e)
            requested_source_lang = (source_lang or "gu").strip().lower()
            requested_target_lang = (target_lang or "gu").strip().lower()
            trace.set_language(requested_source_lang, requested_target_lang)
            needs_output_translation = requested_target_lang in INDIAN_LANGUAGES and requested_target_lang not in {"en", "english"}
            nudge_lang = (requested_target_lang or "en").strip().lower()
            has_meaningful_history = _has_meaningful_history(history)

            # ── STT signal handling (no-audio / unclear speech) ─────────────
            # These are not real user messages — skip translation & agent,
            # generate a short contextual "please repeat" via GPT-5-mini.
            stt_signal = detect_stt_signal(query)
            if stt_signal is not None:
                trace.set_route("stt_signal")
                logger.info(
                    "STT signal detected; session_id=%s process_id=%s signal=%s",
                    session_id,
                    process_id,
                    stt_signal,
                )
                recent_text = "\n\n".join(format_message_pairs(history, 3))
                if await _request_is_stale("before_stt_signal_response"):
                    return
                prior_stt_failures = count_consecutive_stt_signals(history)
                final_attempt = (prior_stt_failures + 1) >= max(1, settings.stt_signal_retry_ceiling)
                with trace.stage("stt_signal_response", as_type="generation"):
                    stt_response = await generate_stt_signal_response(
                        signal=stt_signal,
                        target_lang=requested_target_lang,
                        recent_history_text=recent_text,
                        final_attempt=final_attempt,
                    )
                history_signal = (
                    _canonical_history_user_text("stt_no_audio")
                    if stt_signal == "No audio/User is speaking softly"
                    else _canonical_history_user_text("stt_unclear")
                )
                history_response = _FRAGMENT_RESPONSES["en"] if not final_attempt else "Sorry, I still could not hear you clearly. Please try again later."
                stt_req, stt_resp = _history_pair(history_signal, history_response)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, stt_req, stt_resp])
                trace.set_outcome("stt_signal")
                yield _emit(_prepare_voice_output(stt_response, requested_target_lang))
                return

            # ── Hold message short-circuit ────────────────────────────────
            # Carrier IVR "your call is on hold" messages get transcribed by
            # STT and sent as user input, creating runaway loops of 20+ traces.
            # Respond with "goodbye" so the STT provider disconnects the call.
            if _is_hold_message(query):
                trace.set_route("hold_message")
                logger.info(
                    "Hold message detected; responding with goodbye to cut call - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query[:100],
                )
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang,
                    TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                trace.set_outcome("hold_message")
                yield _emit(_prepare_voice_output(goodbye, requested_target_lang))
                return

            # ── Greeting short-circuit ────────────────────────────────────
            # Bare greetings ("hello", "હલો", "હા") should not trigger the
            # full agent pipeline or a nudge.  Respond immediately.
            # When translation pipeline is active, let greetings flow through
            # the normal agent pipeline so history stays in English.
            if _is_bare_greeting(query) and not has_meaningful_history:
                trace.set_route("greeting_fast_path")
                logger.info(
                    "Bare greeting detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                greeting_history = _GREETING_RESPONSES["en"]
                with trace.stage("greeting_fast_path"):
                    greeting_response = await _canned_for_caller(greeting_history, requested_target_lang, _GREETING_RESPONSES)
                greet_req, greet_resp = _history_pair(_canonical_history_user_text("greeting"), greeting_history)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, greet_req, greet_resp])
                trace.set_outcome("greeting_fast_path")
                yield _emit(_prepare_voice_output(greeting_response, requested_target_lang))
                return

            # ── Identity fast-path ────────────────────────────────────────
            # Pure identity queries ("What is your name?", "What is this service?")
            # should return the canonical Sarlaben identity line directly
            # without running the full agent pipeline.
            if _fast_path_kind_for_query(query) == "identity" and not has_meaningful_history:
                trace.set_route("identity_fast_path")
                logger.info(
                    "Identity fast-path triggered; session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                identity_resp_en = _IDENTITY_RESPONSE_EN
                with trace.stage("identity_fast_path"):
                    identity_resp_for_caller = await _render_text_for_caller(identity_resp_en, requested_target_lang)
                id_req, id_resp = _history_pair(_canonical_history_user_text("greeting"), identity_resp_en)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, id_req, id_resp])
                trace.set_outcome("identity_fast_path")
                yield _emit(_prepare_voice_output(identity_resp_for_caller, requested_target_lang))
                return

            # ── Fragment short-circuit ────────────────────────────────────
            # Very short / garbled input (≤3 chars) that isn't a greeting or
            # STT signal — ask the farmer to repeat instead of routing to agent.
            if _is_fragment_query(query) and not has_meaningful_history:
                trace.set_route("fragment_fast_path")
                logger.info(
                    "Fragment query detected; short-circuiting - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                frag_response_for_history = _FRAGMENT_RESPONSES["en"]
                with trace.stage("fragment_fast_path"):
                    frag_response_for_caller = await _canned_for_caller(frag_response_for_history, requested_target_lang, _FRAGMENT_RESPONSES)
                frag_req, frag_resp = _history_pair(_canonical_history_user_text("fragment"), frag_response_for_history)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, frag_req, frag_resp])
                trace.set_outcome("fragment_fast_path")
                yield _emit(_prepare_voice_output(frag_response_for_caller, requested_target_lang))
                return

            # ── Nudge: arm BEFORE any pre-processing ────────────────────────
            # Fires on whichever happens first:
            #   (a) the configured timer expires, OR
            #   (b) the LLM invokes a tool (signalled via tool_call_event).
            # Cancelled if first text/translated chunk reaches the client first.
            nudge_task = None
            if settings.enable_voice_nudges:
                trace.set_nudge(armed=True, sent=False)
                nudge_sent = False
                tool_call_event = asyncio.Event()
                set_tool_call_nudge_event(tool_call_event)

                async def send_nudge_on_trigger() -> None:
                    nonlocal nudge_sent
                    try:
                        elapsed = max(0.0, time.monotonic() - request_started_at)
                        remaining = max(0.0, float(settings.nudge_timeout_seconds) - elapsed)
                        logger.info(
                            "Nudge armed; session_id=%s process_id=%s elapsed=%.3fs remaining=%.3fs timeout=%.3fs",
                            session_id,
                            process_id,
                            elapsed,
                            remaining,
                            settings.nudge_timeout_seconds,
                        )

                        # Wait for EITHER the timer OR a tool-call signal
                        timer_task = asyncio.create_task(asyncio.sleep(remaining))
                        event_task = asyncio.create_task(tool_call_event.wait())
                        done, pending = await asyncio.wait(
                            {timer_task, event_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()

                        trigger_reason = "tool_call" if event_task in done else "timeout"
                        if await _request_is_stale("before_nudge_send"):
                            return
                        if nudge_sent:
                            return
                        nudge_sent = True
                        trace.set_nudge(
                            sent=True,
                            trigger=trigger_reason,
                            sent_ms=round((time.monotonic() - request_started_at) * 1000.0, 2),
                        )
                        nudge_msg = (
                            get_tool_nudge_message(nudge_lang)
                            if trigger_reason == "tool_call"
                            else get_timeout_nudge_message(nudge_lang)
                        )
                        await send_nudge_message_raya(nudge_msg, session_id, process_id)
                        elapsed = max(0.0, time.monotonic() - request_started_at)
                        logger.info(
                            "Nudge sent (%s); session_id=%s process_id=%s total_elapsed=%.3fs",
                            trigger_reason,
                            session_id,
                            process_id,
                            elapsed,
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.warning(
                            "Nudge task failed; session_id=%s process_id=%s error=%s",
                            session_id,
                            process_id,
                            e,
                        )

                nudge_task = asyncio.create_task(send_nudge_on_trigger())
                logger.info(
                    "Nudge initiated; session_id=%s process_id=%s",
                    session_id,
                    process_id,
                )
            else:
                trace.set_nudge(armed=False, sent=False)
                logger.info(
                    "Voice nudges disabled by config; session_id=%s process_id=%s",
                    session_id,
                    process_id,
                )
            # ── End nudge setup ─────────────────────────────────────────────

            processing_query = query
            processing_lang = "en"
            history_user_text = query
            moderation_recent_history = "\n\n".join(format_message_pairs(history, 2))
            mobile = normalize_phone_to_mobile(user_id)
            signed_in = _is_signed_in_session(user_info, user_id)
            farmer_info = ""
            farmer_unions: list[str] = []
            ai_technician_info = ""
            farmer_cache_task = (
                asyncio.create_task(get_or_fetch_farmer_data(mobile))
                if mobile
                else None
            )

            # Kick off content moderation in parallel with pretranslation.
            # Moderation receives the raw native-language text so it does
            # not need to wait for pretranslation to finish.
            moderation_started_at = time.monotonic()
            # Stamp the true completion time so deferring the await (below) does
            # not inflate the reported moderation latency: the verdict is now
            # consumed after the agent's prefill, which can be later than when the
            # moderation call actually finished.
            moderation_done_at: dict = {"t": None}
            moderation_task = asyncio.create_task(
                check_moderation(
                    text=query,
                    source_lang=requested_source_lang,
                    recent_history_text=moderation_recent_history,
                )
            )
            moderation_task.add_done_callback(
                lambda _t: moderation_done_at.__setitem__("t", time.monotonic())
            )

            if requested_source_lang not in {"en", "english"}:
                _pretrans_provider_label = "vllm" if is_oss else "openai"
                _pretrans_model = OSS_PRETRANSLATION_MODEL if is_oss else OPENAI_PRETRANSLATION_MODEL
                logger.info(
                    "Translation pipeline enabled; pretranslating %s -> en with %s (variant=%s)",
                    requested_source_lang,
                    _pretrans_model,
                    pipeline_variant,
                )
                if await _request_is_stale("before_query_pretranslation"):
                    moderation_task.cancel()
                    return
                try:
                    with trace.stage(
                        "pretranslation",
                        as_type="generation",
                        input=trace.metadata.get("query"),
                        metadata={
                            "provider": _pretrans_provider_label,
                            "source_lang": requested_source_lang,
                            "pipeline_variant": pipeline_variant,
                        },
                        model=_pretrans_model,
                    ):
                        if is_oss:
                            processing_query = await translate_to_english_with_oss_vllm(
                                text=query,
                                source_lang=requested_source_lang,
                            )
                        else:
                            processing_query = await translate_to_english_with_gpt5_mini(
                                text=query,
                                source_lang=requested_source_lang,
                            )
                    trace.set_pretranslation(
                        text=processing_query,
                        provider=_pretrans_provider_label,
                        fallback_used=False,
                    )
                    history_user_text = processing_query or _canonical_history_user_text("low_confidence")
                except Exception as e:
                    logger.error(
                        "OpenAI pretranslation failed for session_id=%s source_lang=%s model=%s error=%s",
                        session_id,
                        requested_source_lang,
                        OPENAI_PRETRANSLATION_MODEL,
                        e,
                    )
                    try:
                        logger.info("Falling back to TranslateGemma pretranslation for session_id=%s", session_id)
                        with trace.stage(
                            "pretranslation_fallback",
                            as_type="generation",
                            input=trace.metadata.get("query"),
                            metadata={"provider": "translategemma", "source_lang": requested_source_lang},
                        ):
                            processing_query = await translate_to_english_with_structured_fallback(
                                text=query,
                                source_lang=requested_source_lang,
                            )
                        trace.set_pretranslation(
                            text=processing_query,
                            provider="translategemma",
                            fallback_used=True,
                        )
                        history_user_text = processing_query or _canonical_history_user_text("low_confidence")
                    except Exception as fallback_error:
                        logger.error(
                            "TranslateGemma pretranslation fallback failed for session_id=%s error=%s",
                            session_id,
                            fallback_error,
                        )
                        processing_query = ""
                        trace.set_pretranslation(
                            text=processing_query,
                            provider="failed",
                            fallback_used=True,
                        )
                        history_user_text = _canonical_history_user_text("pretranslation_failed")

            else:
                history_user_text = query
                trace.set_pretranslation(
                    text=query,
                    provider="none",
                    fallback_used=False,
                )

            # ── Content moderation: deferred gate (runs with the agent) ──────
            # check_moderation() was kicked off at the top of the turn and runs
            # concurrently with pretranslation, the farmer-context load, AND the
            # answer agent's prefill/generation below. We deliberately do NOT
            # block on it here. The verdict is resolved lazily — via
            # _resolve_moderation() — only at the points that can emit
            # caller-facing output for a non-fast-path turn:
            #   1. the empty-pretranslation short-circuit, and
            #   2. just before the agent's first streamed chunk is emitted.
            # This takes the ~1.5s moderation call off the critical path on
            # warm-cache turns (cold farmer fetches already hid it). A rejected
            # query is still declined before any answer reaches the caller, and
            # side-effecting booking tools self-gate on the same verdict via
            # deps.ensure_in_scope(), so optimistic agent execution can never turn
            # a rejected query into a real booking write. Fail-open on any
            # unexpected moderation exception — a flaky check must never drop a
            # real farmer call.
            _moderation_resolved = False
            _moderation_verdict: Optional[ModerationVerdict] = None

            async def _resolve_moderation() -> Optional[ModerationVerdict]:
                nonlocal _moderation_resolved, _moderation_verdict
                if _moderation_resolved:
                    return _moderation_verdict
                _moderation_resolved = True
                try:
                    _moderation_verdict = await moderation_task
                    done_t = moderation_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "moderation",
                        (done_t - moderation_started_at) * 1000.0,
                        source_lang=requested_source_lang,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as moderation_error:
                    done_t = moderation_done_at["t"] or time.monotonic()
                    trace.attach_stage_timing(
                        "moderation",
                        (done_t - moderation_started_at) * 1000.0,
                        status="error",
                        source_lang=requested_source_lang,
                    )
                    logger.error(
                        "Moderation task raised unexpectedly for session_id=%s error=%s",
                        session_id,
                        moderation_error,
                    )
                    _moderation_verdict = None
                trace.set_moderation(_moderation_verdict)
                if _moderation_verdict is not None:
                    logger.info(
                        "Moderation verdict: category=%s rejected=%s failed_open=%s reason=%r session_id=%s process_id=%s",
                        _moderation_verdict.category,
                        _moderation_verdict.rejected,
                        _moderation_verdict.failed_open,
                        _moderation_verdict.reason,
                        session_id,
                        process_id,
                    )
                return _moderation_verdict

            async def _moderation_decline_stream(verdict: ModerationVerdict):
                """Emit the canned decline and write history for a rejected query."""
                trace.set_route("moderation_rejected")
                if nudge_task and not nudge_task.done():
                    nudge_task.cancel()
                    trace.set_nudge(cancel_reason="moderation_rejected")
                    logger.info(
                        "Nudge canceled (moderation rejected); session_id=%s process_id=%s",
                        session_id,
                        process_id,
                    )
                    try:
                        await nudge_task
                    except asyncio.CancelledError:
                        pass
                decline_en = (
                    verdict.decline_text_en()
                    or "This helpline only handles dairy farming and animal husbandry questions."
                )
                decline_for_caller = await _render_text_for_caller(decline_en, requested_target_lang)
                decline_user_text = _canonical_history_user_text("moderation_reject")
                decl_req, decl_resp = _history_pair(decline_user_text, decline_en)
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, decl_req, decl_resp])
                trace.set_outcome("moderation_rejected")
                yield _emit(_prepare_voice_output(decline_for_caller, requested_target_lang))

            # ── Empty-pretranslation guard ───────────────────────────────
            # Only short-circuit when pretranslation produced no usable text
            # at all (i.e. both primary and fallback failed). True noise still
            # routes to the agent, which is better at asking for
            # clarification in context than a canned global retry.
            if (
                requested_source_lang not in {"en", "english"}
                and not (processing_query or "").strip()
            ):
                # This short-circuits the agent, so resolve moderation here: a
                # rejected query must be declined rather than asked to repeat.
                _verdict = await _resolve_moderation()
                if _verdict is not None and _verdict.rejected:
                    if await _request_is_stale("after_moderation_reject"):
                        return
                    async for _c in _moderation_decline_stream(_verdict):
                        yield _c
                    return
                trace.set_route("pretranslation_empty")
                logger.info(
                    "Pretranslation produced no usable text; asking to repeat - session_id=%s process_id=%s query=%r",
                    session_id, process_id, query,
                )
                low_conf_resp_for_history = _FRAGMENT_RESPONSES["en"]
                low_conf_resp_for_caller = await _canned_for_caller(low_conf_resp_for_history, requested_target_lang, _FRAGMENT_RESPONSES)
                low_conf_req, low_conf_rsp = _history_pair(
                    history_user_text or _canonical_history_user_text("low_confidence"),
                    low_conf_resp_for_history,
                )
                with trace.stage("history_write"):
                    await update_message_history(session_id, [*history, low_conf_req, low_conf_rsp])
                trace.set_outcome("pretranslation_empty")
                yield _emit(_prepare_voice_output(low_conf_resp_for_caller, requested_target_lang))
                return

            if farmer_cache_task is not None:
                try:
                    with trace.stage("farmer_context"):
                        envelope = await farmer_cache_task
                    farmer_info = _build_compact_farmer_summary(envelope)
                    farmer_unions = _collect_farmer_unions(envelope)
                    with trace.stage("scheme_summary"):
                        scheme_summary = await _build_union_scheme_summary(farmer_unions)
                    if scheme_summary:
                        farmer_info = f"{farmer_info}\n{scheme_summary}" if farmer_info else scheme_summary
                    ai_technician_info = _build_ai_technician_summary(envelope)
                    trace.set_farmer_context(
                        source=getattr(envelope, "source", None) if envelope else None,
                        stale=getattr(envelope, "stale", None) if envelope else None,
                        unions=farmer_unions,
                        farmer_info_chars=len(farmer_info),
                        technician_info_chars=len(ai_technician_info),
                    )
                    logger.info(
                        "Farmer summary loaded from cache for mobile %s source=%s stale=%s unions=%s summary_chars=%s technician_chars=%s",
                        mobile,
                        getattr(envelope, "source", None) if envelope else None,
                        getattr(envelope, "stale", None) if envelope else None,
                        farmer_unions,
                        len(farmer_info),
                        len(ai_technician_info),
                    )
                    if mobile and should_refresh_farmer_data(envelope):
                        await enqueue_farmer_refresh(mobile)
                        logger.info(
                            "Farmer cache refresh scheduled in background for mobile %s stale=%s status=%s",
                            mobile,
                            getattr(envelope, "stale", None) if envelope else None,
                            getattr(envelope, "lookupStatus", None) if envelope else None,
                        )
                except Exception as e:
                    logger.warning(f"Failed to load farmer summary for mobile {mobile}: {e}")

            logger.info(f"User info: {user_info}")
            deps = FarmerContext(
                query=processing_query,
                lang_code=processing_lang,
                target_lang=requested_target_lang,
                provider=provider,
                session_id=session_id,
                process_id=process_id,
                farmer_info=farmer_info,
                farmer_unions=farmer_unions,
                ai_technician_info=ai_technician_info,
                signed_in=signed_in,
                mobile=mobile,
            )
            # Let side-effecting tools (bookings) self-gate on the concurrent
            # moderation verdict before performing any write.
            deps.set_moderation_task(moderation_task)

            message_pairs = "\n\n".join(format_message_pairs(history, 3))
            logger.info(f"Message pairs: {message_pairs}")
            user_message = deps.get_user_message()
            runtime_context_request = _build_runtime_context_request(deps)
            logger.info(f"Running agent with user message: {user_message}")

            cleaned_history = clean_message_history_for_openai(history)
            if len(cleaned_history) != len(history):
                logger.warning(f"Cleaned {len(history) - len(cleaned_history)} orphaned tool calls from history")
                if not await _request_is_stale("before_cleaned_history_write"):
                    await update_message_history(session_id, cleaned_history)
                history = cleaned_history

            trimmed_history = trim_history(
                history,
                max_tokens=32_000,
                include_system_prompts=False,
                include_tool_calls=True,
            )
            logger.info(f"Trimmed history length: {len(trimmed_history)} messages")
            # pydantic-ai's Agent(instructions=STATIC_VOICE_SYSTEM_PROMPT) already
            # emits the system prompt on every run. Prepending another
            # SystemPromptPart here produced two identical role=system messages
            # (~33 KB each) per turn, which both inflates context and dilutes
            # attention to the actual runtime context. Keep only the runtime
            # context request, which carries the per-turn deps (today's date,
            # farmer profile, ambiguity hints, voice answer mode).
            # Stable context first → [system][stable-context][history] is a single
            # growing prefix vLLM can cache across turns. Per-query hints (if any)
            # go last, right before the user message, so they never break it.
            model_input_history = [runtime_context_request, *trimmed_history]
            query_hints_request = _build_query_hints_request(deps)
            if query_hints_request is not None:
                model_input_history.append(query_hints_request)
            active_agent = voice_agent_signed_in if (signed_in and mobile) else voice_agent
            usage_limits = UsageLimits(request_limit=6 if (signed_in and mobile) else 4)

            if settings.retrieval_audit_log:
                logger.info(
                    "RETRIEVAL_AUDIT query=%r session_id=%s process_id=%s target_lang=%s",
                    processing_query,
                    session_id,
                    process_id,
                    requested_target_lang,
                )

            with boundary_capture_context(
                session_id=session_id,
                process_id=process_id,
                user_query=processing_query,
            ):
                # Restored token streaming on pydantic-ai 1.x. run_stream now drives
                # the full tool-call loop past a tool-call-only first response (the
                # 0.2.4 stall that previously forced a blocking run()), then streams
                # the final English text. We pipe those en deltas straight into the
                # en->gu batch translator below, so agent generation and output
                # translation overlap instead of running strictly back-to-back.
                agent_started_at = time.monotonic()
                _agent_output = ""
                async with active_agent.run_stream(
                    user_prompt=user_message,
                    message_history=model_input_history,
                    deps=deps,
                    usage_limits=usage_limits,
                    model=request_model,
                ) as response_stream:
                    # debounce_by=0 disables pydantic-ai's default 100ms token
                    # debounce so the first agent delta reaches the translation
                    # stage immediately (every ms counts for phone TTFT). Our own
                    # sentence batching downstream re-aggregates the smaller chunks.
                    stream_iter = response_stream.stream_text(delta=True, debounce_by=0)
                    first_text_chunk_received = False
                    sentence_buffer = ""
                    translation_batch: list[str] = []
                    batch_word_count = 0
                    async def _yield_translated_text(text_to_translate: str) -> AsyncGenerator[str, None]:
                        if not text_to_translate:
                            return
                        text_to_translate = _guard_identity_drift(text_to_translate)
                        try:
                            with trace.stage(
                                "output_translation",
                                as_type="generation",
                                input={"chars": len(text_to_translate)},
                                metadata={"target_lang": requested_target_lang},
                            ):
                                async for chunk in _voice_output_stream(
                                    text_to_translate,
                                    requested_target_lang,
                                ):
                                    if await _request_is_stale("during_output_translation"):
                                        return
                                    cleaned = (
                                        _prepare_voice_output(chunk, requested_target_lang)
                                        if isinstance(chunk, str) and chunk
                                        else chunk
                                    )
                                    if isinstance(cleaned, str) and cleaned.strip():
                                        trace.mark("first_translation_chunk_ms")
                                    yield cleaned
                        except Exception as e:
                            trace.increment("output_translation_errors")
                            logger.error(
                                "Translation pipeline output translation failed for session_id=%s error=%s",
                                session_id,
                                e,
                            )
                            trouble = TRANSLATION_TROUBLE_MESSAGE.get(
                                requested_target_lang,
                                TRANSLATION_TROUBLE_MESSAGE["en"],
                            )
                            yield trouble

                    # Deferred moderation gate: resolve the concurrently-running
                    # verdict now, before emitting ANY caller-facing chunk. The
                    # agent has already done its prefill/tool calls (booking tools
                    # self-gated on this verdict); if the query was rejected we
                    # decline here and the agent's streamed output is discarded,
                    # never reaching the caller.
                    _verdict = await _resolve_moderation()
                    if _verdict is not None and _verdict.rejected:
                        if not await _request_is_stale("after_moderation_reject"):
                            async for _c in _moderation_decline_stream(_verdict):
                                yield _c
                        return

                    try:
                        async for chunk in stream_iter:
                            if await _request_is_stale("during_agent_stream"):
                                break

                            if isinstance(chunk, str) and chunk:
                                if not _agent_output and chunk.strip():
                                    trace.mark("first_agent_text_ms")
                                _agent_output += chunk

                            if not needs_output_translation:
                                if (
                                    not first_text_chunk_received
                                    and isinstance(chunk, str)
                                    and chunk
                                    and chunk.strip()
                                ):
                                    first_text_chunk_received = True
                                    if nudge_task:
                                        nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (first text chunk received); session_id=%s process_id=%s chunk_preview=%s",
                                            session_id,
                                            process_id,
                                            chunk[:50] if len(chunk) > 50 else chunk,
                                        )
                                        try:
                                            await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                    trace.set_nudge(cancel_reason="first_text_chunk_received")

                                cleaned_chunk = (
                                    _prepare_voice_output(chunk, requested_target_lang)
                                    if isinstance(chunk, str) and chunk
                                    else chunk
                                )
                                if await _request_is_stale("before_direct_yield"):
                                    break
                                yield _emit(cleaned_chunk)
                                continue

                            sentence_buffer += chunk
                            ready_units, remaining = extract_translation_units(sentence_buffer)
                            if ready_units:
                                for unit in ready_units:
                                    candidate_units = [unit]
                                    if len(unit) >= VOICE_TRANSLATION_BATCH_CHAR_LIMIT:
                                        candidate_units = []
                                        remaining_unit = unit
                                        while remaining_unit:
                                            head, tail = _split_voice_batch_text(remaining_unit)
                                            if not tail or head == remaining_unit:
                                                candidate_units.append(remaining_unit)
                                                break
                                            candidate_units.append(head)
                                            remaining_unit = tail

                                    for candidate in candidate_units:
                                        translation_batch.append(candidate)
                                        batch_word_count += len(candidate.split())
                                        batch_text = "".join(translation_batch)

                                        if should_translate_batch(batch_text, batch_word_count, is_first_batch=not first_text_chunk_received):
                                            async for translated_chunk in _yield_translated_text(batch_text):
                                                if (
                                                    not first_text_chunk_received
                                                    and isinstance(translated_chunk, str)
                                                    and translated_chunk
                                                    and translated_chunk.strip()
                                                ):
                                                    first_text_chunk_received = True
                                                    if nudge_task:
                                                        nudge_task.cancel()
                                                        logger.info(
                                                            "Nudge canceled (first translated chunk received); session_id=%s process_id=%s",
                                                            session_id,
                                                            process_id,
                                                        )
                                                        try:
                                                            await nudge_task
                                                        except asyncio.CancelledError:
                                                            pass
                                                    trace.set_nudge(cancel_reason="first_translated_chunk_received")
                                                if await _request_is_stale("before_translated_yield"):
                                                    break
                                                yield _emit(_prepare_translated_emit(translated_chunk))
                                            translation_batch = []
                                            batch_word_count = 0

                                sentence_buffer = remaining

                        if needs_output_translation and not await _request_is_stale("before_translation_flush"):
                            if translation_batch:
                                batch_text = "".join(translation_batch)
                                async for translated_chunk in _yield_translated_text(batch_text):
                                    if (
                                        not first_text_chunk_received
                                        and isinstance(translated_chunk, str)
                                        and translated_chunk
                                        and translated_chunk.strip()
                                    ):
                                        first_text_chunk_received = True
                                        if nudge_task: nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (final translated batch); session_id=%s process_id=%s",
                                            session_id,
                                            process_id,
                                        )
                                        try:
                                            if nudge_task:
                                                await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                        trace.set_nudge(cancel_reason="final_translated_batch")
                                    if await _request_is_stale("before_final_translated_yield"):
                                        break
                                    yield _emit(_prepare_translated_emit(translated_chunk))

                            if sentence_buffer.strip():
                                async for translated_chunk in _yield_translated_text(sentence_buffer):
                                    if (
                                        not first_text_chunk_received
                                        and isinstance(translated_chunk, str)
                                        and translated_chunk
                                        and translated_chunk.strip()
                                    ):
                                        first_text_chunk_received = True
                                        if nudge_task: nudge_task.cancel()
                                        logger.info(
                                            "Nudge canceled (tail translated fragment); session_id=%s process_id=%s",
                                            session_id,
                                            process_id,
                                        )
                                        try:
                                            if nudge_task:
                                                await nudge_task
                                        except asyncio.CancelledError:
                                            pass
                                        trace.set_nudge(cancel_reason="tail_translated_fragment")
                                    if await _request_is_stale("before_tail_translated_yield"):
                                        break
                                    yield _emit(_prepare_translated_emit(translated_chunk))
                    except StopAsyncIteration:
                        pass
                    except RuntimeError as e:
                        if "StopAsyncIteration" in str(e) or "anext()" in str(e):
                            # anext() errors occur on superseded processes during
                            # teardown — the final process_id has its own generator
                            # and is unaffected, so this is just cleanup noise.
                            logger.debug(
                                "Suppressed stream runtime error (superseded process teardown) - session_id=%s process_id=%s error=%s",
                                session_id,
                                process_id,
                                e,
                            )
                        else:
                            raise
                    finally:
                        if nudge_task and not nudge_task.done():
                            if nudge_task: nudge_task.cancel()
                            trace.set_nudge(cancel_reason="stream_ended")
                            logger.info(
                                "Nudge canceled (stream ended); session_id=%s process_id=%s",
                                session_id,
                                process_id,
                            )
                            try:
                                await nudge_task
                            except asyncio.CancelledError:
                                pass

                    logger.info(f"Streaming complete for session {session_id}")
                    _agent_output = _agent_output.strip()
                    new_messages = response_stream.new_messages()
                    trace.attach_stage_timing(
                        "agent",
                        (time.monotonic() - agent_started_at) * 1000.0,
                        signed_in=bool(signed_in and mobile),
                        request_limit=usage_limits.request_limit,
                        pipeline_variant=pipeline_variant,
                        model=request_model_name,
                        provider=request_provider,
                    )
                    trace.set_agent(
                        signed_in=bool(signed_in and mobile),
                        output=_agent_output,
                        new_messages=new_messages,
                    )

            # If the LLM called signal_conversation_state("conversation_closing"),
            # append the termination token so RAYA disconnects the call.
            # We scan the agent's new messages for the tool call rather than
            # using contextvars, because pydantic-ai runs tools in child tasks
            # whose contextvar writes don't propagate back to the caller.
            closing = any(
                getattr(part, "tool_name", None) == "signal_conversation_state"
                and "conversation_closing" in (getattr(part, "args_as_json_str", lambda: "")() if callable(getattr(part, "args_as_json_str", None)) else str(getattr(part, "args", "")))
                for msg in new_messages
                for part in (getattr(msg, "parts", None) or [])
            )
            if closing and not await _request_is_stale("before_goodbye"):
                goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang,
                    TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                )
                logger.info(
                    "Appending goodbye after conversation_closing signal; session_id=%s process_id=%s",
                    session_id, process_id,
                )
                yield _emit(" " + goodbye)

            if await _request_is_stale("before_history_write"):
                return

            messages = [*history, *new_messages]
            logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
            with trace.stage("history_write"):
                await update_message_history(session_id, messages)
            if trace.outcome is None:
                trace.set_outcome("success")
    except Exception as exc:
        trace.finish(trace.outcome or "error", error=exc)
        raise
    finally:
        release_started_at = time.monotonic()
        released = await release_session_request_ownership(owner)
        trace.attach_stage_timing(
            "ownership_release",
            (time.monotonic() - release_started_at) * 1000.0,
            released=released,
        )
        if owner is not None:
            logger.info(
                "Session ownership released - session_id=%s process_id=%s epoch=%s released=%s",
                session_id,
                process_id,
                owner.epoch,
                released,
            )
        trace.finish(trace.outcome or "success")
