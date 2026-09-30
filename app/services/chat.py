from contextlib import aclosing, contextmanager, nullcontext
from types import MappingProxyType
from typing import Any, AsyncGenerator, Mapping, Optional
from functools import lru_cache
import regex
import re
import time
from fastapi import BackgroundTasks
from agents.agrinet import agrinet_agent
from agents.doctor import doctor_agent
from agents.moderation import doctor_moderation_agent, moderation_agent
from app import llm_core
from app.llm_core import Step as _LlmStep
from helpers.utils import get_logger
from app.utils import (
    update_message_history,
    trim_history,
    format_message_pairs,
    set_cache,
)
from app.tasks.suggestions import create_suggestions
from app.config import settings
from app.core.cache import cache
from agents.deps import FarmerContext
from agents.farmer_context import FarmerContextBundle, get_farmer_context_bundle_by_mobile
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.session_shc import get_session_shc_context
from app.services.translation import (
    translate_text,
    pretranslate_with_tier,
    translate_text_stream_fast,
    INDIAN_LANGUAGES,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from app.services.identity_profile import (
    build_doctor_identity_response,
    build_identity_profile_table,
    is_identity_query,
)
from app.personas import ChatPersona
from app.chat_artifacts import encode_chat_artifacts
from app.services.telemetry_stamps import CHAT_TURN_V1_ROOT, chat_turn_v1_input, chat_turn_v1_metadata
from app.channels.base import ChannelProfile
from app.turn.types import (
    AgentActivityEmission,
    AgentInput,
    ArtifactEmission,
    ClassifierResult,
    DeferredScheduler,
    Emission,
    Pretranslated,
    SideChannelEmission,
    SideChannelSender,
    StalenessCheck,
    Surface,
    SurfaceProfile,
    TextEmission,
    Turn,
    TurnBackground,
)


class SentenceSegmenter:
    sep = 'ŽžŽžSentenceSeparatorŽžŽž'
    latin_terminals = '!?.'
    jap_zh_terminals = '。！？'
    terminals = latin_terminals + jap_zh_terminals

    def __init__(self):
        terminals = self.terminals
        self._re = [
            (regex.compile(r'(\P{N})([' + terminals + r'])(\p{Z}*)'),
             r'\1\2\3' + self.sep),
            (regex.compile(r'(' + terminals + r')(\P{N})'),
             r'\1' + self.sep + r'\2'),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [t for t in line.split(self.sep) if t != '']


sentence_segmenter = SentenceSegmenter()


def extract_complete_sentences(text: str):
    if not text:
        return [], ""
    sentences = sentence_segmenter(text)
    if len(sentences) <= 1:
        return [], text
    complete = sentences[:-1]
    incomplete = sentences[-1]
    return complete, incomplete


def _batch_starts_new_line_or_list(text: str) -> bool:
    """True if text starts with a newline or list marker (bullet/numbered), so we should preserve a line break before it when streaming."""
    if not text or not text.strip():
        return False
    stripped = text.lstrip()
    if text != stripped:
        return True  # leading whitespace (e.g. newline) — lost when we split into sentence batches
    if stripped.startswith(("-", "•")) and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    if stripped.startswith("*") and (len(stripped) == 1 or stripped[1:2].isspace() or stripped[1:2] == "."):
        return True
    if re.match(r"^\d+\.\s", stripped):
        return True
    return False


def should_translate_batch(batch_text: str, word_count: int) -> bool:
    # Tuned for low-latency streaming while keeping reasonable batch size
    MIN_WORDS = 15
    MAX_WORDS = 80

    if word_count < MIN_WORDS:
        # For very short answers, still allow early flush when a sentence ends
        text_end = batch_text.rstrip()
        if text_end.endswith(('.', '!', '?')) and word_count >= 5:
            return True
        return False
    if word_count >= MAX_WORDS:
        return True

    text_end = batch_text.rstrip()

    # Paragraph break
    if text_end.endswith('\n\n'):
        return True

    # Bullet/list endings
    if text_end.endswith('\n') and len(batch_text.split('\n')) > 1:
        lines = batch_text.rstrip('\n').split('\n')
        last_line = lines[-1].strip()
        if last_line.startswith(('-', '*', '•')):
            return True
        if re.match(r'^\d+\.', last_line):
            return True

    # Sentence end
    if text_end.endswith(('.', '!', '?')):
        return True

    return False


_DOCTOR_PROVENANCE_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*{0,2})?"
    r"(?:sources?|references?|citations?|cited\s+sources?|document\s+sources?|"
    r"source\s+documents?|સ્ત્રોત|સ્રોત|સંદર્ભ)"
    r"(?:\*{0,2})?\s*[:：-]",
    re.IGNORECASE,
)
_DOCTOR_INLINE_PROVENANCE_RE = re.compile(
    r"\s*\((?:sources?|references?|citations?|સ્ત્રોત|સ્રોત|સંદર્ભ)\s*:[^)]*\)",
    re.IGNORECASE,
)


def sanitize_doctor_answer(text: str) -> str:
    """Remove source-document audit text from a Doctor-facing answer.

    Prompt compliance is probabilistic, so the user-facing contract is enforced
    here as well. Clinical content is preserved; only explicitly labelled
    provenance lines and inline parenthetical source tags are removed.
    """
    if not text:
        return text
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        if _DOCTOR_PROVENANCE_LINE_RE.match(line):
            continue
        kept.append(_DOCTOR_INLINE_PROVENANCE_RE.sub("", line))
    return "".join(kept).strip()


async def _sanitize_doctor_stream(source):
    """Line-buffer a stream so provenance labels split across chunks are caught."""
    buffer = ""
    async for chunk in source:
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not _DOCTOR_PROVENANCE_LINE_RE.match(line):
                yield _DOCTOR_INLINE_PROVENANCE_RE.sub("", line) + "\n"
    if buffer and not _DOCTOR_PROVENANCE_LINE_RE.match(buffer):
        yield _DOCTOR_INLINE_PROVENANCE_RE.sub("", buffer)



logger = get_logger(__name__)
SUGGESTIONS_PENDING_TTL = 30
# The Gemma pre/post-translation pipeline is the only chat execution path.
# Kept as a named constant purely so existing Langfuse trace names, tags and
# metadata keep the value dashboards already filter on.
_PIPELINE_NAME = "translation"
GENERIC_UNAVAILABLE_MESSAGE_EN = (
    "I am unable to process your request right now. Please try again later."
)
GENERIC_UNAVAILABLE_MESSAGE_GU = (
    "હાલમાં હું તમારી વિનંતી પ્રક્રિયા કરી શકતી નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો."
)
GENERIC_UNAVAILABLE_MESSAGE_BN = (
    "এই মুহূর্তে আমি আপনার অনুরোধটি প্রক্রিয়া করতে পারছি না। অনুগ্রহ করে কিছুক্ষণ পরে আবার চেষ্টা করুন।"
)
GENERIC_UNAVAILABLE_MESSAGE_PA = (
    "ਇਸ ਸਮੇਂ ਮੈਂ ਤੁਹਾਡੀ ਬੇਨਤੀ ਤੇ ਕਾਰਵਾਈ ਨਹੀਂ ਕਰ ਸਕਦੀ। ਕਿਰਪਾ ਕਰਕੇ ਥੋੜ੍ਹੇ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।"
)
GENERIC_UNAVAILABLE_MESSAGE_MR = (
    "सध्या मी तुमची विनंती पूर्ण करू शकत नाही. कृपया थोड्या वेळाने पुन्हा प्रयत्न करा."
)
# Localized fallback when translating a system message fails. Hindi has none, so
# it falls back to the English text.
_UNAVAILABLE_MESSAGE_BY_LANG = {
    "gu": GENERIC_UNAVAILABLE_MESSAGE_GU,
    "gujarati": GENERIC_UNAVAILABLE_MESSAGE_GU,
    "bn": GENERIC_UNAVAILABLE_MESSAGE_BN,
    "bengali": GENERIC_UNAVAILABLE_MESSAGE_BN,
    "pa": GENERIC_UNAVAILABLE_MESSAGE_PA,
    "punjabi": GENERIC_UNAVAILABLE_MESSAGE_PA,
    "mr": GENERIC_UNAVAILABLE_MESSAGE_MR,
    "marathi": GENERIC_UNAVAILABLE_MESSAGE_MR,
}

# Per-language kill switches: settings flag -> the language codes it governs.
# Gujarati is always on. A disabled language drops out of both pretranslation
# (src->en) and output translation (en->target), so its requests are served in
# English like an unsupported language. Adding a language is one entry here.
_LANGUAGE_KILL_SWITCHES: dict[str, tuple[str, ...]] = {
    "hindi_chat_enabled": ("hi", "hindi"),
    "bengali_chat_enabled": ("bn", "bengali"),
    "punjabi_chat_enabled": ("pa", "punjabi"),
    "marathi_chat_enabled": ("mr", "marathi"),
}
_ALWAYS_ON_PRETRANSLATION_LANGS = frozenset({"gu", "gujarati"})

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

# Per-turn resolved-pipeline-config tracer (tracing-only; no behaviour change).
from app.llm_core import trace as _pipeline_trace
from app.channels.chat import profile_for as _profile_for




def _record_turn_outcome(outcome: str, session_id_safe: str) -> None:
    """Emit how the turn ended, on every exit path.

    Deliberately synchronous and never-raising: it runs in a ``finally`` that can
    execute during GeneratorExit, where awaiting is not safe.
    """
    if not get_langfuse_client:
        return
    try:
        get_langfuse_client().score_current_trace(
            name="turn_outcome",
            value=outcome,
            data_type="CATEGORICAL",
            comment="How the turn ended: success | cancelled | error",
        )
    except Exception as e:
        logger.debug("Langfuse: turn_outcome score failed: %s", e)


def _record_trace_output(output: str, label: str) -> None:
    """Best-effort: set the current trace's output. Telemetry never breaks a turn."""
    if not get_langfuse_client:
        return
    try:
        get_langfuse_client().set_current_trace_io(output=output)
    except Exception as e:
        logger.warning("Langfuse: failed to record %s output: %s", label, e)


def _disabled_chat_langs() -> set[str]:
    """Language codes whose kill switch is off. Read per turn so flags apply live."""
    disabled: set[str] = set()
    for flag, codes in _LANGUAGE_KILL_SWITCHES.items():
        if not getattr(settings, flag, True):
            disabled.update(codes)
    return disabled


def _pretranslation_source_langs(disabled_langs: set[str]) -> set[str]:
    enabled = {code for codes in _LANGUAGE_KILL_SWITCHES.values() for code in codes}
    return set(_ALWAYS_ON_PRETRANSLATION_LANGS) | (enabled - disabled_langs)


async def _localize_system_text(
    text_en: str,
    *,
    target_lang: str,
    disabled_langs: set[str],
    execution,
    max_output_chars: int | None,
    request_id: str,
) -> str:
    """Localize a short system-generated message to the target language.

    Falls back to a canned "unavailable" message in the target language when
    translation fails, or to the English text if there is none.
    """
    if not text_en or not target_lang:
        return text_en
    lang = target_lang.lower()
    if lang in {"english", "en"} or lang in disabled_langs or lang not in INDIAN_LANGUAGES:
        return text_en
    try:
        return await translate_text(
            text=text_en,
            source_lang="english",
            target_lang=target_lang,
            max_output_chars=max_output_chars,
            execution=execution,
        )
    except Exception as e:
        logger.warning(
            "request_id=%s system text translation failed target_lang=%s error=%s",
            request_id,
            target_lang,
            e,
        )
        return _UNAVAILABLE_MESSAGE_BY_LANG.get(lang, text_en)


async def _pretranslate_query(
    query: str,
    *,
    source_lang: str,
    target_lang: str,
    disabled_langs: set[str],
    execution,
    request_id: str,
) -> tuple[str, str]:
    """Translate the farmer's query to English when the source language allows it.

    Returns ``(processing_query, processing_lang)``. On failure the original
    query is kept and the agent is asked to answer in ``target_lang``.
    """
    processing_lang = "en" if target_lang.lower() in disabled_langs else target_lang
    if source_lang.lower() not in _pretranslation_source_langs(disabled_langs):
        return query, processing_lang

    pretrans_info = execution.info(_LlmStep.PRE_TRANSLATION)
    logger.info(
        "request_id=%s variant=%s pretranslating %s->en with %s/%s",
        request_id,
        execution.profile_name,
        source_lang,
        pretrans_info.provider,
        pretrans_info.model_name,
    )
    try:
        translated = await execution.run_adapter(
            _LlmStep.PRE_TRANSLATION,
            lambda tier: pretranslate_with_tier(tier, text=query, source_lang=source_lang),
        )
    except Exception as e:
        logger.error(
            "request_id=%s pretranslation_success=False source_lang=%s error=%s",
            request_id,
            source_lang,
            e,
        )
        return query, target_lang
    logger.info(
        "request_id=%s pretranslation_success=True source_preview=%s translated_preview=%s",
        request_id,
        query[:80],
        translated[:80],
    )
    return translated, "en"


async def _chat_pretranslation(turn: Turn, *, execution, background) -> Pretranslated:
    """Chat's pretranslation, for ``SurfaceProfile.pretranslation``. It has no
    answer of its own to give, so the background set is not needed."""
    processing_query, processing_lang = await _pretranslate_query(
        turn.query,
        source_lang=turn.source_lang,
        target_lang=turn.target_lang,
        disabled_langs=_disabled_chat_langs(),
        execution=execution,
        request_id=turn.session_id,
    )
    return Pretranslated(query=processing_query, lang=processing_lang)


async def _load_farmer_context(
    persona: ChatPersona, user_info: Mapping[str, Any], request_id: str
) -> tuple[FarmerContextBundle, str]:
    """Cache-first farmer context for the phone in the JWT, plus its profile status.

    The status (found / anonymous / not_found / unavailable) gates farmer-only
    tools and tells the agent what it cannot do; see FarmerContext.
    """
    if persona != "farmer" or not user_info or not user_info.get("phone"):
        return FarmerContextBundle(markdown=""), "anonymous"
    try:
        bundle = await get_farmer_context_bundle_by_mobile(user_info["phone"])
    except Exception as e:
        logger.warning(f"request_id={request_id} farmer_context_fetch_failed={e}")
        logger.info("request_id=%s farmer_profile_status=unavailable", request_id)
        return FarmerContextBundle(markdown=""), "unavailable"
    status = "found" if bundle.found else "not_found"
    logger.info(f"request_id={request_id} farmer_context_length={len(bundle.markdown)}")
    logger.info("request_id=%s farmer_unions=%s", request_id, bundle.unions)
    logger.info("request_id=%s farmer_district=%s", request_id, bundle.location.get("district"))
    logger.info("request_id=%s farmer_profile_status=%s", request_id, status)
    return bundle, status


def _build_deps(
    *,
    query: str,
    session_id: str,
    lang_code: str,
    persona: ChatPersona,
    channel: ChannelProfile,
    user_info: Mapping[str, Any],
    farmer_context: tuple[FarmerContextBundle, str],
) -> FarmerContext:
    """The ONE FarmerContext of a turn, handed to pydantic-ai as ``deps``.

    This is the only way trusted identity reaches the agent: the caller's phone
    comes from the verified JWT, tools read it from ``ctx.deps`` and validate
    model-authored arguments against it, so the model never chooses whose data
    a tool touches.
    """
    bundle, farmer_profile_status = farmer_context
    return FarmerContext(
        query=query,
        session_id=session_id,
        lang_code=lang_code,
        farmer_info=bundle.markdown,
        farmer_unions=bundle.unions,
        farmer_profile_status=farmer_profile_status,
        farmer_district=bundle.location.get("district") or None,
        farmer_village=bundle.location.get("village") or None,
        farmer_state=bundle.location.get("state") or None,
        response_max_chars=channel.response_max_chars,
        supports_rich_artifacts=channel.supports_rich_artifacts,
        # Normalized caller phone — the micro-loan tool reads this from deps so it
        # never has to trust an LLM-supplied number. None for anonymous sessions.
        mobile=(
            normalize_phone_to_mobile(user_info["phone"])
            if persona == "farmer" and user_info and user_info.get("phone")
            else None
        ),
        persona=persona,
    )


async def _write_short_circuit_history(history, decision, key: str, request_id: str) -> None:
    """Persist what a classifier or the gate answered, when it asks for that."""
    if decision.history_pair is None:
        return
    messages = [*history, *decision.history_pair]
    logger.info(
        "request_id=%s updating_history_%s_path=True total_messages=%s",
        request_id,
        decision.label,
        len(messages),
    )
    await update_message_history(key, messages)


async def _stale_outcome(is_stale: Optional[StalenessCheck], reason: str) -> Optional[str]:
    """The outcome to stop with if this request should no longer speak, else None."""
    if is_stale is None:
        return None
    return await is_stale(reason)


_NO_CHUNK = object()


async def _gate_first_chunk(english_src, background):
    """Pull the agent's first chunk, then ask the gate before anything is emitted.

    The background checks keep running while the agent starts, so their latency
    stays off the turn unless one of them says no. Returns the gate's answer
    and ``None`` when it says no (the agent's stream is closed), else ``None``
    and a stream that replays the first chunk. A failure before the first chunk
    comes out of that stream, as it does without a gate: a turn the checks
    refuse is refused, and the sink sees the failure like any later one.
    """
    first = _NO_CHUNK
    error = None
    try:
        try:
            first = await english_src.__anext__()
        except StopAsyncIteration:
            pass
        except Exception as exc:
            error = exc
        decision = await background.gate()
    except BaseException:
        # A hang-up while the gate waits, or a gate that fails: the agent's
        # stream must not be left open behind the turn.
        await english_src.aclose()
        raise
    if decision is not None:
        await english_src.aclose()
        return decision, None

    async def _replayed():
        async with aclosing(english_src):
            if error is not None:
                raise error
            if first is not _NO_CHUNK:
                yield first
            async for chunk in english_src:
                yield chunk

    return None, _replayed()


async def _stream_to_client(
    english_src,
    *,
    translate_to: str | None,
    max_output_chars: int | None,
    execution,
    output_chunks: list[str],
):
    """Pass the English stream through, or sentence-batch and stream-translate it.

    Everything yielded is also appended to ``output_chunks`` for the trace.
    A batch whose translation fails is sent in English rather than dropped.
    """
    if not translate_to:
        async for chunk in english_src:
            output_chunks.append(chunk)
            yield chunk
        return

    async def translate(text: str, label: str):
        # Sentence batching drops the line break before a list item; restore it.
        if output_chunks and _batch_starts_new_line_or_list(text):
            output_chunks.append("\n")
            yield "\n"
        try:
            async for translated_chunk in translate_text_stream_fast(
                text=text,
                source_lang="english",
                target_lang=translate_to,
                max_output_chars=max_output_chars,
                execution=execution,
            ):
                output_chunks.append(translated_chunk)
                yield translated_chunk
        except Exception as e:
            logger.error(f"{label} translation failed, falling back to English: {e}")
            output_chunks.append(text)
            yield text

    sentence_buffer = ""
    batch: list[str] = []
    batch_word_count = 0
    async for chunk in english_src:
        complete_sentences, sentence_buffer = extract_complete_sentences(sentence_buffer + chunk)
        if not complete_sentences:
            continue
        batch.extend(complete_sentences)
        batch_word_count += sum(len(s.split()) for s in complete_sentences)
        batch_text = "".join(batch)
        if should_translate_batch(batch_text, batch_word_count):
            async for out in translate(batch_text, "Optimised batch"):
                yield out
            batch, batch_word_count = [], 0
    if batch:
        async for out in translate("".join(batch), "Final batch"):
            yield out
    if sentence_buffer.strip():
        async for out in translate(sentence_buffer, "Tail fragment"):
            yield out


class _ChatSink:
    """Chat's sink: English passed through or stream-translated in sentence batches.

    The Doctor persona is sanitised on both sides of translation: provenance
    labels in the English answer, and any that post-translation invents.
    Everything the caller receives is kept for the trace.
    """

    def __init__(
        self,
        turn: Turn,
        *,
        execution,
        deps,
        translate_to: str | None,
        is_stale: Optional[StalenessCheck] = None,
    ) -> None:
        # ``is_stale`` is unused: chat has no staleness check.
        self._persona = turn.persona
        self._translate_to = translate_to
        self._execution = execution
        self._max_output_chars = deps.response_max_chars
        self._chunks: list[str] = []

    def stream(self, english):
        if self._persona == "doctor":
            english = _sanitize_doctor_stream(english)
        out = _stream_to_client(
            english,
            translate_to=self._translate_to,
            max_output_chars=self._max_output_chars,
            execution=self._execution,
            output_chunks=self._chunks,
        )
        if self._persona == "doctor":
            # Defence in depth: also remove a provenance label invented by
            # post-translation rather than present in the English answer.
            out = _sanitize_doctor_stream(out)
        return out

    def final_text(self) -> str | None:
        text = "".join(self._chunks) or None
        if text and self._translate_to and self._persona == "doctor":
            text = sanitize_doctor_answer(text)
        return text


class _ChatTelemetry:
    """Chat's turn telemetry: one chat.turn.v1 root, its input, and how the turn ended.

    The metadata and input shapes are built in this module on purpose: the
    chat.turn.v1 contract test reads them from these calls in chat.py.
    """

    def __init__(self, turn: Turn, *, pipeline_profile: str, pipeline_trace) -> None:
        self._turn = turn
        self._pipeline_profile = pipeline_profile
        self._pipeline_trace = pipeline_trace
        self._session_id_safe = (turn.session_id or "")[:200]

    @contextmanager
    def root(self):
        turn = self._turn
        user_info = turn.authenticated_user
        channel = turn.channel.channel.value
        pipeline_profile = self._pipeline_profile
        persona = turn.persona
        # Langfuse: propagate session_id, metadata, and tags for dashboard filtering (max 200 chars per value)
        # Prefer phone from JWT (weburl-minted tokens) over the query-param user_id
        effective_user_id = (
            (user_info.get("phone") or user_info.get("sub")) if user_info else None
        ) or turn.user_id or "anonymous"
        effective_user_id = effective_user_id[:200]
        langfuse_metadata = chat_turn_v1_metadata(
            pipeline=_PIPELINE_NAME,
            channel=(channel or "web")[:200],
            source_lang=(turn.source_lang or "unknown").lower()[:200],
            target_lang=(turn.target_lang or "unknown").lower()[:200],
            user_id=effective_user_id,
            pipeline_profile=pipeline_profile,
            persona=persona,
        )
        langfuse_tags = [
            f"pipeline:{_PIPELINE_NAME}",
            f"pipeline_profile:{pipeline_profile}",
            f"persona:{persona}",
        ]
        # Serialize the resolved pipeline config into COMPACT flat keys and merge them
        # into the same langfuse_metadata dict propagate_attributes lands on OTEL span
        # attributes (a big nested blob is size-capped/dropped; this SDK has no
        # update_current_trace). Adds `pipeline_profile`, `pipeline_flags`, and one
        # `pc_<step>` per step (~50 chars each). Full static config is in the
        # `llm_core.full_config` boot log. Best-effort — never breaks the turn.
        _pipeline_trace.add_compact_metadata(self._pipeline_trace, langfuse_metadata)
        session_ctx = (
            propagate_attributes(
                session_id=self._session_id_safe,
                user_id=effective_user_id,
                metadata=langfuse_metadata,
                tags=langfuse_tags,
            )
            if propagate_attributes
            else nullcontext()
        )

        # THE TURN ROOT SPAN. Without it, propagate_attributes leaves no active span,
        # so every observation opened during the turn (Moderation, query_pretranslation,
        # Amul AI Agent, suggestions, each tool call) becomes its OWN top-level trace —
        # measured at 6 traces for one turn — and every trace-level write made outside
        # an observation is silently dropped by the SDK ("no active span ... skipped").
        # That is why trace input/output was missing on many turns, why the chat export
        # has gaps, and why turn-level scores never landed.
        root_ctx = (
            get_langfuse_client().start_as_current_observation(
                name=CHAT_TURN_V1_ROOT, as_type="span"
            )
            if get_langfuse_client
            else nullcontext()
        )

        with session_ctx, root_ctx:
            self._record_input(channel)
            yield

    def _record_input(self, channel: str) -> None:
        if not get_langfuse_client:
            return
        turn = self._turn
        try:
            langfuse = get_langfuse_client()
            langfuse.set_current_trace_io(
                input=chat_turn_v1_input(
                    query=turn.query,
                    channel=channel,
                    source_lang=turn.source_lang,
                    target_lang=turn.target_lang,
                    persona=turn.persona,
                )
            )
            #this is the same as the update_current_trace method,
            #but it is more explicit about the type of the output
            # and is supported by the latest version of the langfuse SDK.
            # Emit a categorical pipeline_profile score attached to the
            # *current trace*. Langfuse rolls this up to the session view,
            # so a Sessions filter "pipeline_profile = oss" works directly.
            # `score_id` is deterministic per session so subsequent traces
            # in the same session upsert the same score (no duplicates).
            try:
                langfuse.score_current_trace(
                    name="pipeline_profile",
                    value=self._pipeline_profile,
                    data_type="CATEGORICAL",
                    score_id=f"variant-{self._session_id_safe}",
                    comment="Sticky pipeline variant for this session",
                )
            except Exception as e:
                logger.warning("Langfuse: pipeline_profile score failed: %s", e)
        except Exception as e:
            logger.warning("Langfuse: failed to set trace input: %s", e)

    def record_output(self, text: str, label: str) -> None:
        _record_trace_output(text, label)

    def record_outcome(self, outcome: str) -> None:
        _record_turn_outcome(outcome, self._session_id_safe)


class _NoTelemetry:
    """For a surface that sets no telemetry, such as a test surface: no root span
    is opened and nothing is recorded."""

    @contextmanager
    def root(self):
        yield

    def record_output(self, text: str, label: str) -> None:
        pass

    def record_outcome(self, outcome: str) -> None:
        pass


_NO_TELEMETRY = _NoTelemetry()


async def _identity_classifier(turn: Turn) -> ClassifierResult | None:
    """"Who are you?" — answered from a template, without moderation or the agent.

    Skipped when either language's kill switch is off, so a disabled language
    stays on the English-passthrough path like the rest of its turn.
    """
    disabled_langs = _disabled_chat_langs()
    if turn.source_lang.lower() in disabled_langs or turn.target_lang.lower() in disabled_langs:
        return None
    if not is_identity_query(turn.query):
        return None
    response = (
        build_doctor_identity_response(turn.source_lang, turn.target_lang, turn.query)
        if turn.persona == "doctor"
        else build_identity_profile_table(turn.source_lang, turn.target_lang, turn.query)
    )
    logger.info("request_id=%s identity_short_circuit=True", turn.session_id)
    return ClassifierResult(
        canned_text=response,
        label="identity",
        # One of only two chat exit paths that persist history (the other is
        # normal completion) — see docs/channel-seam-design.md.
        history_pair=(
            ModelRequest(parts=[UserPromptPart(content=turn.query)]),
            ModelResponse(parts=[TextPart(content=response)]),
        ),
    )


async def _chat_agent_input(
    turn: Turn,
    pretranslated: Pretranslated,
    *,
    execution,
    scheduler: DeferredScheduler,
    translate_to: Optional[str],
    background: Optional[TurnBackground] = None,
    is_stale: Optional[StalenessCheck] = None,
) -> AgentInput | ClassifierResult:
    """Chat's agent input, for ``SurfaceProfile.agent_input``.

    Loads the farmer context and builds the turn's FarmerContext, then runs
    moderation before the agent starts: a query it rejects is declined, and one
    it cannot check gets the fail-closed line. A valid farmer query also
    schedules its suggestions. Chat has no background set or staleness check.
    """
    session_id = turn.session_id
    target_lang = turn.target_lang
    user_info = turn.authenticated_user
    history = turn.history
    persona = turn.persona
    request_id = session_id
    session_id_safe = (session_id or "")[:200]
    active_agent = doctor_agent if persona == "doctor" else agrinet_agent
    active_moderation_agent = doctor_moderation_agent if persona == "doctor" else moderation_agent
    # Model selection is resolved by the unified pipeline (the only path): the
    # agent + moderation handles, the provider, and the display model name all
    # come from the resolved primary tier for this session's profile. For the
    # current env this is the same provider/base_url/model the removed
    # get_model_for_variant returned, generalized to the weighted split.
    request_model_name = execution.info(_LlmStep.AGENT).model_name
    disabled_langs = _disabled_chat_langs()

    def localize_system_text(text_en: str):
        return _localize_system_text(
            text_en,
            target_lang=target_lang,
            disabled_langs=disabled_langs,
            execution=execution,
            max_output_chars=turn.channel.response_max_chars,
            request_id=request_id,
        )

    farmer_context = await _load_farmer_context(persona, user_info, request_id)
    deps = _build_deps(
        query=pretranslated.query,
        session_id=session_id,
        # Agent responds in English when the answer is translated for the caller.
        lang_code="en" if translate_to else pretranslated.lang,
        persona=persona,
        channel=turn.channel,
        user_info=user_info,
        farmer_context=farmer_context,
    )

    message_pairs = "\n\n".join(format_message_pairs(history, 3))
    logger.info(f"Message pairs: {message_pairs}")
    if message_pairs:
        last_response = f"**Conversation**\n\n{message_pairs}\n\n---\n\n"
    else:
        last_response = ""

    try:
        user_message = f"{last_response}{deps.get_user_message()}"
        _lf_mod = get_langfuse_client() if get_langfuse_client else None
        _mod_obs_ctx = (
            _lf_mod.start_as_current_observation(
                # Distinct from Pydantic's "Moderation Agent run" OTEL span to avoid triple duplicate sidebar labels.
                name="Moderation",
                as_type="generation",
                input={
                    # Actual model the moderation_agent.run uses below
                    # (gemma for OSS, legacy model otherwise) — not LLM_MODEL_NAME,
                    # which mislabeled OSS gemma moderation as gpt in dashboards.
                    "model_name": request_model_name,
                    "query": user_message,
                    "session_id": session_id_safe,
                },
                model=request_model_name,
                metadata={"pipeline": _PIPELINE_NAME},
            )
            if _lf_mod
            else nullcontext()
        )
        with _mod_obs_ctx as mod_obs:
            moderation_run = await execution.run(
                _LlmStep.MODERATION,
                active_moderation_agent,
                user_message,
            )
            moderation_data = moderation_run.output
            logger.info(
                "request_id=%s moderation_category=%s moderation_action=%s",
                request_id,
                moderation_data.category,
                moderation_data.action,
            )
            if mod_obs is not None:
                mod_obs.update(
                    output={
                        "category": moderation_data.category,
                        "action": moderation_data.action,
                    }
                )
            # Generate suggestions after moderation passes
            if moderation_data.category == "valid_agricultural" and persona == "farmer":
                logger.info(f"Triggering suggestions generation for session {session_id}")
                try:
                    suggestions_cache_key = f"suggestions_{session_id}_{target_lang}"
                    status_key = f"{suggestions_cache_key}:pending"
                    # Mark pending and clear stale suggestions so callers wait for fresh output.
                    await set_cache(status_key, True, ttl=SUGGESTIONS_PENDING_TTL)
                    await cache.delete(suggestions_cache_key)
                    # Deferred, never inline: suggestions read the history
                    # this turn has not written yet.
                    scheduler.schedule(
                        create_suggestions, session_id, target_lang, execution
                    )
                    logger.info("Successfully added suggestions task")
                except Exception as e:
                    logger.error(f"Error adding suggestions task: {str(e)}")
            elif moderation_data.category != "valid_agricultural":
                # Hard gate: do not run retrieval/answer agent for moderated non-agricultural requests.
                decline_text = (moderation_data.action or "").strip() or (
                    "I can only answer agriculture and livestock related questions."
                )
                decline_text = await localize_system_text(decline_text)
                logger.info(
                    "request_id=%s moderation_blocked=True response_preview=%s",
                    request_id,
                    decline_text[:160],
                )
                # The decline IS the turn's answer, and run_turn records it as the
                # trace output. Without that the chat export records the turn as
                # a blank answer (~470 rows on 2026-08-06).
                # Moderation ran and decided: the turn ended the way it was
                # supposed to, a success. Recording "error" here inflated the
                # error rate by one row per moderated query.
                return ClassifierResult(canned_text=decline_text, label="moderation decline")
            deps.update_moderation_str(str(moderation_data))
    except Exception as e:
        logger.error("request_id=%s moderation_error=%s", request_id, str(e))
        fail_closed_message = await localize_system_text(GENERIC_UNAVAILABLE_MESSAGE_EN)
        logger.info(
            "request_id=%s moderation_blocked=True reason=moderation_error response_preview=%s",
            request_id,
            fail_closed_message[:160],
        )
        # Deliberately NOT "success": moderation itself failed, the farmer
        # got a placeholder instead of an answer, and that belongs in the
        # error rate. The trace output is still recorded so the export shows
        # what the farmer actually saw rather than a blank row.
        return ClassifierResult(canned_text=fail_closed_message, label="fail-closed", outcome="error")

    if persona == "farmer":
        deps.soil_health_card_context = (
            await get_session_shc_context(session_id, deps.mobile)
        ) or ""
    user_message = deps.get_user_message()
    logger.info(
        "request_id=%s running_agent=True user_query=%s private_shc_context=%s",
        request_id,
        deps.query,
        bool(deps.soil_health_card_context),
    )

    # Run the main agent
    # Strip prior-turn tool calls + their search_documents results from the
    # replayed history. The agent re-searches fresh every turn, so the only
    # effect of keeping them was dragging old RAG chunks forward and bloating
    # prefill (the gemma 10k history budget was mostly stale doc text). The
    # current turn's search is unaffected — it runs live inside the agent
    # loop, not via message_history. Suggestions already runs this way.
    trimmed_history = trim_history(
        history,
        max_tokens=execution.capabilities.history_max_tokens,
        include_system_prompts=False,
        include_tool_calls=False
    )

    logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

    agent_observation_name = "Amul Doctor Agent" if persona == "doctor" else "Amul AI Agent"

    def observe():
        _lf_ag = get_langfuse_client() if get_langfuse_client else None
        return (
            _lf_ag.start_as_current_observation(
                # Distinct from Pydantic's "Amul AI Agent run" span; keeps gen_ai/tool children grouped under that name.
                name=agent_observation_name,
                as_type="generation",
                input={
                    "action": moderation_data.action,
                    "model_name": request_model_name,
                    "persona": persona,
                },
                model=request_model_name,
                metadata={
                    "pipeline": _PIPELINE_NAME,
                    "pipeline_profile": execution.profile_name,
                    "persona": persona,
                },
            )
            if _lf_ag
            else nullcontext()
        )

    return AgentInput(
        agent=active_agent,
        prompt=user_message,
        message_history=trimmed_history,
        deps=deps,
        history=history,
        observe=observe,
    )


#: The chat surface's population of the seam. Chat is the degenerate case: one
#: classifier where voice has five.
CHAT_SURFACE = SurfaceProfile(
    surface=Surface.CHAT,
    classifiers=(_identity_classifier,),
    sink=_ChatSink,
    telemetry=_ChatTelemetry,
    pretranslation=_chat_pretranslation,
    agent_input=_chat_agent_input,
)


class _BackgroundTasksScheduler:
    """``DeferredScheduler`` over FastAPI's BackgroundTasks.

    Starlette runs them once the response — streaming or not — has finished,
    i.e. after the turn has written its history.
    """

    def __init__(self, background_tasks: BackgroundTasks) -> None:
        self._background_tasks = background_tasks

    def schedule(self, fn, /, *args) -> None:
        self._background_tasks.add_task(fn, *args)


async def stream_chat_messages(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    channel: str,
    user_id: str,
    history: list,
    user_info: dict,
    background_tasks: BackgroundTasks,
    persona: ChatPersona = "farmer",
    history_session_id: str | None = None,
    artifact_sink: list[dict[str, Any]] | None = None,
    emit_artifact_frames: bool = True,
) -> AsyncGenerator[str, None]:
    """The chat transport's adapter over ``run_turn``.

    Composes the turn — request values into a ``Turn``, FastAPI's
    BackgroundTasks behind the deferred-work scheduler — then renders each
    emission onto the chat wire: text as-is; artifacts into ``artifact_sink``
    (the non-streaming JSON body) and, when streaming, as the terminal frame.
    Chat has no side channel and no consumer for the agent-activity signal.
    """
    turn = Turn(
        query=query,
        session_id=session_id,
        source_lang=source_lang,
        target_lang=target_lang,
        user_id=user_id,
        authenticated_user=MappingProxyType(dict(user_info or {})),
        history=tuple(history),
        history_session_id=history_session_id or session_id,
        # The turn's channel profile: what differs between delivery channels,
        # resolved once here rather than re-derived at each use site.
        channel=_profile_for(channel),
        persona=persona,
    )
    scheduler = _BackgroundTasksScheduler(background_tasks)
    # aclosing: when the caller closes THIS generator (a hang-up), the turn is
    # closed with it, synchronously, so it records "cancelled" and unwinds its
    # root span now rather than whenever the event loop finalises it. The
    # response layer guarantees that close; see app/routers/chat.py.
    async with aclosing(run_turn(turn, CHAT_SURFACE, scheduler=scheduler)) as emissions:
        async for emission in emissions:
            try:
                chunks = _render_chat_emission(emission, artifact_sink, emit_artifact_frames)
            except Exception as exc:
                # Rendering failed: the turn failed, the caller did not leave.
                # Raise it inside run_turn so its outcome guard records "error";
                # letting aclosing close the turn would inject GeneratorExit and
                # record the failure as "cancelled".
                await emissions.athrow(exc)
                raise
            for chunk in chunks:
                yield chunk


def _render_chat_emission(
    emission: Emission,
    artifact_sink: list[dict[str, Any]] | None,
    emit_artifact_frames: bool,
) -> tuple[str, ...]:
    """What one emission becomes on the chat wire: zero or one chunk."""
    if isinstance(emission, TextEmission):
        return (emission.text,)
    if isinstance(emission, ArtifactEmission):
        if artifact_sink is not None:
            artifact_sink.extend(emission.artifacts)
        if emit_artifact_frames and emission.artifacts:
            return (encode_chat_artifacts(emission.artifacts),)
        return ()
    if isinstance(emission, (AgentActivityEmission, SideChannelEmission)):
        # Chat has no side channel and no consumer for the commit signal.
        return ()
    # A new emission kind must be handled here, not silently dropped: dropping
    # unknown output is exactly how telephony liveness would have died in the merge.
    raise TypeError(f"chat adapter cannot render emission {emission!r}")


async def run_turn(
    turn: Turn,
    surface: SurfaceProfile,
    *,
    scheduler: DeferredScheduler,
    side_channel: Optional[SideChannelSender] = None,
    is_stale: Optional[StalenessCheck] = None,
) -> AsyncGenerator[Emission, None]:
    """One turn, transport-free.

    Opens and closes exactly one root span, records how the turn ended on every
    exit path, runs the surface's pre-turn classifiers before anything is
    spawned, and yields ``Emission`` values for the transport adapter to render.
    Deferred work goes through ``scheduler``; private documents leave only as an
    ``ArtifactEmission``.

    ``side_channel`` and ``is_stale`` are wired where the turn is composed, like
    the scheduler. A surface's liveness speaks through the first. The second is
    asked before the turn commits anything (a classifier's answer, the gate's,
    the history write), and a stale turn stops there; the sink is given it too,
    to stop mid-answer. Chat passes neither.
    """
    started_at = time.monotonic()
    session_id = turn.session_id
    target_lang = turn.target_lang
    user_info = turn.authenticated_user
    history = turn.history
    message_history_session_id = turn.history_session_id

    execution = await llm_core.context(session_id)
    pipeline_profile = execution.profile_name
    # Open the per-turn pipeline-config tracer and hold the EXPLICIT instance.
    # Populate the static fields directly and pass the trace state explicitly
    # across Starlette's StreamingResponse async-generator boundary.
    try:
        pt = execution.begin_trace()
    except Exception as _pt_exc:  # pragma: no cover - tracing must never break the turn
        logger.debug("pipeline_config populate skipped: %s", _pt_exc)
        pt = _pipeline_trace.begin(pipeline_profile)
    # The turn's one root span, its stamps and how the turn ended come from the
    # surface: chat's is chat.turn.v1, voice's will be voice.turn.v1.
    telemetry = (
        surface.telemetry(turn, pipeline_profile=pipeline_profile, pipeline_trace=pt)
        if surface.telemetry is not None
        else _NO_TELEMETRY
    )

    with telemetry.root():
        # ONE exit point for the turn. Without this, an outcome is recorded only
        # on normal completion: a client disconnect or an exception leaves the
        # trace with no output and no signal at all, which is #179's B2. It lives
        # in run_turn, not the adapter, so every surface's turn carries it.
        _turn_outcome = "error"
        background = None
        liveness = None
        try:
            # Resolve per-language kill switches before any response path. This
            # keeps deterministic short-circuits and tool language selection in
            # the same English-passthrough mode as the translation pipeline.
            disabled_langs = _disabled_chat_langs()
            request_id = session_id

            logger.info("request_id=%s user_info=%s", request_id, dict(user_info))

            # The pre-turn classifier chain: runs before anything is spawned, so a
            # match has nothing to cancel. First match answers the turn.
            for classify in surface.classifiers:
                short_circuit = await classify(turn)
                if short_circuit is None:
                    continue
                stale = await _stale_outcome(is_stale, f"before_{short_circuit.label}_response")
                if stale is not None:
                    _turn_outcome = stale
                    return
                telemetry.record_output(short_circuit.canned_text, short_circuit.label)
                await _write_short_circuit_history(
                    history, short_circuit, message_history_session_id, request_id
                )
                # A short-circuit that answered the farmer is a completed turn, not
                # an error. `_turn_outcome` defaults to "error" so that an exit we
                # did not anticipate is loud; every exit that DID answer has to say
                # so on its way out.
                _turn_outcome = short_circuit.outcome
                yield TextEmission(short_circuit.canned_text, raw=short_circuit.raw)
                if short_circuit.raw_tail:
                    yield TextEmission(short_circuit.raw_tail, raw=True)
                return

            # What the caller hears while the model works starts here, for the
            # same reason, and stops before the first thing they hear.
            if surface.liveness is not None and side_channel is not None:
                liveness = surface.liveness(
                    turn, started_at=started_at, send=side_channel, is_stale=is_stale
                )

            # A surface's background checks start here: after the classifiers, so
            # a match has nothing to cancel, and before everything else, so they
            # run alongside it. Chat has none.
            if surface.background is not None:
                background = surface.background(turn, execution=execution)

            if surface.pretranslation is None:
                raise TypeError(f"surface {surface.surface.value!r} reached pretranslation without one")
            stale = await _stale_outcome(is_stale, "before_query_pretranslation")
            if stale is not None:
                _turn_outcome = stale
                return
            pretranslated = await surface.pretranslation(
                turn, execution=execution, background=background
            )
            if isinstance(pretranslated, ClassifierResult):
                # An answer instead of a query for the agent: said like the gate's.
                stale = await _stale_outcome(is_stale, f"before_{pretranslated.label}_response")
                if stale is not None:
                    _turn_outcome = stale
                    return
                if liveness is not None:
                    await liveness.stop()
                    liveness = None
                telemetry.record_output(pretranslated.canned_text, pretranslated.label)
                await _write_short_circuit_history(
                    history, pretranslated, message_history_session_id, request_id
                )
                _turn_outcome = pretranslated.outcome
                yield TextEmission(pretranslated.canned_text, raw=pretranslated.raw)
                if pretranslated.raw_tail:
                    yield TextEmission(pretranslated.raw_tail, raw=True)
                return
            needs_output_translation = (
                target_lang.lower() in INDIAN_LANGUAGES
                and target_lang.lower() not in disabled_langs
            )
            # Set when the agent answers in English and the sink translates.
            translate_to = target_lang if needs_output_translation else None
            if surface.agent_input is None:
                raise TypeError(f"surface {surface.surface.value!r} reached the agent without an agent input")
            agent_input = await surface.agent_input(
                turn,
                pretranslated,
                execution=execution,
                scheduler=scheduler,
                translate_to=translate_to,
                background=background,
                is_stale=is_stale,
            )
            if isinstance(agent_input, ClassifierResult):
                # An answer instead of the agent's (chat's moderation decline):
                # said like the gate's.
                stale = await _stale_outcome(is_stale, f"before_{agent_input.label}_response")
                if stale is not None:
                    _turn_outcome = stale
                    return
                if liveness is not None:
                    await liveness.stop()
                    liveness = None
                telemetry.record_output(agent_input.canned_text, agent_input.label)
                await _write_short_circuit_history(
                    history, agent_input, message_history_session_id, request_id
                )
                _turn_outcome = agent_input.outcome
                yield TextEmission(agent_input.canned_text, raw=agent_input.raw)
                if agent_input.raw_tail:
                    yield TextEmission(agent_input.raw_tail, raw=True)
                return

            if surface.sink is None:
                raise TypeError(f"surface {surface.surface.value!r} reached the agent without a sink")
            sink = surface.sink(
                turn,
                execution=execution,
                deps=agent_input.deps,
                translate_to=translate_to,
                is_stale=is_stale,
            )

            with agent_input.observe() as agent_obs:
                # ── ONE agent-streaming path ─────────────────────────────────
                # Collapsed from the three duplicated blocks (fallback / anthropic
                # .iter / openai .run_stream) into a single token stream parameterized
                # by the resolved tier's provider+model, plus a single downstream that
                # sentence-batches + stream-translates (or passes English through). The
                # disconnect-safe first-token-commit primitives are reused verbatim.
                new_messages: list = []
                # Only a surface that sets limits passes them (voice's).
                limits = {} if agent_input.usage_limits is None else {"usage_limits": agent_input.usage_limits}

                english_src = execution.stream(
                    agent_input.agent,
                    agent_input.prompt,
                    message_history=agent_input.message_history,
                    deps=agent_input.deps,
                    new_messages=new_messages,
                    **limits,
                )

                if background is not None:
                    gated, english_src = await _gate_first_chunk(english_src, background)
                    if gated is not None:
                        stale = await _stale_outcome(is_stale, f"before_{gated.label}_response")
                        if stale is not None:
                            _turn_outcome = stale
                            return
                        if liveness is not None:
                            await liveness.stop()
                            liveness = None
                        telemetry.record_output(gated.canned_text, gated.label)
                        await _write_short_circuit_history(
                            history, gated, message_history_session_id, request_id
                        )
                        _turn_outcome = gated.outcome
                        yield TextEmission(gated.canned_text, raw=gated.raw)
                        if gated.raw_tail:
                            yield TextEmission(gated.raw_tail, raw=True)
                        return

                async for _out in sink.stream(english_src):
                    # A blank chunk is not heard, so the caller is still waiting.
                    if liveness is not None and _out.strip():
                        await liveness.stop()
                        liveness = None
                    yield TextEmission(_out)
                logger.info(f"Streaming complete for session {session_id}")

                # Record trace output: what the caller received, as the sink saw it.
                trace_output = sink.final_text()
                if trace_output:
                    telemetry.record_output(trace_output, "final")
                if get_langfuse_client:
                    try:
                        # Match moderation: structured output so Langfuse shows JSON in the observation panel.
                        if agent_obs is not None:
                            agent_obs.update(
                                output={"response": trace_output or ""},
                            )
                        # Which tier ACTUALLY answered. compact_metadata reports the
                        # configured primary, and it is snapshotted before any step
                        # runs, so a health-prune or failure fallback is invisible in
                        # the trace without this. Emitted here because the walker only
                        # knows the answer once the turn is over.
                        served = _pipeline_trace.served_summary(pt)
                        if served:
                            get_langfuse_client().score_current_trace(
                                name="served_tier",
                                value=served,
                                data_type="CATEGORICAL",
                                comment="Tier that actually produced this turn, per step",
                            )
                    except Exception as e:
                        logger.warning("Langfuse: failed to record agent output / served_tier: %s", e)

            # Rich provider documents are deliberately emitted only after the
            # model/translation/trace pipeline is finished. They are not model
            # output and must never enter TTS, prompt history, or trace bodies.
            # They leave the turn only as an emission; how they reach the caller
            # (terminal SSE frame, JSON array, never spoken) is the adapter's call.
            chat_artifacts = agent_input.deps.take_chat_artifacts()
            if chat_artifacts:
                yield ArtifactEmission(artifacts=tuple(chat_artifacts))

            # A last raw line after the answer: voice's hang-up token when the
            # agent said the conversation is closing.
            closing = agent_input.closing_line(new_messages) if agent_input.closing_line else None
            if closing:
                stale = await _stale_outcome(is_stale, "before_goodbye")
                if stale is not None:
                    _turn_outcome = stale
                    return
                yield TextEmission(closing, raw=True)

            # Post-processing happens AFTER streaming is complete
            messages = [
                *agent_input.history,
                *new_messages
            ]

            stale = await _stale_outcome(is_stale, "before_history_write")
            if stale is not None:
                _turn_outcome = stale
                return
            logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
            await update_message_history(message_history_session_id, messages)
            _turn_outcome = "success"
        except GeneratorExit:
            # Client hung up mid-stream. Re-raised so generator teardown is normal.
            _turn_outcome = "cancelled"
            raise
        except BaseException:
            _turn_outcome = "error"
            raise
        finally:
            telemetry.record_outcome(_turn_outcome)
            if liveness is not None:
                await liveness.stop()
            if background is not None:
                await background.close()
