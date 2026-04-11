import asyncio
import re
import time
from functools import lru_cache
from typing import AsyncGenerator, Literal, Optional

import regex
from fastapi import Request
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.deps import FarmerContext
from agents.farmer_context import get_farmer_full_data_by_mobile
from agents.tools.common import (
    get_random_nudge_message,
    send_nudge_message_raya,
    set_tool_call_nudge_event,
)
from agents.tools.farmer import normalize_phone_to_mobile
from agents.voice import voice_agent
from app.config import settings
from app.redis.feedback import (
    clear_feedback_initiated,
    extract_conversation_events_from_messages,
    get_feedback_state,
    set_feedback_initiated,
    set_feedback_rating_received,
)
from app.redis.locks import (
    SessionRequestOwner,
    is_session_request_owner,
    refresh_session_request_ownership,
    release_session_request_ownership,
)
from app.services.feedback import (
    get_feedback_ack,
    get_feedback_question,
    parse_feedback_with_llm,
    send_feedback,
)
from app.services.stt_signals import detect_stt_signal, generate_stt_signal_response
from app.services.translation import (
    INDIAN_LANGUAGES,
    translate_text,
    translate_text_stream_fast,
    translate_to_english_with_gemma4,
)
from app.utils import (
    clean_message_history_for_openai,
    format_message_pairs,
    trim_history,
    update_message_history,
)
from helpers.utils import clean_output_by_language, get_logger

logger = get_logger(__name__)

OPENAI_PRETRANSLATION_MODEL = "gemma4"


class SentenceSegmenter:
    sep = "ŽžŽžSentenceSeparatorŽžŽž"
    terminals = "!?.。！？"

    def __init__(self):
        self._re = [
            (regex.compile(r"(\P{N})([" + self.terminals + r"])(\p{Z}*)"), r"\1\2\3" + self.sep),
            (regex.compile(r"([" + self.terminals + r"])(\P{N})"), r"\1" + self.sep + r"\2"),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [t for t in line.split(self.sep) if t != ""]


sentence_segmenter = SentenceSegmenter()


def extract_complete_sentences(text: str):
    if not text:
        return [], ""
    sentences = sentence_segmenter(text)
    if len(sentences) <= 1:
        return [], text
    return sentences[:-1], sentences[-1]


def should_translate_batch(batch_text: str, word_count: int) -> bool:
    if word_count < 15:
        return batch_text.rstrip().endswith((".", "!", "?")) and word_count >= 5
    if word_count >= 80:
        return True
    return batch_text.rstrip().endswith((".", "!", "?"))


def _is_bare_greeting(query: str) -> bool:
    cleaned = re.sub(r"[*\s]+", " ", query).strip().lower()
    if not cleaned:
        return False
    cleaned = re.sub(r"[.,!?।]+$", "", cleaned).strip()
    return cleaned in {"hello", "hi", "hey", "હેલો", "હલો", "નમસ્તે", "નમસ્કાર"}


def _is_fragment_query(query: str) -> bool:
    cleaned = re.sub(r"[*\s.,!?।]+", " ", query).strip()
    return len(cleaned) <= 3


def _is_hold_message(query: str) -> bool:
    lower = query.lower()
    return any(
        token in lower
        for token in [
            "હોલ્ડ પર",
            "લાઇન પર રહો",
            "લાઈન પર રહો",
            "put your call on hold",
            "call has been put on hold",
            "call on hold",
            "please stay on the line",
            "please remain on the line",
        ]
    )


_GREETING_RESPONSES = {
    "gu": "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો.",
    "en": "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.",
}
_FRAGMENT_RESPONSES = {
    "gu": "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો.",
    "en": "I could not understand your question. Please ask your question again.",
}


async def stream_voice_message(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    provider: Optional[Literal["RAYA"]] = None,
    process_id: Optional[str] = None,
    user_info: dict | None = None,
    use_translation_pipeline: bool = False,
    owner: Optional[SessionRequestOwner] = None,
    http_request: Optional[Request] = None,
) -> AsyncGenerator[str, None]:
    request_started_at = time.monotonic()
    last_owner_refresh_at = 0.0

    async def _request_is_stale(reason: str) -> bool:
        nonlocal last_owner_refresh_at
        if http_request is not None and await http_request.is_disconnected():
            return True
        now = time.monotonic()
        if owner is not None and (
            last_owner_refresh_at == 0.0
            or now - last_owner_refresh_at >= settings.session_owner_refresh_interval_seconds
        ):
            refreshed = await refresh_session_request_ownership(owner)
            last_owner_refresh_at = now
            if not refreshed:
                return True
        if owner is not None and not await is_session_request_owner(owner):
            return True
        return False

    try:
        if await _request_is_stale("before_feedback_check"):
            return

        feedback_state = await get_feedback_state(session_id)
        if feedback_state.get("initiated") and not feedback_state.get("rating_received"):
            requested_target_lang = (target_lang or "gu").strip().lower()
            trigger = feedback_state.get("trigger") or "conversation_closing"
            parsed = await parse_feedback_with_llm(query, requested_target_lang)
            rating = parsed.get("rating") if isinstance(parsed.get("rating"), int) else None
            valid_rating = parsed.get("is_feedback") is True and rating is not None and 1 <= rating <= 5
            if valid_rating:
                ack = get_feedback_ack(rating, requested_target_lang)
                await send_feedback(
                    session_id=session_id,
                    user_id=user_id,
                    process_id=process_id,
                    rating=rating,
                    trigger=trigger,
                    source_lang=source_lang or "gu",
                    target_lang=requested_target_lang,
                    message_history_summary={"turn_count": len(history)},
                    farmer_info=None,
                    raw_input=None,
                )
                await set_feedback_rating_received(session_id)
                feedback_question = get_feedback_question(requested_target_lang)
                await update_message_history(
                    session_id,
                    [
                        *history,
                        ModelResponse(parts=[TextPart(content=feedback_question)]),
                        ModelRequest(parts=[UserPromptPart(content=query)]),
                        ModelResponse(parts=[TextPart(content=ack)]),
                    ],
                )
                yield ack
                return
            await clear_feedback_initiated(session_id)

        requested_source_lang = (source_lang or "gu").strip().lower()
        requested_target_lang = (target_lang or "gu").strip().lower()
        needs_output_translation = use_translation_pipeline and requested_target_lang in INDIAN_LANGUAGES

        stt_signal = detect_stt_signal(query)
        if stt_signal is not None:
            stt_response = await generate_stt_signal_response(
                signal=stt_signal,
                target_lang=requested_target_lang,
                recent_history_text="\n\n".join(format_message_pairs(history, 3)),
            )
            await update_message_history(
                session_id,
                [*history, ModelRequest(parts=[UserPromptPart(content=query)]), ModelResponse(parts=[TextPart(content=stt_response)])],
            )
            yield stt_response
            return

        if _is_hold_message(query):
            goodbye = "Goodbye."
            await update_message_history(
                session_id,
                [*history, ModelRequest(parts=[UserPromptPart(content=query)]), ModelResponse(parts=[TextPart(content=goodbye)])],
            )
            yield goodbye
            return

        if _is_bare_greeting(query):
            greeting = _GREETING_RESPONSES.get(requested_target_lang, _GREETING_RESPONSES["gu"])
            await update_message_history(
                session_id,
                [*history, ModelRequest(parts=[UserPromptPart(content=query)]), ModelResponse(parts=[TextPart(content=greeting)])],
            )
            yield greeting
            return

        if _is_fragment_query(query):
            frag = _FRAGMENT_RESPONSES.get(requested_target_lang, _FRAGMENT_RESPONSES["gu"])
            await update_message_history(
                session_id,
                [*history, ModelRequest(parts=[UserPromptPart(content=query)]), ModelResponse(parts=[TextPart(content=frag)])],
            )
            yield frag
            return

        processing_query = query
        processing_lang = requested_source_lang
        if use_translation_pipeline and requested_source_lang in {"gu", "gujarati"}:
            try:
                processing_query = await translate_to_english_with_gemma4(query, requested_source_lang)
                processing_lang = "en"
            except Exception:
                processing_query = await translate_text(query, requested_source_lang, "english")
                processing_lang = "en"
        if use_translation_pipeline and needs_output_translation:
            processing_lang = "en"

        tool_call_event = asyncio.Event()
        set_tool_call_nudge_event(tool_call_event)
        nudge_task = asyncio.create_task(asyncio.sleep(max(0.0, settings.nudge_timeout_seconds)))

        mobile = normalize_phone_to_mobile(user_id)
        farmer_info = ""
        if mobile:
            try:
                farmer_info = await get_farmer_full_data_by_mobile(mobile)
            except Exception:
                farmer_info = ""

        deps = FarmerContext(
            query=processing_query,
            lang_code=processing_lang,
            target_lang=requested_target_lang,
            provider=provider,
            session_id=session_id,
            process_id=process_id,
            farmer_info=farmer_info,
            use_translation_pipeline=use_translation_pipeline,
        )

        cleaned_history = clean_message_history_for_openai(history)
        if len(cleaned_history) != len(history):
            await update_message_history(session_id, cleaned_history)
            history = cleaned_history

        trimmed_history = trim_history(history, max_tokens=80_000, include_system_prompts=True, include_tool_calls=True)

        async with voice_agent.run_stream(user_prompt=deps.get_user_message(), message_history=trimmed_history, deps=deps) as response_stream:
            sentence_buffer = ""
            translation_batch: list[str] = []
            batch_word_count = 0
            first_chunk = False

            async for chunk in response_stream.stream_text(delta=True):
                if await _request_is_stale("during_agent_stream"):
                    break

                if not first_chunk and chunk and chunk.strip():
                    first_chunk = True
                    if not nudge_task.done():
                        nudge_task.cancel()

                if not use_translation_pipeline or not needs_output_translation:
                    yield clean_output_by_language(chunk, requested_target_lang)
                    continue

                sentence_buffer += chunk
                complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
                for sentence in complete_sentences:
                    translation_batch.append(sentence)
                    batch_word_count += len(sentence.split())
                batch_text = "".join(translation_batch)
                if batch_text and should_translate_batch(batch_text, batch_word_count):
                    async for translated_chunk in translate_text_stream_fast(batch_text, "english", requested_target_lang):
                        yield clean_output_by_language(translated_chunk, requested_target_lang)
                    translation_batch = []
                    batch_word_count = 0
                sentence_buffer = remaining

            if use_translation_pipeline and needs_output_translation:
                remaining_text = "".join(translation_batch) + sentence_buffer
                if remaining_text.strip():
                    async for translated_chunk in translate_text_stream_fast(remaining_text, "english", requested_target_lang):
                        yield clean_output_by_language(translated_chunk, requested_target_lang)

            new_messages = response_stream.new_messages()

        if not nudge_task.done():
            nudge_task.cancel()
            try:
                await nudge_task
            except asyncio.CancelledError:
                pass
        else:
            if not first_chunk and not await _request_is_stale("before_nudge_send"):
                await send_nudge_message_raya(get_random_nudge_message(requested_target_lang), session_id, process_id)

        if await _request_is_stale("before_history_write"):
            return

        messages = [*history, *new_messages]
        await update_message_history(session_id, messages)

        feedback_state = await get_feedback_state(session_id)
        if not feedback_state.get("initiated"):
            events = extract_conversation_events_from_messages(new_messages)
            trigger = next((e for e in events if e in ("conversation_closing", "user_frustration")), None)
            if trigger:
                await set_feedback_initiated(session_id, trigger)
                feedback_question = get_feedback_question(requested_target_lang)
                yield clean_output_by_language(" " + feedback_question, requested_target_lang)
    finally:
        await release_session_request_ownership(owner)


def stream_voice_messages(*args, **kwargs):
    """Compatibility alias expected by integration layer."""
    return stream_voice_message(*args, **kwargs)

