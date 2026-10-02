"""Voice's sink: the agent's English in, what the caller hears out.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``), the
output side of ``stream_voice_message``. English goes out as it streams,
cleaned for the phone. Any other caller language is cut into translation units
and batches (the first one small, so the caller hears something soon), checked
for a leaked model identity, and stream-translated, with the agreed union-ban
line in place of its translation. A request that goes stale stops where voice
stopped it, and an agent that fails is answered with voice's trouble line.

Translation is this repo's ``app/services/translation.py`` in its voice channel.
``render_for_caller`` is voice's way of putting a fixed English line into the
caller's language; the classifiers and the gate take it as ``render``.

The turn's trace gets what voice's recorded here: every chunk spoken, when the
agent's first text and the first translated text came, each batch's translation,
and why the nudge was stopped.
"""
from __future__ import annotations

import re
from contextlib import aclosing
from functools import lru_cache
from typing import AsyncIterator, Optional

import regex

from agents.tools.models.union import UNION_BANNED_MESSAGE, union_banned_message
from app.services.translation import translate_text, translate_text_stream_fast, translation_channel
from app.turn.types import StalenessCheck, Turn
from app.voice.classifiers import _IDENTITY_RESPONSE_EN
from app.voice.liveness import nudge_stopped
from app.voice.trace import current_trace
from helpers.utils import get_logger, normalize_voice_output

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


_IDENTITY_DRIFT_PATTERN = re.compile(
    r"\b(?:OpenAI|ChatGPT|GPT|Claude|Anthropic|large language model|"
    r"I am an AI assistant made by|I am an AI made by|created by OpenAI|"
    r"made by Anthropic)\b",
    re.IGNORECASE,
)


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


TRANSLATION_TROUBLE_MESSAGE = {
    "gu": "માફ કરશો, હાલમાં તમારા સવાલનો જવાબ આપવામાં તકલીફ થઈ રહી છે. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    "en": "I'm having some trouble answering your question right now, please call in some time.",
}


def clean_output_by_language(text: str, lang_code: str | None) -> str:
    """Filter model output based on language.

    - Always allow whitespace.
    - Always allow basic sentence/word punctuation (.,!? and similar) for all languages.
    - For Gujarati (lang_code 'gu'), additionally restrict letters to the Gujarati Unicode block
      U+0A80..U+0AFF; everything else (Latin letters, other scripts) is stripped.
    """
    if not text:
        return text

    text = normalize_voice_output(text, lang_code)

    lang = (lang_code or "").strip().lower()
    # Basic punctuation to always allow
    allowed_punct = set(".!?,;:()[]{}\"'“”‘’-–—…")

    def _allowed(ch: str) -> bool:
        if ch.isspace():
            return True
        if ch in allowed_punct:
            return True
        code = ord(ch)
        if lang == "gu":
            # Gujarati block U+0A80..U+0AFF (includes letters, digits, signs)
            return 0x0A80 <= code <= 0x0AFF
        # For non-Gujarati, don't restrict characters beyond punctuation/whitespace
        return True

    return "".join(ch for ch in text if _allowed(ch))


def _prepare_voice_output(text: str, lang_code: str) -> str:
    """Normalize model output for voice delivery."""
    return clean_output_by_language(text, lang_code)


def _canned_union_ban_translation(text_en: str, target_lang: str) -> str | None:
    """Pinned union-ban copy when the English batch is that line."""
    if (text_en or "").strip() != UNION_BANNED_MESSAGE:
        return None
    return union_banned_message(target_lang)


def _trouble_message(target_lang: str) -> str:
    return TRANSLATION_TROUBLE_MESSAGE.get(target_lang, TRANSLATION_TROUBLE_MESSAGE["en"])


async def render_for_caller(text_en: str, target_lang: str) -> str:
    """Render English loop text for the caller's language outside the agent loop."""
    normalized_target = (target_lang or "en").strip().lower()
    if normalized_target in {"en", "english"}:
        return _prepare_voice_output(text_en, "en")

    canned_ban = _canned_union_ban_translation(text_en, normalized_target)
    if canned_ban is not None:
        return _prepare_voice_output(canned_ban, normalized_target)

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
        return _trouble_message(normalized_target)


async def _voice_translation(text: str, target_lang: str, execution) -> AsyncIterator[str]:
    """``translate_text_stream_fast`` in the voice channel.

    The channel is set around each step of the translation rather than across
    this generator's yields, so it never leaks into the code consuming them.
    """
    stream = translate_text_stream_fast(
        text=text,
        source_lang="english",
        target_lang=target_lang,
        execution=execution,
    )
    async with aclosing(stream):
        while True:
            with translation_channel("voice"):
                try:
                    chunk = await stream.__anext__()
                except StopAsyncIteration:
                    return
            yield chunk


# Why the nudge was stopped when the first translated chunk came, as voice names
# it, by the batch the chunk came from.
_NUDGE_STOPPED_BY = {
    "before_translated_yield": "first_translated_chunk_received",
    "before_final_translated_yield": "final_translated_batch",
    "before_tail_translated_yield": "tail_translated_fragment",
}


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


class VoiceSink:
    """Voice's sink for one call turn, for ``SurfaceProfile.sink``.

    ``translate_to`` is set when the caller's language needs translation;
    otherwise the English is spoken as it comes. ``is_stale`` is asked where
    voice asked it, and a stale request stops without saying more.
    """

    def __init__(
        self,
        turn: Turn,
        *,
        execution,
        deps,
        translate_to: Optional[str],
        is_stale: Optional[StalenessCheck] = None,
    ) -> None:
        self._session_id = turn.session_id
        self._process_id = turn.call.process_id if turn.call is not None else None
        self._target_lang = (turn.target_lang or "gu").strip().lower()
        self._translating = translate_to is not None
        self._execution = execution
        self._is_stale = is_stale
        self._agent_started = False
        self._first_text_chunk_received = False
        self._last_emitted_sig_char: str | None = None
        self._spoken: list[str] = []

    def stream(self, english: AsyncIterator[str]) -> AsyncIterator[str]:
        return self._stream(english)

    def final_text(self) -> Optional[str]:
        return "".join(self._spoken) or None

    async def _stale(self, reason: str) -> bool:
        return self._is_stale is not None and await self._is_stale(reason) is not None

    def _emit(self, text: str) -> str:
        current_trace().record_emit(text)
        # What the trace keeps as the answer: every chunk the caller hears.
        if text.strip():
            self._spoken.append(text)
        return text

    def _prepare_translated_emit(self, text: str) -> str:
        if not text:
            return text

        first_sig = _first_sig_char(text)
        if (
            self._last_emitted_sig_char in {".", "!", "?", "।"}
            and first_sig is not None
            and re.match(r"[A-Za-z઀-૿]", first_sig)
            and not text[0].isspace()
        ):
            text = " " + text

        last_sig = _last_sig_char(text)
        if last_sig is not None:
            self._last_emitted_sig_char = last_sig
        return text

    async def _stream(self, english: AsyncIterator[str]) -> AsyncIterator[str]:
        try:
            async with aclosing(self._speak(english)) as spoken:
                async for text in spoken:
                    yield text
        except Exception as error:
            logger.error(
                "Voice agent stream failed %s first token; session_id=%s process_id=%s error=%s",
                "after" if self._agent_started else "before",
                self._session_id,
                self._process_id,
                error,
            )
            if not await self._stale("after_stream_error"):
                yield self._emit(_trouble_message(self._target_lang))
        finally:
            nudge_stopped("stream_ended")

    async def _speak(self, english: AsyncIterator[str]) -> AsyncIterator[str]:
        sentence_buffer = ""
        translation_batch: list[str] = []
        batch_word_count = 0

        async with aclosing(english):
            async for chunk in english:
                self._agent_started = True
                if await self._stale("during_agent_stream"):
                    break
                if chunk and chunk.strip():
                    current_trace().mark("first_agent_text_ms")

                if not self._translating:
                    if not self._first_text_chunk_received and chunk and chunk.strip():
                        self._first_text_chunk_received = True
                        current_trace().set_nudge(cancel_reason="first_text_chunk_received")
                    cleaned_chunk = _prepare_voice_output(chunk, self._target_lang) if chunk else chunk
                    if await self._stale("before_direct_yield"):
                        break
                    yield self._emit(cleaned_chunk)
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

                            if should_translate_batch(
                                batch_text, batch_word_count, is_first_batch=not self._first_text_chunk_received
                            ):
                                async for translated_chunk in self._translated(batch_text, "before_translated_yield"):
                                    yield translated_chunk
                                translation_batch = []
                                batch_word_count = 0

                    sentence_buffer = remaining

        if self._translating and not await self._stale("before_translation_flush"):
            if translation_batch:
                async for translated_chunk in self._translated(
                    "".join(translation_batch), "before_final_translated_yield"
                ):
                    yield translated_chunk
            if sentence_buffer.strip():
                async for translated_chunk in self._translated(sentence_buffer, "before_tail_translated_yield"):
                    yield translated_chunk

    async def _translated(self, text: str, before_yield: str) -> AsyncIterator[str]:
        """One batch, translated and ready to be spoken."""
        async with aclosing(self._yield_translated_text(text)) as translated:
            async for translated_chunk in translated:
                if translated_chunk and translated_chunk.strip():
                    if not self._first_text_chunk_received:
                        current_trace().set_nudge(cancel_reason=_NUDGE_STOPPED_BY[before_yield])
                    self._first_text_chunk_received = True
                if await self._stale(before_yield):
                    break
                yield self._emit(self._prepare_translated_emit(translated_chunk))

    async def _yield_translated_text(self, text_to_translate: str) -> AsyncIterator[str]:
        if not text_to_translate:
            return
        text_to_translate = _guard_identity_drift(text_to_translate)
        canned_ban = _canned_union_ban_translation(text_to_translate, self._target_lang)
        if canned_ban is not None:
            yield _prepare_voice_output(canned_ban, self._target_lang)
            return
        trace = current_trace()
        try:
            with trace.stage(
                "output_translation",
                as_type="generation",
                input={"chars": len(text_to_translate)},
                metadata={"target_lang": self._target_lang},
            ):
                async with aclosing(
                    _voice_translation(text_to_translate, self._target_lang, self._execution)
                ) as translated:
                    async for chunk in translated:
                        if await self._stale("during_output_translation"):
                            return
                        cleaned = _prepare_voice_output(chunk, self._target_lang) if chunk else chunk
                        if cleaned and cleaned.strip():
                            trace.mark("first_translation_chunk_ms")
                        yield cleaned
        except Exception as e:
            trace.increment("output_translation_errors")
            logger.error(
                "Translation pipeline output translation failed for session_id=%s error=%s",
                self._session_id,
                e,
            )
            yield _trouble_message(self._target_lang)
