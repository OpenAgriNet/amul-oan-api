"""STRUCTURE 4 — the sink: how a surface turns the agent's English token stream
into what the caller actually receives.

The two surfaces need **different objects here, not two settings of one**:

* Chat accumulates and caps at the channel's ``response_max_chars``, optionally
  stream-translating English into the target language in sentence batches.
* Voice is a streaming batcher (``BATCH_CHAR_LIMIT`` 600 / ``SOFT_SPLIT_MIN``
  180) carrying five pieces of cross-chunk state, where the first-text-chunk
  flag doubles as the liveness-nudge cancel latch and the ``is_first_batch``
  argument, plus an 80-character lookback buffer for protected proper nouns.

Modelling that as flags on one object was the rejected design. The sink is a
protocol with a per-surface implementation instead.

The sentence-segmentation and batch-flush helpers live here rather than in
``app/services/chat`` because they exist solely to serve this structure. They
had no importers outside that module.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import AsyncGenerator, AsyncIterable, Awaitable, Callable, Optional

import regex

from helpers.utils import get_logger

logger = get_logger(__name__)


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


def batch_starts_new_line_or_list(text: str) -> bool:
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


class ChatSink:
    """Chat's sink: accumulate, optionally stream-translate, cap at the channel limit.

    ``translate_stream`` is INJECTED rather than imported. The orchestrator passes
    its own module-level reference, which is what test doubles replace; importing
    it here would bind a different object and silently bypass those doubles.
    """

    def __init__(
        self,
        *,
        needs_output_translation: bool,
        target_lang: str,
        response_max_chars: Optional[int],
        translate_stream: Callable[..., AsyncIterable[str]],
    ) -> None:
        self._needs_output_translation = needs_output_translation
        self._target_lang = target_lang
        self._response_max_chars = response_max_chars
        self._translate_stream = translate_stream
        #: Cross-chunk state. Chat carries two lists; voice's sink carries five
        #: pieces plus a lookback buffer -- hence separate objects.
        self.translated_chunks: list[str] = []
        self.raw_chunks: list[str] = []

    async def _emit_translated(self, batch_text: str) -> AsyncGenerator[str, None]:
        """Translate one batch, falling back to emitting the English batch."""
        if self.translated_chunks and batch_starts_new_line_or_list(batch_text):
            self.translated_chunks.append("\n")
            yield "\n"
        try:
            async for translated_chunk in self._translate_stream(
                text=batch_text,
                source_lang="english",
                target_lang=self._target_lang,
                max_output_chars=self._response_max_chars,
            ):
                self.translated_chunks.append(translated_chunk)
                yield translated_chunk
        except Exception as e:
            logger.error(f"Batch translation failed, falling back to English batch: {e}")
            self.translated_chunks.append(batch_text)
            yield batch_text

    async def emit(self, english_src: AsyncIterable[str]) -> AsyncGenerator[str, None]:
        """Consume the agent's English stream, yield what the caller receives."""
        if not self._needs_output_translation:
            async for chunk in english_src:
                self.raw_chunks.append(chunk)
                yield chunk
            return

        sentence_buffer = ""
        translation_batch: list[str] = []
        batch_word_count = 0
        async for chunk in english_src:
            sentence_buffer += chunk
            complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
            if complete_sentences:
                for sentence in complete_sentences:
                    translation_batch.append(sentence)
                    batch_word_count += len(sentence.split())
                batch_text = "".join(translation_batch)
                if should_translate_batch(batch_text, batch_word_count):
                    async for out in self._emit_translated(batch_text):
                        yield out
                    translation_batch = []
                    batch_word_count = 0
                sentence_buffer = remaining
        # Whatever did not meet the flush threshold, then the tail fragment.
        if translation_batch:
            async for out in self._emit_translated("".join(translation_batch)):
                yield out
        if sentence_buffer.strip():
            async for out in self._emit_translated(sentence_buffer):
                yield out

    def final_text(self) -> Optional[str]:
        """The turn's complete output, for the trace. None if nothing was produced."""
        if self._needs_output_translation and self.translated_chunks:
            return "".join(self.translated_chunks)
        if self.raw_chunks:
            return "".join(self.raw_chunks)
        return None
