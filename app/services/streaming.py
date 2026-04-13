"""
Shared streaming helpers for chat and voice agent output.

Preserves existing batching, translation, and flush behavior from each caller;
do not change sentence segmentation or should_translate_batch in callers without
reviewing both pipelines.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Optional

from app.services.translation import translate_text_stream_fast
from helpers.utils import get_logger

logger = get_logger(__name__)


async def stream_agent_response(
    agent_stream: AsyncIterator[str],
    *,
    use_translation_pipeline: bool,
    source_lang: str,
    target_lang: str,
    translated_output_chunks: list[str] | None,
    extract_complete_sentences: Callable[[str], tuple[list[str], str]],
    should_translate_batch: Callable[[str, int], bool],
    batch_starts_new_line_or_list: Callable[[str], bool] | None = None,
    format_out: Callable[[str], str] | None = None,
    on_stale: Optional[Callable[[], Awaitable[bool]]] = None,
    on_first_nonempty_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
    needs_output_translation: bool | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent text deltas with optional batched translation.

    * Chat (OpenAI/vLLM path): pass ``on_stale=None``, ``translated_output_chunks`` list,
      ``batch_starts_new_line_or_list`` set, ``format_out`` default (identity), and call
      only when ``needs_output_translation`` is already True.

    * Voice: pass ``on_stale`` and optional ``on_first_nonempty_chunk``,
      ``needs_output_translation`` explicit, ``translated_output_chunks=None``,
      ``batch_starts_new_line_or_list=None``, ``format_out`` set (e.g. clean_output_by_language).

    ``source_lang`` is kept for API parity with routers; it is not used inside the chat
    translation delta loop (same as before extraction).
    """
    _ = source_lang

    if on_stale is not None:
        if needs_output_translation is None:
            raise ValueError("needs_output_translation is required when on_stale is set (voice mode)")
        fmt = format_out if format_out is not None else (lambda t: t)
        async for chunk in _stream_voice_mixed_deltas(
            agent_stream,
            use_translation_pipeline=use_translation_pipeline,
            needs_output_translation=needs_output_translation,
            target_lang=target_lang,
            extract_complete_sentences=extract_complete_sentences,
            should_translate_batch=should_translate_batch,
            format_out=fmt,
            on_stale=on_stale,
            on_first_nonempty_chunk=on_first_nonempty_chunk,
        ):
            yield chunk
        return

    if needs_output_translation is None:
        needs_output_translation = use_translation_pipeline

    # Chat passthrough mode (no post-translation).
    if not use_translation_pipeline or not needs_output_translation:
        fmt = format_out if format_out is not None else (lambda t: t)
        async for chunk in agent_stream:
            yield fmt(chunk)
        return

    if translated_output_chunks is None:
        raise ValueError("translated_output_chunks is required for chat translation streaming")
    if batch_starts_new_line_or_list is None:
        raise ValueError("batch_starts_new_line_or_list is required for chat translation streaming")
    async for chunk in _stream_chat_openai_translation_deltas(
        agent_stream,
        target_lang=target_lang,
        translated_output_chunks=translated_output_chunks,
        extract_complete_sentences=extract_complete_sentences,
        should_translate_batch=should_translate_batch,
        batch_starts_new_line_or_list=batch_starts_new_line_or_list,
    ):
        yield chunk


async def _stream_chat_openai_translation_deltas(
    response_stream_text: AsyncIterator[str],
    *,
    target_lang: str,
    translated_output_chunks: list[str],
    extract_complete_sentences: Callable[[str], tuple[list[str], str]],
    should_translate_batch: Callable[[str, int], bool],
    batch_starts_new_line_or_list: Callable[[str], bool],
) -> AsyncGenerator[str, None]:
    """Batched translation over ``stream_text(delta=True)`` (chat OpenAI/vLLM path)."""
    sentence_buffer = ""
    translation_batch: list[str] = []
    batch_word_count = 0

    async for chunk in response_stream_text:
        sentence_buffer += chunk

        complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
        if complete_sentences:
            for sentence in complete_sentences:
                translation_batch.append(sentence)
                batch_word_count += len(sentence.split())

            batch_text = "".join(translation_batch)
            if should_translate_batch(batch_text, batch_word_count):
                if translated_output_chunks and batch_starts_new_line_or_list(batch_text):
                    translated_output_chunks.append("\n")
                    yield "\n"
                try:
                    logger.info(
                        "Translation pipeline: streaming optimised batch to %s (%s words)",
                        target_lang,
                        batch_word_count,
                    )
                    async for translated_chunk in translate_text_stream_fast(
                        text=batch_text,
                        source_lang="english",
                        target_lang=target_lang,
                    ):
                        translated_output_chunks.append(translated_chunk)
                        yield translated_chunk
                except Exception as e:
                    logger.error(
                        "Optimised batch translation failed, falling back to English batch: %s. "
                        "If this repeats, check TRANSLATEGEMMA_27B_BASE_ENDPOINT(S) and vLLM availability.",
                        e,
                    )
                    translated_output_chunks.append(batch_text)
                    yield batch_text

                translation_batch = []
                batch_word_count = 0

            sentence_buffer = remaining

    if translation_batch:
        batch_text = "".join(translation_batch)
        if translated_output_chunks and batch_starts_new_line_or_list(batch_text):
            translated_output_chunks.append("\n")
            yield "\n"
        try:
            logger.info(
                "Translation pipeline: flushing final batch to %s (%s words)",
                target_lang,
                batch_word_count,
            )
            async for translated_chunk in translate_text_stream_fast(
                text=batch_text,
                source_lang="english",
                target_lang=target_lang,
            ):
                translated_output_chunks.append(translated_chunk)
                yield translated_chunk
        except Exception as e:
            logger.error(
                "Final batch translation failed, falling back to English batch: %s",
                e,
            )
            translated_output_chunks.append(batch_text)
            yield batch_text

    if sentence_buffer.strip():
        if translated_output_chunks and batch_starts_new_line_or_list(sentence_buffer):
            translated_output_chunks.append("\n")
            yield "\n"
        try:
            logger.info(
                "Translation pipeline: flushing tail fragment to %s",
                target_lang,
            )
            async for translated_chunk in translate_text_stream_fast(
                text=sentence_buffer,
                source_lang="english",
                target_lang=target_lang,
            ):
                translated_output_chunks.append(translated_chunk)
                yield translated_chunk
        except Exception as e:
            logger.error(
                "Tail fragment translation failed, falling back to English fragment: %s",
                e,
            )
            translated_output_chunks.append(sentence_buffer)
            yield sentence_buffer


async def _stream_voice_mixed_deltas(
    response_stream_text: AsyncIterator[str],
    *,
    use_translation_pipeline: bool,
    needs_output_translation: bool,
    target_lang: str,
    extract_complete_sentences: Callable[[str], tuple[list[str], str]],
    should_translate_batch: Callable[[str, int], bool],
    format_out: Callable[[str], str],
    on_stale: Callable[[], Awaitable[bool]],
    on_first_nonempty_chunk: Optional[Callable[[str], Awaitable[None]]],
) -> AsyncGenerator[str, None]:
    """Voice ``run_stream`` + ``stream_text(delta=True)`` loop (STT + optional translation)."""
    sentence_buffer = ""
    translation_batch: list[str] = []
    batch_word_count = 0

    async for chunk in response_stream_text:
        if await on_stale():
            break

        if on_first_nonempty_chunk is not None:
            await on_first_nonempty_chunk(chunk)

        if not use_translation_pipeline or not needs_output_translation:
            yield format_out(chunk)
            continue

        sentence_buffer += chunk
        complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
        for sentence in complete_sentences:
            translation_batch.append(sentence)
            batch_word_count += len(sentence.split())
        batch_text = "".join(translation_batch)
        if batch_text and should_translate_batch(batch_text, batch_word_count):
            async for translated_chunk in translate_text_stream_fast(
                batch_text,
                "english",
                target_lang,
            ):
                yield format_out(translated_chunk)
            translation_batch = []
            batch_word_count = 0
        sentence_buffer = remaining

    if use_translation_pipeline and needs_output_translation:
        remaining_text = "".join(translation_batch) + sentence_buffer
        if remaining_text.strip():
            async for translated_chunk in translate_text_stream_fast(
                remaining_text,
                "english",
                target_lang,
            ):
                yield format_out(translated_chunk)
