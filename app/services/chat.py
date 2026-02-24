from contextlib import nullcontext
from typing import AsyncGenerator
from functools import lru_cache
import regex
import re
from fastapi import BackgroundTasks
from agents.agrinet import agrinet_agent
from agents.moderation import moderation_agent
from agents.models import LLM_PROVIDER
from helpers.utils import get_logger
from app.utils import (
    update_message_history, 
    trim_history, 
    format_message_pairs
)
from helpers.telemetry import create_moderation_event, TelemetryRequest
from app.tasks.telemetry import send_telemetry
from app.tasks.suggestions import create_suggestions
from agents.deps import FarmerContext
from agents.tools.farmer import get_farmer_data_by_mobile
from app.services.translation import (
    translate_text,
    translate_text_stream_fast,
    INDIAN_LANGUAGES,
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


def should_translate_batch(batch_text: str, word_count: int) -> bool:
    MIN_WORDS = 60
    MAX_WORDS = 120

    if word_count < MIN_WORDS:
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


logger = get_logger(__name__)

try:
    from langfuse import propagate_attributes
except ImportError:
    propagate_attributes = None

async def stream_chat_messages(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    user_info: dict,
    background_tasks: BackgroundTasks,
    use_translation_pipeline: bool = False,
) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    # Langfuse: propagate session_id, metadata, and tags for dashboard filtering (max 200 chars per value)
    session_id_safe = (session_id or "")[:200]
    pipeline_name = "translation" if use_translation_pipeline else "default"
    langfuse_metadata = {
        "pipeline": pipeline_name,
        "source_lang": (source_lang or "unknown").lower()[:200],
        "target_lang": (target_lang or "unknown").lower()[:200],
        "user_id": (user_id or "anonymous")[:200],
    }
    langfuse_tags = [f"pipeline:{pipeline_name}"]
    session_ctx = (
        propagate_attributes(
            session_id=session_id_safe,
            metadata=langfuse_metadata,
            tags=langfuse_tags,
        )
        if propagate_attributes
        else nullcontext()
    )

    with session_ctx:
        # Generate a unique content ID for this query
        content_id = f"query_{session_id}_{len(history)//2 + 1}"
        logger.info(f"User info: {user_info}")

        # Extract farmer context: prefer JWT 'data' field; if JWT has 'phone' and no data, fetch by phone
        farmer_data = user_info.get('data') if user_info else None
        if not farmer_data and user_info and user_info.get('phone'):
            try:
                farmer_records = await get_farmer_data_by_mobile(user_info['phone'])
                if farmer_records:
                    farmer_data = {"farmer_records": farmer_records}
                    logger.info(f"Injected farmer context from phone for {len(farmer_records)} record(s)")
            except Exception as e:
                logger.warning(f"Could not fetch farmer data by phone: {e}")

        # Translation pipeline: send query as-is to agent; post-translate response if target is Indian
        # (Pre-translation commented out so e.g. Gujarati question goes directly to the agrinet agent.)
        processing_query = query
        processing_lang = target_lang
        needs_output_translation = use_translation_pipeline and target_lang.lower() in INDIAN_LANGUAGES

        # if use_translation_pipeline and source_lang.lower() in INDIAN_LANGUAGES:
        #     logger.info(f"Translation pipeline: pre-translating {source_lang} query to English (Gemma)")
        #     try:
        #         processing_query = await translate_text(
        #             text=query,
        #             source_lang=source_lang,
        #             target_lang="english",
        #         )
        #         processing_lang = "en"
        #         logger.info(f"Query translated: {query[:50]}... -> {processing_query[:50]}...")
        #     except Exception as e:
        #         logger.error(f"Pre-translation failed, using original query: {e}")
        #         processing_query = query
        #         processing_lang = target_lang
        if use_translation_pipeline and needs_output_translation:
            # Agent responds in English; response will be translated to target_lang downstream
            processing_lang = "en"

        deps = FarmerContext(
            query=processing_query,
            lang_code=processing_lang,
            farmer_info=farmer_data if farmer_data else None,
            use_translation_pipeline=use_translation_pipeline,
        )

        message_pairs = "\n\n".join(format_message_pairs(history, 3))
        logger.info(f"Message pairs: {message_pairs}")
        if message_pairs:
            last_response = f"**Conversation**\n\n{message_pairs}\n\n---\n\n"
        else:
            last_response = ""

        try:
            user_message    = f"{last_response}{deps.get_user_message()}"
            moderation_run  = await moderation_agent.run(user_message)
            moderation_data = moderation_run.output
            logger.info(f"Moderation data: {moderation_data}")

            # # Create the moderation event
            # moderation_event = create_moderation_event(...)
            # Generate suggestions after moderation passes
            if moderation_data.category == "valid_agricultural":
                logger.info(f"Triggering suggestions generation for session {session_id}")
                try:
                    background_tasks.add_task(create_suggestions, session_id, target_lang)
                    logger.info("Successfully added suggestions task")
                except Exception as e:
                    logger.error(f"Error adding suggestions task: {str(e)}")
            deps.update_moderation_str(str(moderation_data))
        except Exception as e:
            logger.error(f"Error in moderation: {str(e)}")

        user_message = deps.get_user_message()
        logger.info(f"Running agent with user message: {user_message}")

        # Run the main agent
        trimmed_history = trim_history(
            history,
            max_tokens=80_000,
            include_system_prompts=True,
            include_tool_calls=True
        )

        logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

        english_response = ""
        did_stream_translated = False

        def _collect_or_yield(chunk: str):
            """Collect chunk if output translation needed, else yield immediately.

            This helper is used for providers where we still do full-response translation.
            """
            nonlocal english_response
            if needs_output_translation:
                english_response += chunk
            else:
                return chunk
            return None

        if LLM_PROVIDER == 'anthropic':
            # For Anthropic: Use agent.iter() + node.stream() instead of run_stream()
            async with agrinet_agent.iter(
                user_prompt=user_message,
                message_history=trimmed_history,
                deps=deps,
            ) as agent_run:
                if needs_output_translation:
                    # Optimised batched streaming for Anthropic as well
                    sentence_buffer = ""
                    translation_batch = []
                    batch_word_count = 0
                    full_english = ""

                    async for node in agent_run:
                        node_type = type(node).__name__

                        if node_type == 'ModelRequestNode':
                            async with node.stream(agent_run.ctx) as request_stream:
                                async for event in request_stream:
                                    event_type = type(event).__name__

                                    text = None
                                    if event_type == 'PartStartEvent' and hasattr(event, 'part'):
                                        part_type = type(event.part).__name__
                                        if part_type == 'TextPart' and hasattr(event.part, 'content'):
                                            text = event.part.content
                                    elif event_type == 'PartDeltaEvent' and hasattr(event, 'delta'):
                                        delta_type = type(event.delta).__name__
                                        if delta_type == 'TextPartDelta':
                                            text = event.delta.content_delta

                                    if text:
                                        sentence_buffer += text
                                        full_english += text

                                        complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
                                        if complete_sentences:
                                            for sentence in complete_sentences:
                                                translation_batch.append(sentence)
                                                batch_word_count += len(sentence.split())

                                            batch_text = "".join(translation_batch)
                                            if should_translate_batch(batch_text, batch_word_count):
                                                try:
                                                    logger.info(
                                                        f"Translation pipeline (Anthropic): "
                                                        f"streaming optimised batch to {target_lang} "
                                                        f"({batch_word_count} words)"
                                                    )
                                                    async for translated_chunk in translate_text_stream_fast(
                                                        text=batch_text,
                                                        source_lang="english",
                                                        target_lang=target_lang,
                                                    ):
                                                        yield translated_chunk
                                                except Exception as e:
                                                    logger.error(
                                                        "Optimised batch translation (Anthropic) failed, "
                                                        f"falling back to English batch: {e}"
                                                    )
                                                    yield batch_text

                                                translation_batch = []
                                                batch_word_count = 0

                                            sentence_buffer = remaining

                    # Flush remaining batches/fragments at end of stream
                    if translation_batch:
                        batch_text = "".join(translation_batch)
                        try:
                            logger.info(
                                f"Translation pipeline (Anthropic): flushing final batch to {target_lang} "
                                f"({batch_word_count} words)"
                            )
                            async for translated_chunk in translate_text_stream_fast(
                                text=batch_text,
                                source_lang="english",
                                target_lang=target_lang,
                            ):
                                yield translated_chunk
                        except Exception as e:
                            logger.error(
                                "Final batch translation (Anthropic) failed, "
                                f"falling back to English batch: {e}"
                            )
                            yield batch_text

                    if sentence_buffer.strip():
                        try:
                            logger.info(
                                "Translation pipeline (Anthropic): flushing tail fragment "
                                f"to {target_lang}"
                            )
                            async for translated_chunk in translate_text_stream_fast(
                                text=sentence_buffer,
                                source_lang="english",
                                target_lang=target_lang,
                            ):
                                yield translated_chunk
                        except Exception as e:
                            logger.error(
                                "Tail fragment translation (Anthropic) failed, "
                                f"falling back to English fragment: {e}"
                            )
                            yield sentence_buffer

                    did_stream_translated = True
                else:
                    async for node in agent_run:
                        node_type = type(node).__name__

                        if node_type == 'ModelRequestNode':
                            async with node.stream(agent_run.ctx) as request_stream:
                                async for event in request_stream:
                                    event_type = type(event).__name__

                                    text = None
                                    if event_type == 'PartStartEvent' and hasattr(event, 'part'):
                                        part_type = type(event.part).__name__
                                        if part_type == 'TextPart' and hasattr(event.part, 'content'):
                                            text = event.part.content
                                    elif event_type == 'PartDeltaEvent' and hasattr(event, 'delta'):
                                        delta_type = type(event.delta).__name__
                                        if delta_type == 'TextPartDelta':
                                            text = event.delta.content_delta

                                    if text:
                                        out = _collect_or_yield(text)
                                        if out is not None:
                                            yield out

                logger.info(f"Streaming complete for session {session_id}")
                new_messages = agent_run.result.new_messages()
        else:
            # For OpenAI/vLLM: Use standard run_stream()
            async with agrinet_agent.run_stream(
                user_prompt=user_message,
                message_history=trimmed_history,
                deps=deps,
            ) as response_stream:
                if needs_output_translation:
                    # Optimised batched streaming: segment English into sentences and translate in good-sized batches
                    sentence_buffer = ""
                    translation_batch = []
                    batch_word_count = 0
                    full_english = ""

                    async for chunk in response_stream.stream_text(delta=True):
                        sentence_buffer += chunk
                        full_english += chunk

                        complete_sentences, remaining = extract_complete_sentences(sentence_buffer)
                        if complete_sentences:
                            for sentence in complete_sentences:
                                translation_batch.append(sentence)
                                batch_word_count += len(sentence.split())

                            batch_text = "".join(translation_batch)
                            if should_translate_batch(batch_text, batch_word_count):
                                try:
                                    logger.info(
                                        f"Translation pipeline: streaming optimised batch to {target_lang} "
                                        f"({batch_word_count} words)"
                                    )
                                    async for translated_chunk in translate_text_stream_fast(
                                        text=batch_text,
                                        source_lang="english",
                                        target_lang=target_lang,
                                    ):
                                        yield translated_chunk
                                except Exception as e:
                                    logger.error(
                                        f"Optimised batch translation failed, falling back to English batch: {e}"
                                    )
                                    yield batch_text

                                translation_batch = []
                                batch_word_count = 0

                            sentence_buffer = remaining

                    # Flush remaining batches/fragments at end of stream
                    if translation_batch:
                        batch_text = "".join(translation_batch)
                        try:
                            logger.info(
                                f"Translation pipeline: flushing final batch to {target_lang} "
                                f"({batch_word_count} words)"
                            )
                            async for translated_chunk in translate_text_stream_fast(
                                text=batch_text,
                                source_lang="english",
                                target_lang=target_lang,
                            ):
                                yield translated_chunk
                        except Exception as e:
                            logger.error(
                                f"Final batch translation failed, falling back to English batch: {e}"
                            )
                            yield batch_text

                    if sentence_buffer.strip():
                        try:
                            logger.info(
                                f"Translation pipeline: flushing tail fragment to {target_lang}"
                            )
                            async for translated_chunk in translate_text_stream_fast(
                                text=sentence_buffer,
                                source_lang="english",
                                target_lang=target_lang,
                            ):
                                yield translated_chunk
                        except Exception as e:
                            logger.error(
                                f"Tail fragment translation failed, falling back to English fragment: {e}"
                            )
                            yield sentence_buffer

                    did_stream_translated = True
                    logger.info(f"Streaming complete for session {session_id}")
                    new_messages = response_stream.new_messages()
                else:
                    async for chunk in response_stream.stream_text(delta=True):
                        out = _collect_or_yield(chunk)
                        if out is not None:
                            yield out

                    logger.info(f"Streaming complete for session {session_id}")
                    new_messages = response_stream.new_messages()

        # Post-translation: stream English response translated to target_lang
        if needs_output_translation and english_response and not did_stream_translated:
            logger.info(f"Translation pipeline: streaming response translation to {target_lang}")
            try:
                async for translated_chunk in translate_text_stream_fast(
                    text=english_response,
                    source_lang="english",
                    target_lang=target_lang,
                ):
                    yield translated_chunk
            except Exception as e:
                logger.error(f"Post-translation failed, using English response: {e}")
                yield english_response

        # Post-processing happens AFTER streaming is complete
        messages = [
            *history,
            *new_messages
        ]

        logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
        await update_message_history(session_id, messages)