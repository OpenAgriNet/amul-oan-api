from contextlib import nullcontext
from typing import AsyncGenerator
from functools import lru_cache
import os
import regex
import re
from fastapi import BackgroundTasks
from agents.agrinet import agrinet_agent
from agents.moderation import moderation_agent
from app.llm_core import resolver as _llm_resolver
from app.llm_core.config_model import Step as _LlmStep
from helpers.utils import get_logger
from app.utils import (
    update_message_history,
    trim_history,
    format_message_pairs,
    set_cache,
)
from app.tasks.suggestions import create_suggestions
from app.config import settings
from app.services.fallback import AGENT_ACTIVITY, execute_with_fallback, stream_with_fallback, with_first_token_deadline
from app.core.cache import cache
from agents.deps import FarmerContext
from agents.farmer_context import get_farmer_context_bundle_by_mobile
from agents.tools.farmer import normalize_phone_to_mobile
from app.services.translation import (
    translate_text,
    translate_to_english_pretranslation,
    translate_text_stream_fast,
    INDIAN_LANGUAGES,
    PRETRANSLATION_PROVIDER,
    PRETRANSLATION_MODEL,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from app.services.identity_profile import is_identity_query, build_identity_profile_table


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


def _chat_history_trim_max_tokens(agent_provider: str, agent_model_name: str) -> int:
    """Keep fewer past turns for smaller-context vLLM gemma backends so
    system+tools+history+user fit.

    Driven by the RESOLVED agent tier (provider + model), not a startup singleton:
    a self-hosted vLLM gemma tier — whether the session's primary is the gemma
    profile or the startup default is gemma-on-vLLM — gets the tighter gemma cap
    (tune via CHAT_HISTORY_MAX_TOKENS_VLLM_GEMMA); everything else gets 80k. This
    reproduces the old ``is_oss_gemma or is_startup_vllm_gemma`` decision now that
    the tier is resolved by app/llm_core.
    """
    override = os.getenv("CHAT_HISTORY_MAX_TOKENS")
    if override and override.isdigit():
        return int(override)
    if (agent_provider or "").lower() == "vllm" and "gemma" in (agent_model_name or "").lower():
        cap = os.getenv("CHAT_HISTORY_MAX_TOKENS_VLLM_GEMMA", "10000")
        return int(cap) if cap.isdigit() else 10_000
    return 80_000


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



logger = get_logger(__name__)
WHATSAPP_RESPONSE_MAX_CHARS = 1600
SUGGESTIONS_PENDING_TTL = 30
GENERIC_UNAVAILABLE_MESSAGE_EN = (
    "I am unable to process your request right now. Please try again later."
)
GENERIC_UNAVAILABLE_MESSAGE_GU = (
    "હાલમાં હું તમારી વિનંતી પ્રક્રિયા કરી શકતી નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો."
)

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

# Per-turn resolved-pipeline-config tracer (tracing-only; no behaviour change).
from app.llm_core import trace as _pipeline_trace


def _response_max_chars_for_channel(channel: str | None) -> int | None:
    if (channel or "").lower() == "whatsapp":
        return WHATSAPP_RESPONSE_MAX_CHARS
    return None

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
    use_translation_pipeline: bool = True,
    pipeline_variant: str = "legacy",
) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    # OSS sticky variant => run the dev OSS path (translation pipeline + vLLM
    # agent model). 'legacy' keeps the current prod behaviour byte-for-byte;
    # with the OSS profile at weight 0 every session is 'legacy'.
    is_oss = pipeline_variant == "oss"
    use_translation_pipeline = bool(use_translation_pipeline) or is_oss
    # Open the per-turn pipeline-config tracer and hold the EXPLICIT instance.
    # The ContextVar does NOT survive Starlette's StreamingResponse async-generator
    # consumption, so we populate the must-have static fields (profile, variant,
    # flags, per-step PRIMARY tier) directly onto `pt` here and pass `pt` to every
    # emit site — never relying on a contextvar read at emit time. Deep trigger /
    # served-tier recording stays best-effort on top (via the contextvar).
    pt = _pipeline_trace.begin(pipeline_variant)
    try:
        from app.llm_core import resolver as _lr, runtime as _lrt
        from app.llm_core.config_model import Step as _LS
        _pipeline_trace.populate(
            pt, _lrt.get_pipeline(), _lr.primary_tier, pipeline_variant,
            (_LS.PRE_TRANSLATION, _LS.MODERATION, _LS.AGENT, _LS.SUGGESTIONS, _LS.POST_TRANSLATION),
        )
    except Exception as _pt_exc:  # pragma: no cover - tracing must never break the turn
        logger.debug("pipeline_config populate skipped: %s", _pt_exc)
    # Model selection is resolved by the unified pipeline (the only path): the
    # agent + moderation handles, the provider, and the display model name all
    # come from the resolved primary tier for this session's variant. For the
    # current env this is the same provider/base_url/model the removed
    # get_model_for_variant returned, generalized to the weighted-profile split.
    agent_tier = _llm_resolver.primary_tier(_LlmStep.AGENT, pipeline_variant)
    request_model = agent_tier.handle
    request_provider = agent_tier.provider
    request_model_name = agent_tier.model_name
    moderation_model = _llm_resolver.primary_handle(_LlmStep.MODERATION, pipeline_variant)
    # Langfuse: propagate session_id, metadata, and tags for dashboard filtering (max 200 chars per value)
    session_id_safe = (session_id or "")[:200]
    pipeline_name = "translation" if use_translation_pipeline else "default"
    # Prefer phone from JWT (weburl-minted tokens) over the query-param user_id
    effective_user_id = (
        (user_info.get("phone") or user_info.get("sub")) if user_info else None
    ) or user_id or "anonymous"
    effective_user_id = effective_user_id[:200]
    langfuse_metadata = {
        "pipeline": pipeline_name,
        "channel": (channel or "web")[:200],
        "source_lang": (source_lang or "unknown").lower()[:200],
        "target_lang": (target_lang or "unknown").lower()[:200],
        "user_id": effective_user_id,
        "variant": pipeline_variant,
    }
    langfuse_tags = [f"pipeline:{pipeline_name}", f"variant:{pipeline_variant}"]
    # Serialize the resolved pipeline config into COMPACT flat keys and merge them
    # into the same langfuse_metadata dict propagate_attributes lands on OTEL span
    # attributes (a big nested blob is size-capped/dropped; this SDK has no
    # update_current_trace). Adds `pipeline_profile`, `pipeline_flags`, and one
    # `pc_<step>` per step (~50 chars each). Full static config is in the
    # `llm_core.full_config` boot log. Best-effort — never breaks the turn.
    _pipeline_trace.add_compact_metadata(pt, langfuse_metadata)
    session_ctx = (
        propagate_attributes(
            session_id=session_id_safe,
            user_id=effective_user_id,
            #trace_name=f"chat.{pipeline_name}",
            #the above line causes all the traces to be named chat.translation
            #if the use_translation_pipeline is true, chat.default if false.
            metadata=langfuse_metadata,
            tags=langfuse_tags,
        )
        if propagate_attributes
        else nullcontext()
    )

    with session_ctx:
        if get_langfuse_client:
            try:
                langfuse = get_langfuse_client()
                langfuse.set_current_trace_io(
                    input={
                        "query": query,
                        "channel": channel,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "use_translation_pipeline": use_translation_pipeline,
                    }
                )
                #this is the same as the update_current_trace method,
                #but it is more explicit about the type of the output
                # and is supported by the latest version of the langfuse SDK.
                # Emit a categorical pipeline_variant score attached to the
                # *current trace*. Langfuse rolls this up to the session view,
                # so a Sessions filter "pipeline_variant = oss" works directly.
                # `score_id` is deterministic per session so subsequent traces
                # in the same session upsert the same score (no duplicates).
                try:
                    langfuse.score_current_trace(
                        name="pipeline_variant",
                        value=pipeline_variant,
                        data_type="CATEGORICAL",
                        score_id=f"variant-{session_id_safe}",
                        comment="Sticky pipeline variant for this session",
                    )
                except Exception as e:
                    logger.warning("Langfuse: pipeline_variant score failed: %s", e)
            except Exception as e:
                logger.warning("Langfuse: failed to set trace input: %s", e)

        async def localize_system_text(text_en: str) -> str:
            """
            Localize short system-generated outputs to target language when needed.
            Falls back to Gujarati default text if translation fails for Gujarati targets.
            """
            if not text_en:
                return text_en
            if not target_lang:
                return text_en

            lang = target_lang.lower()
            if lang == "english" or lang == "en":
                return text_en

            if lang in INDIAN_LANGUAGES:
                try:
                    return await translate_text(
                        text=text_en,
                        source_lang="english",
                        target_lang=target_lang,
                        max_output_chars=_response_max_chars_for_channel(channel),
                    )
                except Exception as e:
                    logger.warning(
                        "request_id=%s system text translation failed target_lang=%s error=%s",
                        request_id if 'request_id' in locals() else "unknown",
                        target_lang,
                        e,
                    )
                    if lang in {"gu", "gujarati"}:
                        return GENERIC_UNAVAILABLE_MESSAGE_GU
            return text_en

        request_id = session_id
        # Generate a unique content ID for this query
        content_id = f"query_{session_id}_{len(history)//2 + 1}"
        logger.info("request_id=%s user_info=%s", request_id, user_info)

        if is_identity_query(query):
            identity_response = build_identity_profile_table(
                source_lang=source_lang,
                target_lang=target_lang,
                query=query,
            )
            logger.info("request_id=%s identity_short_circuit=True", request_id)
            if get_langfuse_client:
                try:
                    langfuse = get_langfuse_client()
                    langfuse.set_current_trace_io(output=identity_response)
                except Exception as e:
                    logger.warning("Langfuse: failed to record identity output: %s", e)

            messages = [
                *history,
                ModelRequest(parts=[UserPromptPart(content=query)]),
                ModelResponse(parts=[TextPart(content=identity_response)]),
            ]
            logger.info(
                "request_id=%s updating_history_identity_path=True total_messages=%s",
                request_id,
                len(messages),
            )
            await update_message_history(session_id, messages)
            yield identity_response
            return

        # Extract farmer context from phone in JWT via cache-first fetch
        farmer_data = ""
        farmer_unions: list[str] = []
        if user_info and user_info.get('phone'):
            try:
                farmer_data, farmer_unions = await get_farmer_context_bundle_by_mobile(user_info['phone'])
                logger.info(f"request_id={request_id} farmer_context_length={len(farmer_data)}")
                logger.info("request_id=%s farmer_unions=%s", request_id, farmer_unions)
            except Exception as e:
                logger.warning(f"request_id={request_id} farmer_context_fetch_failed={e}")

        processing_query = query
        processing_lang = target_lang
        needs_output_translation = use_translation_pipeline and target_lang.lower() in INDIAN_LANGUAGES

        if use_translation_pipeline and source_lang.lower() in {"gu", "gujarati"}:
            # OSS sessions force pre-translation onto the self-hosted vLLM endpoint
            # (provider="vllm"); legacy keeps the configured PRETRANSLATION_PROVIDER
            # (None => default). Equivalent to the resolved PRE_TRANSLATION primary
            # tier: an OSS-endpoint tier for an OSS session, the managed tier
            # otherwise.
            pretrans_provider = "vllm" if is_oss else None
            logger.info(
                "request_id=%s translation_pipeline=True variant=%s pretranslating gu->en with %s/%s",
                request_id,
                pipeline_variant,
                pretrans_provider or PRETRANSLATION_PROVIDER,
                request_model_name if is_oss else PRETRANSLATION_MODEL,
            )
            if settings.fallback_enabled:
                # Standard OSS -> managed fallback. Drops the legacy TranslateGemma
                # stopgap (decision #7): TranslateGemma is also self-hosted vLLM, so
                # it shared a failure domain with the OSS pretranslation it backed up.
                try:
                    processing_query = await execute_with_fallback(
                        pipeline="pretranslation",
                        session_id=session_id_safe,
                        variant=pipeline_variant,
                        run=lambda a: translate_to_english_pretranslation(
                            text=query,
                            source_lang=source_lang,
                            provider="vllm" if a.kind == "oss" else None,
                        ),
                    )
                    processing_lang = "en"
                    logger.info(
                        "request_id=%s pretranslation_success=True source_preview=%s translated_preview=%s",
                        request_id,
                        query[:80],
                        processing_query[:80],
                    )
                except Exception as e:
                    logger.error(
                        "request_id=%s pretranslation_success=False (all tiers) source_lang=%s error=%s",
                        request_id,
                        source_lang,
                        e,
                    )
                    processing_query = query
                    processing_lang = target_lang
            else:
                try:
                    processing_query = await translate_to_english_pretranslation(
                        text=query,
                        source_lang=source_lang,
                        provider=pretrans_provider,
                    )
                    processing_lang = "en"
                    logger.info(
                        "request_id=%s pretranslation_success=True source_preview=%s translated_preview=%s",
                        request_id,
                        query[:80],
                        processing_query[:80],
                    )
                except Exception as e:
                    logger.error(
                        "request_id=%s pretranslation_success=False source_lang=%s error=%s",
                        request_id,
                        source_lang,
                        e,
                    )
                    try:
                        logger.info(
                            "request_id=%s pretranslation_fallback=translategemma source_lang=%s",
                            request_id,
                            source_lang,
                        )
                        processing_query = await translate_text(
                            text=query,
                            source_lang=source_lang,
                            target_lang="english",
                        )
                        processing_lang = "en"
                    except Exception as fallback_error:
                        logger.error(
                            "request_id=%s pretranslation_fallback_failed=True source_lang=%s error=%s",
                            request_id,
                            source_lang,
                            fallback_error,
                        )
                        processing_query = query
                        processing_lang = target_lang
        if use_translation_pipeline and needs_output_translation:
            # Agent responds in English; response will be translated to target_lang downstream
            processing_lang = "en"

        # Normalized caller phone — the micro-loan tool reads this from deps so it
        # never has to trust an LLM-supplied number. None for anonymous sessions.
        loan_mobile = normalize_phone_to_mobile(user_info['phone']) if user_info and user_info.get('phone') else None

        deps = FarmerContext(
            query=processing_query,
            session_id=session_id,
            lang_code=processing_lang,
            farmer_info=farmer_data,
            farmer_unions=farmer_unions,
            use_translation_pipeline=use_translation_pipeline,
            response_max_chars=_response_max_chars_for_channel(channel),
            mobile=loan_mobile,
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
                        "use_translation_pipeline": bool(use_translation_pipeline),
                    },
                    model=request_model_name,
                    metadata={"pipeline": pipeline_name},
                )
                if _lf_mod
                else nullcontext()
            )
            with _mod_obs_ctx as mod_obs:
                if settings.fallback_enabled:
                    moderation_run = await execute_with_fallback(
                        pipeline="moderation",
                        session_id=session_id_safe,
                        variant=pipeline_variant,
                        run=lambda a: moderation_agent.run(user_message, model=a.model),
                    )
                else:
                    moderation_run = await moderation_agent.run(user_message, model=moderation_model)
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
                if moderation_data.category == "valid_agricultural":
                    logger.info(f"Triggering suggestions generation for session {session_id}")
                    try:
                        suggestions_cache_key = f"suggestions_{session_id}_{target_lang}"
                        status_key = f"{suggestions_cache_key}:pending"
                        # Mark pending and clear stale suggestions so callers wait for fresh output.
                        await set_cache(status_key, True, ttl=SUGGESTIONS_PENDING_TTL)
                        await cache.delete(suggestions_cache_key)
                        background_tasks.add_task(create_suggestions, session_id, target_lang, pipeline_variant)
                        logger.info("Successfully added suggestions task")
                    except Exception as e:
                        logger.error(f"Error adding suggestions task: {str(e)}")
                else:
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
                    yield decline_text
                    return
                deps.update_moderation_str(str(moderation_data))
        except Exception as e:
            logger.error("request_id=%s moderation_error=%s", request_id, str(e))
            fail_closed_message = await localize_system_text(GENERIC_UNAVAILABLE_MESSAGE_EN)
            logger.info(
                "request_id=%s moderation_blocked=True reason=moderation_error response_preview=%s",
                request_id,
                fail_closed_message[:160],
            )
            yield fail_closed_message
            return

        user_message = deps.get_user_message()
        logger.info("request_id=%s running_agent=True user_message=%s", request_id, user_message)

        # Run the main agent
        # Strip prior-turn tool calls + their search_documents results from the
        # replayed history. The agent re-searches fresh every turn, so the only
        # effect of keeping them was dragging old RAG chunks forward and bloating
        # prefill (the gemma 10k history budget was mostly stale doc text). The
        # current turn's search is unaffected — it runs live inside the agent
        # loop, not via message_history. Suggestions already runs this way.
        trimmed_history = trim_history(
            history,
            max_tokens=_chat_history_trim_max_tokens(request_provider, request_model_name),
            include_system_prompts=False,
            include_tool_calls=False
        )

        logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

        # Buffer streamed output for Langfuse trace output
        translated_output_chunks: list[str] = []
        raw_output_chunks: list[str] = []

        _lf_ag = get_langfuse_client() if get_langfuse_client else None
        _agrinet_obs_ctx = (
            _lf_ag.start_as_current_observation(
                # Distinct from Pydantic's "Amul AI Agent run" span; keeps gen_ai/tool children grouped under that name.
                name="Amul AI Agent",
                as_type="generation",
                input={
                    "action": moderation_data.action,
                    "model_name": request_model_name,
                },
                model=request_model_name,
                metadata={"pipeline": pipeline_name, "variant": pipeline_variant},
            )
            if _lf_ag
            else nullcontext()
        )

        with _agrinet_obs_ctx as agrinet_obs:
            # ── ONE agent-streaming path ─────────────────────────────────
            # Collapsed from the three duplicated blocks (fallback / anthropic
            # .iter / openai .run_stream) into a single token stream parameterized
            # by the resolved tier's provider+model, plus a single downstream that
            # sentence-batches + stream-translates (or passes English through). The
            # disconnect-safe first-token-commit primitives are reused verbatim.
            _stream_holder: dict = {}

            async def _raw_agent_text_stream(provider, model):
                # (D) COMMIT-ON-FIRST-ACTIVITY. Both providers now iterate the agent via
                # agent.iter()+node.stream() (anthropic always required it — run_stream()
                # is unsupported for its tool loop; every other provider joins so the fix
                # is uniform for the OSS gemma tier where the slow 20s milk-collection
                # tool lives). The FIRST pydantic-ai model event — a tool-call part that
                # pydantic-ai emits BEFORE it runs the tools and long before the first
                # TEXT delta — is surfaced once as the AGENT_ACTIVITY sentinel.
                # with_first_token_deadline treats that sentinel as the first-token
                # commit, so a slow tool can no longer trip the TTFT deadline and force a
                # cross-tier re-run of side-effecting tools. The sentinel is swallowed by
                # the deadline wrapper and never reaches the client; TEXT extraction is
                # unchanged. Liveness is preserved: a truly hung endpoint emits no event,
                # so no sentinel arrives and the deadline still fires -> swap.
                # new_messages is captured before the run context closes.
                _activity_signaled = False
                async with agrinet_agent.iter(
                    user_prompt=user_message,
                    message_history=trimmed_history,
                    deps=deps,
                    model=model,
                ) as agent_run:
                    async for node in agent_run:
                        if type(node).__name__ == 'ModelRequestNode':
                            async with node.stream(agent_run.ctx) as request_stream:
                                async for event in request_stream:
                                    if not _activity_signaled:
                                        _activity_signaled = True
                                        yield AGENT_ACTIVITY
                                    event_type = type(event).__name__
                                    text = None
                                    if event_type == 'PartStartEvent' and hasattr(event, 'part'):
                                        if type(event.part).__name__ == 'TextPart' and hasattr(event.part, 'content'):
                                            text = event.part.content
                                    elif event_type == 'PartDeltaEvent' and hasattr(event, 'delta'):
                                        if type(event.delta).__name__ == 'TextPartDelta':
                                            text = event.delta.content_delta
                                    if text:
                                        yield text
                    _stream_holder["new_messages"] = agent_run.result.new_messages()

            async def _stream_to_client(english_src):
                if needs_output_translation:
                    sentence_buffer = ""
                    translation_batch = []
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
                                if translated_output_chunks and _batch_starts_new_line_or_list(batch_text):
                                    translated_output_chunks.append("\n")
                                    yield "\n"
                                try:
                                    async for translated_chunk in translate_text_stream_fast(
                                        text=batch_text,
                                        source_lang="english",
                                        target_lang=target_lang,
                                        max_output_chars=deps.response_max_chars,
                                    ):
                                        translated_output_chunks.append(translated_chunk)
                                        yield translated_chunk
                                except Exception as e:
                                    logger.error(f"Optimised batch translation failed, falling back to English batch: {e}")
                                    translated_output_chunks.append(batch_text)
                                    yield batch_text
                                translation_batch = []
                                batch_word_count = 0
                            sentence_buffer = remaining
                    if translation_batch:
                        batch_text = "".join(translation_batch)
                        if translated_output_chunks and _batch_starts_new_line_or_list(batch_text):
                            translated_output_chunks.append("\n")
                            yield "\n"
                        try:
                            async for translated_chunk in translate_text_stream_fast(
                                text=batch_text,
                                source_lang="english",
                                target_lang=target_lang,
                                max_output_chars=deps.response_max_chars,
                            ):
                                translated_output_chunks.append(translated_chunk)
                                yield translated_chunk
                        except Exception as e:
                            logger.error(f"Final batch translation failed, falling back to English batch: {e}")
                            translated_output_chunks.append(batch_text)
                            yield batch_text
                    if sentence_buffer.strip():
                        if translated_output_chunks and _batch_starts_new_line_or_list(sentence_buffer):
                            translated_output_chunks.append("\n")
                            yield "\n"
                        try:
                            async for translated_chunk in translate_text_stream_fast(
                                text=sentence_buffer,
                                source_lang="english",
                                target_lang=target_lang,
                                max_output_chars=deps.response_max_chars,
                            ):
                                translated_output_chunks.append(translated_chunk)
                                yield translated_chunk
                        except Exception as e:
                            logger.error(f"Tail fragment translation failed, falling back to English fragment: {e}")
                            translated_output_chunks.append(sentence_buffer)
                            yield sentence_buffer
                else:
                    async for chunk in english_src:
                        raw_output_chunks.append(chunk)
                        yield chunk

            if settings.fallback_enabled:
                # OSS -> managed first-token-commit fallback: swap tiers only BEFORE
                # the first token reaches the client. with_first_token_deadline bounds
                # time-to-first-token (disconnect-safe); after commit the stream runs
                # to completion on the resolved tier. Reused verbatim from fallback.py.
                async def _make_agent_text_stream(attempt):
                    async for chunk in with_first_token_deadline(
                        attempt, _raw_agent_text_stream(attempt.provider, attempt.model)
                    ):
                        yield chunk

                english_src = stream_with_fallback(
                    pipeline="chat",
                    session_id=session_id_safe,
                    variant=pipeline_variant,
                    make_stream=_make_agent_text_stream,
                )
            else:
                # No fallback: stream the single resolved primary tier directly.
                # _raw_agent_text_stream still yields the internal AGENT_ACTIVITY
                # commit sentinel; with no with_first_token_deadline wrapper on this
                # path, strip it here so only text reaches _stream_to_client.
                async def _strip_activity(_src):
                    async for _c in _src:
                        if _c is not AGENT_ACTIVITY:
                            yield _c
                english_src = _strip_activity(_raw_agent_text_stream(request_provider, request_model))

            async for _out in _stream_to_client(english_src):
                yield _out
            logger.info(f"Streaming complete for session {session_id}")
            new_messages = _stream_holder.get("new_messages", [])

            # Record trace output: translated response for translation pipeline, raw agent output otherwise.
            if get_langfuse_client:
                try:
                    if needs_output_translation and translated_output_chunks:
                        trace_output = "".join(translated_output_chunks)
                    elif raw_output_chunks:
                        trace_output = "".join(raw_output_chunks)
                    else:
                        trace_output = None
                    if trace_output:
                        langfuse = get_langfuse_client()
                        langfuse.set_current_trace_io(output=trace_output)
                        #this is the same as the update_current_trace method,
                        #but it is more explicit about the type of the output
                        # and is supported by the latest version of the langfuse SDK.
                        logger.debug("Langfuse: updated trace output")
                    # Match moderation: structured output so Langfuse shows JSON in the observation panel.
                    if agrinet_obs is not None:
                        agrinet_obs.update(
                            output={"response": trace_output or ""},
                        )
                except Exception as e:
                    logger.warning(f"Langfuse: failed to record trace output: {e}")

        # Post-processing happens AFTER streaming is complete
        messages = [
            *history,
            *new_messages
        ]

        logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
        await update_message_history(session_id, messages)
