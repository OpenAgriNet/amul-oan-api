"""
Tasks for creating conversation suggestions.
"""

from contextlib import nullcontext
import time
import re
from typing import Optional
from helpers.utils import get_logger
from app.utils import _get_message_history, trim_history, format_message_pairs, set_cache, get_cache
from app.core.cache import cache
from app.llm_core import resolver as _llm_resolver
from app.llm_core.config_model import Step as _LlmStep
from agents.suggestions import suggestions_agent
from app.config import settings
from app.services.fallback import execute_with_fallback
from langcodes import Language

logger = get_logger(__name__)

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t}


def _overlap_ratio(query: str, text: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    t = _tokenize(text)
    if not t:
        return 0.0
    return len(q & t) / len(q)


def _is_shadow_retrieval_usable(shadow_payload: dict | None) -> tuple[bool, str]:
    if not shadow_payload:
        return False, "missing_shadow_payload"
    if int(shadow_payload.get("search_return_count", 0)) < 1:
        return False, "no_search_returns"

    distilled_calls = shadow_payload.get("distilled_calls", []) or []
    if not distilled_calls:
        return False, "no_distilled_calls"

    all_snippets: list[str] = []
    has_no_result = False
    best_overlap = 0.0
    for call in distilled_calls:
        query = str(call.get("query", "") or "")
        snippets = call.get("snippets", []) or []
        if bool(call.get("no_results", False)):
            has_no_result = True
        for snippet in snippets:
            snippet_text = str(snippet or "").strip()
            if snippet_text:
                all_snippets.append(snippet_text)
                if query:
                    best_overlap = max(best_overlap, _overlap_ratio(query, snippet_text))

    if has_no_result:
        return False, "explicit_no_results"
    if len(all_snippets) < settings.suggestions_hybrid_min_snippets:
        return False, "insufficient_snippets"
    if sum(len(x) for x in all_snippets) < settings.suggestions_hybrid_min_chars:
        return False, "insufficient_evidence_chars"
    if best_overlap < settings.suggestions_hybrid_min_overlap:
        return False, "low_query_overlap"

    return True, "ok"


def _is_shadow_retrieval_usable_for_turn(
    shadow_payload: dict | None,
    expected_turn_id: str | None,
) -> tuple[bool, str]:
    if expected_turn_id:
        actual_turn_id = str((shadow_payload or {}).get("request_turn_id", "")).strip()
        if not actual_turn_id:
            return False, "missing_turn_id"
        if actual_turn_id != expected_turn_id:
            return False, "turn_id_mismatch"
    return _is_shadow_retrieval_usable(shadow_payload)


def _format_retrieval_evidence(shadow_payload: dict) -> str:
    lines: list[str] = []
    for idx, call in enumerate(shadow_payload.get("distilled_calls", []) or [], start=1):
        query = str(call.get("query", "") or "").strip() or "unknown"
        snippets = call.get("snippets", []) or []
        if not snippets:
            continue
        lines.append(f"Search {idx} Query: {query}")
        for s_idx, snippet in enumerate(snippets, start=1):
            lines.append(f"- Evidence {s_idx}: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


async def create_suggestions(
    session_id: str,
    target_lang: str = 'mr',
    profile_name: str = "managed",
    request_turn_id: str | None = None,
    *,
    queued_wall_ms: Optional[int] = None,
    queued_monotonic_s: Optional[float] = None,
):
    """
    Create and save suggestions for a session
    """
    task_start_monotonic_s = time.monotonic()
    task_start_wall_ms = int(time.time() * 1000)
    queue_delay_ms = None
    if queued_monotonic_s is not None:
        queue_delay_ms = int((task_start_monotonic_s - queued_monotonic_s) * 1000)
    elif queued_wall_ms is not None:
        queue_delay_ms = task_start_wall_ms - queued_wall_ms

    logger.info(
        "suggestions_task_started session_id=%s target_lang=%s profile=%s start_wall_ms=%s queued_wall_ms=%s queue_delay_ms=%s",
        session_id,
        target_lang,
        profile_name,
        task_start_wall_ms,
        queued_wall_ms,
        queue_delay_ms,
    )

    # Run suggestions on the same backend as the session's pipeline. The model
    # handle + display name come from the resolved primary SUGGESTIONS tier.
    sug_tier = _llm_resolver.primary_tier(_LlmStep.SUGGESTIONS, profile_name)
    sug_model = sug_tier.handle
    sug_model_name = sug_tier.model_name

    status_key = f"suggestions_{session_id}_{target_lang}:pending"
    try:
        hybrid_enabled = bool(getattr(settings, "suggestions_hybrid_enabled", False))
        normalized_target_lang = (target_lang or "").strip().lower()
        hybrid_lang_supported = normalized_target_lang in {"en", "english", "gu", "gujarati"}

        # Get message history
        raw_history = await _get_message_history(session_id)
        shadow_payload = None
        retrieval_usable = False
        retrieval_reason = "hybrid_feature_flag_disabled"
        if hybrid_enabled and hybrid_lang_supported:
            shadow_cache_key = f"suggestions_shadow_{session_id}_{target_lang}"
            shadow_payload = await get_cache(shadow_cache_key)
            retrieval_usable, retrieval_reason = _is_shadow_retrieval_usable_for_turn(
                shadow_payload,
                request_turn_id,
            )
        elif hybrid_enabled and not hybrid_lang_supported:
            retrieval_reason = "hybrid_language_not_supported"

        history = trim_history(raw_history,
                          30_000,
                          include_tool_calls=False,
                          include_system_prompts=False
                          )
        conversation_limit = (
            settings.suggestions_conversation_limit_hybrid
            if retrieval_usable
            else settings.suggestions_conversation_limit_fallback
        )
        message_pairs = "\n\n".join(format_message_pairs(history, conversation_limit))

        target_lang_name = Language.get(target_lang).display_name(target_lang)
        retrieval_section = ""
        if retrieval_usable and shadow_payload:
            retrieval_section = _format_retrieval_evidence(shadow_payload)

        if retrieval_section:
            message = (
                f"**Recent Conversation**\n\n{message_pairs}\n\n"
                f"**Retrieved Evidence**\n\n{retrieval_section}\n\n"
                f"**Using both the recent conversation and retrieved evidence, suggest 3-5 practical questions "
                f"the farmer can ask in {target_lang_name}. If there is a conflict, prioritize evidence-grounded "
                f"and answerable questions.**"
            )
            suggestion_input_mode = "hybrid"
        else:
            message = (
                f"**Conversation**\n\n{message_pairs}\n\n"
                f"**Based on the conversation, suggest 3-5 questions the farmer can ask in {target_lang_name}.**"
            )
            suggestion_input_mode = "conversation_only"

        logger.info(
            "suggestions_input_mode session_id=%s mode=%s retrieval_reason=%s conversation_pairs=%s retrieval_usable=%s hybrid_enabled=%s",
            session_id,
            suggestion_input_mode,
            retrieval_reason,
            conversation_limit,
            retrieval_usable,
            hybrid_enabled,
        )
        
        session_id_safe = (session_id or "")[:200]
        session_ctx = (
            propagate_attributes(
                session_id=session_id_safe,
                metadata={
                    "task": "suggestions",
                    "target_lang": (target_lang or "unknown")[:200],
                },
            )
            if propagate_attributes
            else nullcontext()
        )

        _lf = get_langfuse_client() if get_langfuse_client else None
        _suggestions_obs_ctx = (
            _lf.start_as_current_observation(
                name="suggestions",
                as_type="generation",
                input={
                    "session_id": session_id,
                    "target_lang": target_lang,
                    "model_name": sug_model_name,
                    "input_mode": suggestion_input_mode,
                    "retrieval_reason": retrieval_reason,
                    "request_turn_id": request_turn_id,
                    "message": message,
                },
                model=sug_model_name,
                metadata={"task": "suggestions", "target_lang": (target_lang or "unknown")[:200]},
            )
            if _lf
            else nullcontext()
        )

        with session_ctx:
            with _suggestions_obs_ctx as sug_obs:
                if settings.fallback_enabled:
                    agent_run = await execute_with_fallback(
                        pipeline="suggestions",
                        session_id=session_id_safe,
                        profile_name=profile_name,
                        run=lambda a: suggestions_agent.run(message, model=a.model),
                    )
                else:
                    agent_run = await suggestions_agent.run(message, model=sug_model)
                suggestions = [x for x in agent_run.output]
                if sug_obs is not None:
                    sug_obs.update(
                        output={"suggestions": suggestions},
                    )

        logger.info(f"Suggestions: {suggestions}")
        
        # Store suggestions in cache
        result = await set_cache(
            f"suggestions_{session_id}_{target_lang}",
            suggestions,
            ttl=settings.suggestions_cache_ttl,
        )
        cache_write_wall_ms = int(time.time() * 1000)
        logger.info(
            "suggestions_cache_written session_id=%s target_lang=%s success=%s cache_write_wall_ms=%s task_elapsed_ms=%s queue_delay_ms=%s",
            session_id,
            target_lang,
            result,
            cache_write_wall_ms,
            int((time.monotonic() - task_start_monotonic_s) * 1000),
            queue_delay_ms,
        )
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error creating suggestions: {str(e)}")
        return [] 
    finally:
        try:
            await cache.delete(status_key)
        except Exception as e:
            logger.warning(f"Error clearing suggestions pending status for session {session_id}: {e}")