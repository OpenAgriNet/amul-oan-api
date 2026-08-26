"""
Tasks for creating conversation suggestions.
"""

from contextlib import nullcontext

from agents.suggestions import suggestions_agent
from app.config import settings
from app.core.cache import cache
from app.llm_core import resolver as _llm_resolver
from app.llm_core.config_model import Step as _LlmStep
from app.services.fallback import execute_with_fallback
from app.services.suggestion_context import (
    capability_allowlist,
    extract_returned_docs,
    load_suggestion_banks,
    load_union_scheme_catalog,
    open_bank_domains,
    pick_candidates,
    tools_called_this_turn,
)
from app.utils import _get_message_history, format_message_pairs, set_cache, trim_history
from helpers.utils import get_logger
from langcodes import Language

logger = get_logger(__name__)

SUGGESTIONS_CACHE_TTL = 60*30 # 30 minutes

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

def _lang_field_for_questions(target_lang: str) -> str:
    """Pick the closest curated bank language; final output is still target_lang."""
    code = (target_lang or "").strip().lower()
    if code.startswith("gu"):
        return "gu"
    if code.startswith("hi"):
        return "hi"
    return "en"


def _format_candidates(candidates: list[dict], *, lang_field: str) -> str:
    if not candidates:
        return "- None"
    lines = []
    for item in candidates:
        text = (item.get(lang_field) or item.get("en") or "").strip()
        if not text:
            continue
        domain = item.get("domain", "unknown")
        tag = item.get("tag")
        scope = f"{domain}/{tag}" if tag else domain
        lines.append(f"- [{scope}] {text}")
    return "\n".join(lines) if lines else "- None"


def _format_docs(title: str, docs: list[str]) -> str:
    if not docs:
        return ""
    joined = "\n\n".join(f"- {text}" for text in docs if text)
    if not joined:
        return ""
    return f"**{title}**\n{joined}\n\n"


def _format_named_payloads(title: str, payloads: list[dict[str, str]]) -> str:
    if not payloads:
        return ""
    lines = []
    for payload in payloads:
        tool_name = (payload.get("tool_name") or "tool").strip()
        content = (payload.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"- [{tool_name}] {content}")
    if not lines:
        return ""
    return f"**{title}**\n" + "\n".join(lines) + "\n\n"


def _has_returned_grounding(
    returned_docs: dict,
    scheme_catalog: list[str],
) -> bool:
    return bool(
        returned_docs.get("search_chunks")
        or returned_docs.get("scheme_tool_returns")
        or returned_docs.get("contextual_tool_returns")
        or scheme_catalog
    )


def _build_suggestions_message(
    *,
    message_pairs: str,
    candidates: list[dict],
    capability_domains: list[str],
    returned_docs: dict,
    scheme_catalog: list[str],
    lang_field: str,
    target_lang_name: str,
) -> str:
    has_candidates = bool(candidates)
    has_docs = _has_returned_grounding(returned_docs, scheme_catalog)
    use_conversation_fallback = not has_candidates and not has_docs

    parts = [
        f"**Conversation**\n\n{message_pairs or '- None'}\n\n",
        f"**Candidate questions (capability-approved bank; may be en/gu/hi):**\n"
        f"{_format_candidates(candidates, lang_field=lang_field)}\n\n",
        f"**Capability allowlist:**\n"
        f"- {', '.join(capability_domains) if capability_domains else 'none'}\n\n",
        _format_docs(
            "Retrieved documents (doc-grounded questions only from these):",
            returned_docs.get("search_chunks", []),
        ),
        _format_named_payloads(
            "Scheme information (doc-grounded questions only from these):",
            returned_docs.get("scheme_tool_returns", []),
        ),
        _format_docs(
            "Union scheme catalog (cached; doc-grounded questions only from these):",
            scheme_catalog,
        ),
        _format_named_payloads(
            "Other returned tool data (optional context; bank candidates remain valid):",
            returned_docs.get("contextual_tool_returns", []),
        ),
    ]

    if use_conversation_fallback:
        parts.append(
            f"**Conversation fallback:** No tool candidates or returned docs this turn. "
            f"Suggest 3-5 follow-up questions in {target_lang_name} grounded in the "
            f"conversation and limited to the capability allowlist.\n"
        )
    else:
        parts.append(
            f"Suggest 3-5 questions the farmer can ask in {target_lang_name}. "
            f"Prefer candidate-bank questions when present (rewrite into {target_lang_name} "
            f"if needed). You may also add questions grounded only in returned docs/catalog. "
            f"Bank candidates do not need to be answerable from returned tool docs.\n"
        )

    return "".join(parts)


async def create_suggestions(
    session_id: str,
    target_lang: str = "mr",
    profile_name: str = "managed",
    farmer_unions: list[str] | None = None,
):
    """
    Create and save suggestions for a session
    """
    logger.info(f"Getting suggestions for session {session_id}")

    # Run suggestions on the same backend as the session's pipeline: OSS sessions
    # use the self-hosted gemma model (no API cost; completes full-OSS for chat),
    # legacy stays on the default model. The suggestions model handle + display
    # name come from the resolved primary SUGGESTIONS tier (the only path); when
    # the session's variant has no OSS profile the resolver falls back to the
    # managed tier, matching the old oss_model_available() guard.
    sug_tier = _llm_resolver.primary_tier(_LlmStep.SUGGESTIONS, profile_name)
    sug_model = sug_tier.handle
    sug_model_name = sug_tier.model_name

    status_key = f"suggestions_{session_id}_{target_lang}:pending"
    try:
        raw_history = await _get_message_history(session_id)
        history = trim_history(raw_history,
                          30_000,
                          include_tool_calls=False,
                          include_system_prompts=False
                          )
        message_pairs = "\n\n".join(format_message_pairs(history, 5))
        tools_called = tools_called_this_turn(raw_history)
        banks = load_suggestion_banks()
        open_domains = open_bank_domains(
            tools_called,
            farmer_unions,
            enable_network=settings.enable_network,
            loan_feature_enabled=settings.loan_feature_enabled,
            banks=banks,
        )
        candidates = pick_candidates(
            open_domains,
            banks,
            tools_called=tools_called,
            max_candidates=10,
        )
        returned_docs = extract_returned_docs(raw_history, max_search_chunks=2, max_chars=1200)
        scheme_catalog = await load_union_scheme_catalog(tools_called, farmer_unions)
        lang_field = _lang_field_for_questions(target_lang)
        capability_domains = capability_allowlist(
            farmer_unions,
            enable_network=settings.enable_network,
            loan_feature_enabled=settings.loan_feature_enabled,
        )

        target_lang_name = Language.get(target_lang).display_name(target_lang)
        message = _build_suggestions_message(
            message_pairs=message_pairs,
            candidates=candidates,
            capability_domains=capability_domains,
            returned_docs=returned_docs,
            scheme_catalog=scheme_catalog,
            lang_field=lang_field,
            target_lang_name=target_lang_name,
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
                    "tools_called": tools_called,
                    "open_domains": open_domains,
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
        result = await set_cache(f"suggestions_{session_id}_{target_lang}", suggestions, ttl=SUGGESTIONS_CACHE_TTL)
        logger.info(f"Suggestions saved for session {session_id}: {result}")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error creating suggestions: {str(e)}")
        return [] 
    finally:
        try:
            await cache.delete(status_key)
        except Exception as e:
            logger.warning(f"Error clearing suggestions pending status for session {session_id}: {e}")