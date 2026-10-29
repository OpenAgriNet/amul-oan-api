"""
Tasks for creating conversation suggestions.
"""

from contextlib import nullcontext
from helpers.utils import get_logger
from app.utils import _get_message_history, trim_history, format_message_pairs, set_cache
from app.core.cache import cache
from app import llm_core
from app.llm_core import Step as _LlmStep
from app.llm_core import trace as _pipeline_trace
from agents.suggestions import suggestions_agent
from app.observability import surface_environment
from langcodes import Language

logger = get_logger(__name__)

SUGGESTIONS_CACHE_TTL = 60*30 # 30 minutes

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None


def _step_metadata(execution) -> dict[str, str]:
    """turn.v1's metadata for the suggestions step: the tier that served it."""
    try:
        return _pipeline_trace.step_metadata(execution.begin_trace(), _LlmStep.SUGGESTIONS)
    except Exception as e:  # tracing must never break suggestions
        logger.debug("suggestions step metadata unavailable: %s", e)
        return {}


async def create_suggestions(
    session_id: str,
    target_lang: str,
    execution: llm_core.ExecutionContext,
    turn_root: dict[str, str] | None = None,
):
    """
    Create and save suggestions for a session.

    The ``suggestions`` step of the turn that asked for them (amul-telemetry
    telemetry/turn.v1.yaml), which runs after the response: ``turn_root`` is that
    turn's root span as a Langfuse trace context, so the step nests under the root
    though it has closed. A step writes no trace-level field: a late one would
    rewrite the turn.
    """
    logger.info(f"Getting suggestions for session {session_id}")

    sug_model_name = execution.info(_LlmStep.SUGGESTIONS).model_name

    status_key = f"suggestions_{session_id}_{target_lang}:pending"
    try:
        # Get message history
        raw_history = await _get_message_history(session_id)
        history = trim_history(raw_history,
                          30_000,
                          include_tool_calls=False,
                          include_system_prompts=False
                          )
        message_pairs = "\n\n".join(format_message_pairs(history, 5))

        target_lang_name = Language.get(target_lang).display_name(target_lang)
        message = f"**Conversation**\n\n{message_pairs}\n\n**Based on the conversation, suggest 3-5 questions the farmer can ask in {target_lang_name}.**"
        
        # Deferred past the turn, so outside its context: name chat's environment
        # explicitly. Only the environment, which is not a trace-level field.
        environment_ctx = (
            propagate_attributes(environment=surface_environment("chat"))
            if propagate_attributes
            else nullcontext()
        )

        _lf = get_langfuse_client() if get_langfuse_client else None
        _suggestions_obs_ctx = (
            _lf.start_as_current_observation(
                trace_context=turn_root,
                name="suggestions",
                as_type="generation",
                input={
                    "session_id": session_id,
                    "target_lang": target_lang,
                    "model_name": sug_model_name,
                    "message": message,
                },
                model=sug_model_name,
            )
            if _lf
            else nullcontext()
        )

        with environment_ctx:
            with _suggestions_obs_ctx as sug_obs:
                agent_run = await execution.run(
                    _LlmStep.SUGGESTIONS,
                    suggestions_agent,
                    message,
                )
                suggestions = [x for x in agent_run.output]
                if sug_obs is not None:
                    sug_obs.update(
                        output={"suggestions": suggestions},
                        metadata=_step_metadata(execution),
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
