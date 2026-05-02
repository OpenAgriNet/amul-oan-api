"""
Tasks for creating conversation suggestions.
"""

from contextlib import nullcontext
from fastapi import BackgroundTasks
from helpers.utils import get_logger
from app.utils import _get_message_history, trim_history, format_message_pairs, set_cache
from agents.suggestions import suggestions_agent
from langcodes import Language

logger = get_logger(__name__)

SUGGESTIONS_CACHE_TTL = 60*30 # 30 minutes

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

async def create_suggestions(session_id: str, target_lang: str = 'mr'):
    """
    Create and save suggestions for a session
    """
    logger.info(f"Getting suggestions for session {session_id}")

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
        
        session_ctx = (
            propagate_attributes(
                session_id=(session_id or "")[:200],
                trace_name="suggestions",
                metadata={
                    "task": "suggestions",
                    "target_lang": (target_lang or "unknown")[:200],
                },
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
                            "session_id": session_id,
                            "target_lang": target_lang,
                            "message": message,
                        }
                    )
                except Exception as e:
                    logger.warning("Langfuse: failed to set suggestions trace input: %s", e)

            # Run the agent
            agent_run = await suggestions_agent.run(message)
            suggestions = [x for x in agent_run.output]

            if get_langfuse_client:
                try:
                    langfuse = get_langfuse_client()
                    langfuse.set_current_trace_io(output={"suggestions": suggestions})
                except Exception as e:
                    logger.warning("Langfuse: failed to set suggestions trace output: %s", e)

        logger.info(f"Suggestions: {suggestions}")
        
        # Store suggestions in cache
        result = await set_cache(f"suggestions_{session_id}_{target_lang}", suggestions, ttl=SUGGESTIONS_CACHE_TTL)
        logger.info(f"Suggestions saved for session {session_id}: {result}")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error creating suggestions: {str(e)}")
        return [] 