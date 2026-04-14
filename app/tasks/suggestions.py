"""
Tasks for creating conversation suggestions.

Cache writes are owned by the suggestions router (cache-aside); this module only
generates suggestion lists from message history.
"""

from helpers.utils import get_logger
from app.utils import _get_message_history, trim_history, format_message_pairs
from agents.suggestions import suggestions_agent
from langcodes import Language

logger = get_logger(__name__)


async def create_suggestions(session_id: str, target_lang: str = "mr") -> list:
    """
    Generate suggestions for a session from Redis message history (no cache I/O).
    """
    logger.info("Generating suggestions for session %s", session_id)

    raw_history = await _get_message_history(session_id)
    history = trim_history(
        raw_history,
        30_000,
        include_tool_calls=False,
        include_system_prompts=False,
    )
    message_pairs = "\n\n".join(format_message_pairs(history, 5))

    target_lang_name = Language.get(target_lang).display_name(target_lang)
    message = (
        f"**Conversation**\n\n{message_pairs}\n\n"
        f"**Based on the conversation, suggest 3-5 questions the farmer can ask "
        f"in {target_lang_name}.**"
    )

    agent_run = await suggestions_agent.run(message)
    suggestions = [x for x in agent_run.output]
    logger.info("Suggestions: %s", suggestions)
    return suggestions