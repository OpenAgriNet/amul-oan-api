"""Feedback-related tools and utilities."""

from typing import Literal

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from helpers.utils import get_logger

logger = get_logger(__name__)


def signal_conversation_state(
    ctx: RunContext[FarmerContext],
    event: Literal["conversation_closing", "user_frustration", "in_progress"],
) -> str:
    logger.info("Conversation state signaled: %s", event)
    return f"State {event} recorded."

