"""What voice keeps in history in place of the caller's words, the pair it writes,
and how it cleans history before the agent reads it.

From voice-oan-api (``app/services/voice.py`` and ``app/utils.py`` at amul-dev
``3b19835``), shared by the classifiers, the gate and the agent input.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from helpers.utils import get_logger

logger = get_logger(__name__)

HISTORY_MARKERS = {
    "greeting": "hello",
    "fragment": "[fragment]",
    "low_confidence": "[unclear-user-input]",
    "pretranslation_failed": "[pretranslation-failed]",
    "stt_no_audio": "[stt:no-audio]",
    "stt_unclear": "[stt:unclear-speech]",
    "moderation_reject": "[moderation-rejected]",
    "outbound_intro": "[outbound-call-started]",
}


def history_pair(user_text: str, assistant_text: str) -> tuple[ModelRequest, ModelResponse]:
    return (
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=assistant_text)]),
    )


def clean_message_history_for_openai(history: List[ModelMessage]) -> List[ModelMessage]:
    """Clean message history to ensure it's safe for OpenAI API.
    
    Removes orphaned tool calls (tool calls without responses) from the message history
    to prevent OpenAI API errors. Processes messages in order and removes any tool call 
    parts that don't have corresponding tool response parts.
    
    Args:
        history: List of messages to clean
        
    Returns:
        Cleaned list of messages safe for OpenAI API
    """
    if not history:
        return []
    
    logger.debug(f"Cleaning message history with {len(history)} messages")
    
    # First pass: collect all tool call IDs and their corresponding responses
    tool_calls = set()
    tool_responses = set()
    
    for message in history:
        for part in message.parts:
            part_kind = getattr(part, "part_kind", "")
            tool_call_id = getattr(part, "tool_call_id", None)
            
            if not tool_call_id:
                continue
                
            if part_kind == "tool-call":
                tool_calls.add(tool_call_id)
            elif part_kind in ("tool-return", "retry-prompt"):
                tool_responses.add(tool_call_id)
    
    # Identify orphaned tool calls (calls without responses)
    orphaned_calls = tool_calls - tool_responses
    
    # Second pass: filter out orphaned tool calls and their responses
    cleaned_history = []
    
    for message in history:
        cleaned_parts = []
        
        for part in message.parts:
            part_kind = getattr(part, "part_kind", "")
            tool_call_id = getattr(part, "tool_call_id", None)
            
            # Skip orphaned tool calls
            if part_kind == "tool-call" and tool_call_id in orphaned_calls:
                logger.debug(f"Removing orphaned tool call: {tool_call_id}")
                continue
            
            # Skip responses to orphaned tool calls
            if part_kind in ("tool-return", "retry-prompt") and tool_call_id in orphaned_calls:
                logger.debug(f"Removing response to orphaned tool call: {tool_call_id}")
                continue
            
            cleaned_parts.append(part)
        
        # Only keep messages with remaining parts
        if cleaned_parts:
            cleaned_message = deepcopy(message)
            cleaned_message.parts = cleaned_parts
            cleaned_history.append(cleaned_message)
    
    if orphaned_calls:
        logger.warning(f"Removed {len(orphaned_calls)} orphaned tool calls: {orphaned_calls}")
    
    logger.info(f"Cleaned message history: {len(history)} -> {len(cleaned_history)} messages")
    return cleaned_history
