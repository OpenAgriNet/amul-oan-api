from typing import AsyncGenerator
from fastapi import BackgroundTasks
from agents.agrinet import agrinet_agent
from agents.moderation import moderation_agent
from helpers.utils import get_logger
from app.utils import (
    update_message_history, 
    trim_history, 
    format_message_pairs
)
from app.tasks.suggestions import create_suggestions
from agents.deps import FarmerContext


logger = get_logger(__name__)

async def stream_chat_messages(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    user_info: dict,
    background_tasks: BackgroundTasks,
    
) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    # Generate a unique content ID for this query
    content_id = f"query_{session_id}_{len(history)//2 + 1}"
    logger.info(f"User info: {user_info}")
    deps = FarmerContext(query=query,
                         lang_code=target_lang,
                         farmer_id=user_info.get('farmer_id')
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

    # Use run_stream which handles tool calls internally and only streams the final text response
    async with agrinet_agent.run_stream(
        user_prompt=user_message,
        message_history=trimmed_history,
        deps=deps
    ) as response:
        async for text in response.stream_text(delta=True):
            yield text
        
    # Get the result and new messages after streaming completes
    new_messages = response.new_messages() if response else []
    logger.info(f"Streaming complete for session {session_id}")

    # Post-processing happens AFTER streaming is complete
    messages = [
        *history,
        *new_messages
    ]

    logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
    await update_message_history(session_id, messages)