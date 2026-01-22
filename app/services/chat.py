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
# from app.tasks.suggestions import create_suggestions  # Commented out: suggestion agent disabled
from agents.deps import FarmerContext
from pydantic_ai import Agent, FinalResultEvent, ToolCallPart, BuiltinToolCallPart

logger = get_logger(__name__)

async def stream_chat_messages(
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    background_tasks: BackgroundTasks
) -> AsyncGenerator[str, None]:
    """Async generator for streaming chat messages."""
    # Generate a unique content ID for this query
    content_id = f"query_{session_id}_{len(history)//2 + 1}"
    logger.info(f"User info: {user_info}")
    
    # Extract only the 'data' field from JWT token (ignore standard JWT fields like iss, sub, exp)
    farmer_data = user_info.get('data') if user_info else None
    
    deps = FarmerContext(
        query=query,
        lang_code=target_lang,
        session_id=session_id,
        farmer_info=farmer_data if farmer_data else None
    )

    message_pairs = "\n\n".join(format_message_pairs(history, 3))
    logger.info(f"Message pairs: {message_pairs}")
    if message_pairs:
        last_response = f"**Conversation**\n\n{message_pairs}\n\n---\n\n"
    else:
        last_response = ""
    
    user_message    = f"{last_response}{deps.get_user_message()}"
    moderation_run  = await moderation_agent.run(user_message)
    moderation_data = moderation_run.output
    logger.info(f"Moderation data: {moderation_data}")

    
    deps.update_moderation_str(str(moderation_data))

    user_message = deps.get_user_message()
    logger.info(f"Running agent with user message: {user_message}")

    # Run the main agent
    trimmed_history = trim_history(
        history,
        max_tokens=80_000
    )
    
    logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

    async with agrinet_agent.iter(user_prompt=user_message, message_history=trimmed_history, deps=deps) as agent_run:
        async for node in agent_run:
            if Agent.is_user_prompt_node(node):
                logger.info(f"User prompt node: {node.user_prompt}")
            elif Agent.is_model_request_node(node):                
                # is_call_tools_node / is_end_node
                if Agent.is_call_tools_node(node):
                    logger.info("Tool execution node")
                elif Agent.is_end_node(node):
                    logger.info(f"End node reached: {node.data.output}")
                    break

                async with node.stream(agent_run.ctx) as request_stream:
                    final_result_seen = False
                    
                    async for event in request_stream:
                        if isinstance(event, FinalResultEvent):
                            final_result_seen = True
                    
                    # After stream completes, check if response contains tool calls
                    # Only stream text if there are NO tool calls (this is the final answer)
                    if final_result_seen:
                        has_tool_calls = any(
                            isinstance(part, (ToolCallPart, BuiltinToolCallPart))
                            for part in request_stream.response.parts
                        )
                        parts_info = [(type(p).__name__, getattr(p, 'tool_name', None) or (p.content[:50] if hasattr(p, 'content') and p.content else '')) for p in request_stream.response.parts]
                        logger.info(f"[Stream Complete] Response parts: {parts_info}, has_tool_calls: {has_tool_calls}")
                        
                        if has_tool_calls:
                            logger.info("[Result] Skipping text output - response contains tool calls")
                        else:
                            # No tool calls - this is the final result, yield the complete text
                            logger.info("[Result] Streaming final result text")
                            # Get text parts from the response and yield them
                            for part in request_stream.response.parts:
                                if hasattr(part, 'content') and part.content and not isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
                                    # Skip ThinkingPart content - only yield TextPart
                                    if type(part).__name__ == 'TextPart':
                                        yield part.content
            elif Agent.is_call_tools_node(node):
                logger.info("Tool execution node")
            elif Agent.is_end_node(node):
                logger.info(f"End node reached: {node.data.output}")
                break

    # Get the result and new messages after streaming completes
    new_messages = agent_run.result.new_messages() if agent_run and agent_run.result else []
    logger.info(f"Streaming complete for session {session_id}")
    
    # Post-processing happens AFTER streaming is complete
    messages = [
        *history,
        *new_messages
    ]

    logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
    await update_message_history(session_id, messages)
