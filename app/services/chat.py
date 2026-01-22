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
from pydantic_ai import (
    Agent,
    FinalResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.messages import TextPart, ThinkingPart

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
       
    deps = FarmerContext(query=query, lang_code=target_lang, session_id=session_id)

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
        max_tokens=80_000,
        include_system_prompts=True,
        include_tool_calls=True
    )
    
    logger.info(f"Trimmed history length: {len(trimmed_history)} messages")

    async with agrinet_agent.iter(user_prompt=user_message, message_history=trimmed_history, deps=deps) as agent_run:
        async for node in agent_run:
            if Agent.is_user_prompt_node(node):
                logger.info(f"User prompt node: {node.user_prompt}")
                continue
            elif Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as response_stream:
                    final_result_found = False
                    
                    async for event in response_stream:
                        if isinstance(event, PartStartEvent):
                            if isinstance(event.part, ThinkingPart):
                                logger.info("Reasoning part started (not streamed to user)")
                            elif isinstance(event.part, TextPart):
                                pass
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, ThinkingPartDelta):
                                # Don't stream reasoning to user
                                pass
                            elif isinstance(event.delta, TextPartDelta):
                                # Only yield text deltas after FinalResultEvent
                                if final_result_found and event.delta.content_delta:
                                    yield event.delta.content_delta
                        elif isinstance(event, FinalResultEvent):
                            logger.info("[Result] The model started producing a final result")
                            final_result_found = True
                            # Don't break - continue to collect text deltas
            elif Agent.is_call_tools_node(node):
                logger.info("Tool execution node")
                continue
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
