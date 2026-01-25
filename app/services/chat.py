from typing import AsyncGenerator
from fastapi import BackgroundTasks
from agents.agrinet import agrinet_agent
from agents.moderation import moderation_agent
from agents.models import LLM_PROVIDER
from helpers.utils import get_logger
from app.utils import (
    update_message_history, 
    trim_history, 
    format_message_pairs
)
from helpers.telemetry import create_moderation_event, TelemetryRequest
from app.tasks.telemetry import send_telemetry
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
    
    # Extract only the 'data' field from JWT token (ignore standard JWT fields like iss, sub, exp)
    farmer_data = user_info.get('data') if user_info else None
    
    deps = FarmerContext(
        query=query,
        lang_code=target_lang,
        farmer_info=farmer_data if farmer_data else None
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

    # # Create the moderation event
    # moderation_event = create_moderation_event(
    #     question_text=query,
    #     moderation_type="TEXT_MODERATION",
    #     content_id=content_id,
    #     session_id=session_id,
    #     content_type="text",
    #     moderation_service="agent_moderation",
    #     flagged=moderation_data.category != "valid_agricultural",
    #     category=moderation_data.category,
    #     action=moderation_data.action,
    #     uid=user_id
    # )                    
    
    # # Create the telemetry request and send it
    # _telemetry_request = TelemetryRequest(events=[moderation_event])
    # background_tasks.add_task(send_telemetry, _telemetry_request.dict())
        # Generate suggestions after moderation passes
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

    if LLM_PROVIDER == 'anthropic':
        # For Anthropic: Use agent.iter() + node.stream() instead of run_stream()
        # This correctly handles streaming through tool calls (run_stream only yields pre-tool text)
        async with agrinet_agent.iter(
            user_prompt=user_message,
            message_history=trimmed_history,
            deps=deps,
        ) as agent_run:
            async for node in agent_run:
                node_type = type(node).__name__

                if node_type == 'ModelRequestNode':
                    # Stream text from ModelRequestNode
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            event_type = type(event).__name__

                            # Capture initial text from PartStartEvent (TextPart)
                            if event_type == 'PartStartEvent':
                                if hasattr(event, 'part'):
                                    part_type = type(event.part).__name__
                                    if part_type == 'TextPart' and hasattr(event.part, 'content'):
                                        text = event.part.content
                                        if text:
                                            yield text

                            # Capture text deltas from PartDeltaEvent
                            elif event_type == 'PartDeltaEvent':
                                if hasattr(event, 'delta'):
                                    delta_type = type(event.delta).__name__
                                    if delta_type == 'TextPartDelta':
                                        text = event.delta.content_delta
                                        if text:
                                            yield text

            logger.info(f"Streaming complete for session {session_id}")
            new_messages = agent_run.new_messages()
    else:
        # For OpenAI/vLLM: Use standard run_stream()
        async with agrinet_agent.run_stream(
            user_prompt=user_message,
            message_history=trimmed_history,
            deps=deps,
        ) as response_stream:
            async for chunk in response_stream.stream_text(delta=True):
                yield chunk

            logger.info(f"Streaming complete for session {session_id}")
            # Capture the data we need while response_stream is still available
            new_messages = response_stream.new_messages()

    # Post-processing happens AFTER streaming is complete
    messages = [
        *history,
        *new_messages
    ]

    logger.info(f"Updating message history for session {session_id} with {len(messages)} messages")
    await update_message_history(session_id, messages)