"""Redis-backed chat history state."""

from typing import List

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

from app.redis.cache import get_cache, set_cache
from app.redis.config import MESSAGE_HISTORY_TTL_SECONDS
from app.redis.config import key as redis_key

HISTORY_NAMESPACE = "history"
MODERATION_HISTORY_NAMESPACE = "moderation_history"


def _history_key(session_id: str) -> str:
    return redis_key(HISTORY_NAMESPACE, session_id)


def _moderation_history_key(session_id: str) -> str:
    return redis_key(MODERATION_HISTORY_NAMESPACE, session_id)


async def get_message_history(session_id: str) -> List[ModelMessage]:
    message_history = await get_cache(_history_key(session_id))
    if message_history:
        return ModelMessagesTypeAdapter.validate_python(message_history)
    return []


async def get_moderation_history(session_id: str) -> List[ModelMessage]:
    moderation_history = await get_cache(_moderation_history_key(session_id))
    if moderation_history:
        return ModelMessagesTypeAdapter.validate_python(moderation_history)
    return []


async def update_message_history(session_id: str, all_messages: List[ModelMessage]):
    await set_cache(
        _history_key(session_id),
        to_jsonable_python(all_messages),
        ttl=MESSAGE_HISTORY_TTL_SECONDS,
    )


async def update_moderation_history(
    session_id: str,
    moderation_messages: List[ModelMessage],
):
    await set_cache(
        _moderation_history_key(session_id),
        to_jsonable_python(moderation_messages),
        ttl=MESSAGE_HISTORY_TTL_SECONDS,
    )

