import uuid
from dataclasses import dataclass
from typing import List, Optional
from app.core.cache import cache, redis_client, build_cache_key  # Import cache instance from core
from app.config import settings
from helpers.utils import get_logger, count_tokens_for_part
from copy import deepcopy
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelMessage,
    SystemPromptPart,
)
from pydantic_core import to_jsonable_python

HISTORY_SUFFIX = "_SVA"
SESSION_OWNER_SUFFIX = "_active_request"
SESSION_EPOCH_SUFFIX = "_request_epoch"
SESSION_TAKEOVER_SUFFIX = "_taken_over_by"

# Conversation-history retention in Redis (env HISTORY_CACHE_TTL_SECONDS, default 2h).
# Rolling inactivity window: update_message_history rewrites the key with this TTL
# each turn, so history expires this long after the last turn. session_id is
# client-supplied and the backend enforces no session/call length, so the only
# requirement is TTL > the gap between turns; exact value is non-load-bearing.
DEFAULT_CACHE_TTL = settings.history_cache_ttl_seconds

logger = get_logger(__name__)


@dataclass(frozen=True)
class SessionRequestOwner:
    session_id: str
    request_token: str
    epoch: int
    #: The trace id this request's turn opens with, made when it claimed the session,
    #: so the request it takes over from can name it (turn.v1's amul.superseded_by).
    trace_id: Optional[str] = None
    #: turn.v1's amul.utterance_id: a request that comes in while another holds the
    #: session is the same utterance, so it keeps that one's; else its own trace id.
    utterance_id: Optional[str] = None


def _session_owner_key(session_id: str) -> str:
    return build_cache_key(f"{session_id}_{SESSION_OWNER_SUFFIX}")


def _session_epoch_key(session_id: str) -> str:
    return build_cache_key(f"{session_id}_{SESSION_EPOCH_SUFFIX}")


def _session_takeover_key(session_id: str) -> str:
    return build_cache_key(f"{session_id}_{SESSION_TAKEOVER_SUFFIX}")


# One step, so two requests claiming at once cannot leave the older one owning the
# session. The owner's token is "<epoch>:<trace id>:<utterance id>". A request that
# finds another still holding the session takes its utterance, and records its own
# trace id under that one's epoch, for it to read once it finds itself stale. A token
# of another form (an older build's) names no utterance: the request starts its own.
_CLAIM_SCRIPT = """
local previous = redis.call('get', KEYS[2])
local epoch = redis.call('incr', KEYS[1])
local utterance = ARGV[1]
if previous then
    utterance = string.match(previous, '^%d+:%x+:(%x+)$') or ARGV[1]
    local previous_epoch = string.match(previous, '^(%d+):')
    if previous_epoch then
        redis.call('hset', KEYS[3], previous_epoch, ARGV[1])
        redis.call('expire', KEYS[3], tonumber(ARGV[3]))
    end
end
local token = epoch .. ':' .. ARGV[1] .. ':' .. utterance
redis.call('set', KEYS[2], token, 'ex', tonumber(ARGV[2]))
return {epoch, token, utterance}
"""


async def claim_session_request_ownership(session_id: str) -> SessionRequestOwner:
    trace_id = uuid.uuid4().hex
    epoch, request_token, utterance_id = await redis_client.eval(
        _CLAIM_SCRIPT,
        3,
        _session_epoch_key(session_id),
        _session_owner_key(session_id),
        _session_takeover_key(session_id),
        trace_id,
        str(settings.session_owner_ttl_seconds),
        # As long as the conversation's history: a request reads it before its turn ends.
        str(DEFAULT_CACHE_TTL),
    )
    return SessionRequestOwner(
        session_id=session_id,
        request_token=request_token,
        epoch=int(epoch),
        trace_id=trace_id,
        utterance_id=utterance_id,
    )


async def session_request_taken_over_by(owner: SessionRequestOwner | None) -> Optional[str]:
    """The trace id of the request that took the session over from ``owner``, if one did."""
    if owner is None:
        return None
    return await redis_client.hget(_session_takeover_key(owner.session_id), str(owner.epoch))


async def is_session_request_owner(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    current = await redis_client.get(_session_owner_key(owner.session_id))
    return current == owner.request_token


async def refresh_session_request_ownership(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    refreshed = await redis_client.eval(
        """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], tonumber(ARGV[2]))
        end
        return 0
        """,
        1,
        _session_owner_key(owner.session_id),
        owner.request_token,
        str(settings.session_owner_ttl_seconds),
    )
    return bool(refreshed)


async def release_session_request_ownership(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    deleted = await redis_client.eval(
        """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """,
        1,
        _session_owner_key(owner.session_id),
        owner.request_token,
    )
    return bool(deleted)

# Cache utility functions
async def get_cache(key: str):
    """
    Get value from cache.
    
    Args:
        key: Cache key to retrieve
        
    Returns:
        Cached value or None if not found
    """
    return await cache.get(key)


async def set_cache(key: str, value, ttl: int = DEFAULT_CACHE_TTL):
    """
    Set value in cache with TTL.
    
    Args:
        key: Cache key to store under
        value: Value to cache (will be JSON serialized)
        ttl: Time to live in seconds (default: 24 hours)
        
    Returns:
        True if successful
    """
    await cache.set(key, value, ttl=ttl)
    return True


# pydantic-ai usage integer fields. The 0.2.4 schema (request_tokens /
# response_tokens / total_tokens / requests) and the 1.x schema (input_tokens /
# output_tokens / cache_* tokens) both store these as ints, but legacy turns
# where the model reported no usage (streamed / vLLM-gemma responses) persisted
# them as null. The 1.x ModelMessagesTypeAdapter requires int, so loading that
# history raises ValidationError. Coerce nulls to 0 on read.
_USAGE_INT_FIELDS = (
    "requests", "request_tokens", "response_tokens", "total_tokens",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "input_audio_tokens", "cache_audio_read_tokens", "output_audio_tokens",
)


def _sanitize_legacy_usage(message_history):
    """Coerce null usage token counts in cached history to 0 so the pydantic-ai
    1.x adapter can validate history written by older revisions. Mutates and
    returns the same structure; best-effort and never raises."""
    if not isinstance(message_history, list):
        return message_history
    for msg in message_history:
        if not isinstance(msg, dict):
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        for field in _USAGE_INT_FIELDS:
            if field in usage and usage.get(field) is None:
                usage[field] = 0
        details = usage.get("details")
        if isinstance(details, dict):
            for key, value in list(details.items()):
                if value is None:
                    details[key] = 0
        elif details is None and "details" in usage:
            usage["details"] = {}
    return message_history


async def _get_message_history(session_id: str) -> List[ModelMessage]:
    """Get or initialize message history."""
    message_history = await get_cache(f"{session_id}_{HISTORY_SUFFIX}")
    if not message_history:
        return []
    message_history = _sanitize_legacy_usage(message_history)
    try:
        return ModelMessagesTypeAdapter.validate_python(message_history)
    except Exception as exc:
        # Never 500 a live call on unreadable history (e.g. future format
        # drift). Drop it and proceed as a fresh turn — degraded, not broken.
        logger.warning(
            "Discarding unreadable message history for session %s: %s",
            session_id, exc,
        )
        return []

async def update_message_history(session_id: str, all_messages: List[ModelMessage], ttl: Optional[int] = None):
    """Update message history. ``ttl`` is the surface's own retention (``history_ttl``)."""
    await set_cache(f"{session_id}_{HISTORY_SUFFIX}", to_jsonable_python(all_messages), ttl=ttl or DEFAULT_CACHE_TTL)


def history_ttl(surface: str) -> dict:
    """``update_message_history``'s ``ttl`` when the surface's pipeline file sets
    its history retention; empty otherwise, so the write is as before."""
    from app.llm_core import runtime

    ttl = runtime.surface_settings(surface).history_ttl_seconds
    return {} if ttl is None else {"ttl": ttl}


def get_message_pairs(history: List[ModelMessage], limit: int = None) -> List[List]:
    """Extract user/assistant message part pairs from history, starting with the most recent.
    
    Args:
        history: List of messages (ModelMessage objects)
        limit: Maximum number of message pairs to return (None = all pairs)
        
    Returns:
        List of [UserPromptPart, TextPart] pairs, starting with the most recent
    """
    if not history:
        return []
    
    pairs = []
    # Process messages in reverse chronological order (newest first)
    i = len(history) - 1
    
    while i > 0 and (limit is None or len(pairs) < limit):
        # Find the nearest assistant message (with 'text' part)
        assistant_idx = None
        text_part = None
        for j in range(i, -1, -1):
            # Find the TextPart in the message
            for part in history[j].parts:
                if getattr(part, "part_kind", "") == "text":
                    assistant_idx = j
                    text_part = part
                    break
            if assistant_idx is not None:
                break
        
        if assistant_idx is None or text_part is None:
            break  # No more assistant messages
            
        # Find the nearest user message before the assistant message
        user_idx = None
        user_part = None
        for j in range(assistant_idx - 1, -1, -1):
            # Find the UserPromptPart in the message
            for part in history[j].parts:
                if getattr(part, "part_kind", "") == "user-prompt":
                    user_idx = j
                    user_part = part
                    break
            if user_idx is not None:
                break
                
        if user_idx is None or user_part is None:
            break  # No more user messages
            
        # Add the pair and continue searching from before this pair
        pairs.append([deepcopy(user_part), deepcopy(text_part)])
        i = user_idx - 1
        
    return pairs

def format_message_pairs(history: List[ModelMessage], limit: int = None) -> List[str]:
    """Format user/assistant message pairs as strings with custom headers.
    
    Args:
        history: List of messages (ModelMessage objects)
        limit: Maximum number of message pairs to return (None = all pairs)
        
    Returns:
        List of formatted strings containing user and assistant messages
    """
    pairs = get_message_pairs(history, limit)
    formatted_messages = []
    
    for user_part, assistant_part in pairs:
        formatted_pair = f"""**User Message**:\n{user_part.content}\n\n**Assistant Message**:\n{assistant_part.content}"""
        formatted_messages.append(formatted_pair)
    
    return formatted_messages

def trim_history(
    history: List[ModelMessage],
    max_tokens: int = 28_000,
    *,
    include_system_prompts: bool = True,
    include_tool_calls: bool = True,
) -> List[ModelMessage]:
    # 1. Pre-process system parts: strip them or keep whole messages
    prepped: List[ModelMessage] = []
    for msg in history:
        if include_system_prompts:
            prepped.append(msg)
        else:
            # remove only the system parts, keep any other parts (like user-prompt)
            new_parts = [p for p in msg.parts if not isinstance(p, SystemPromptPart)]
            if new_parts:
                m2 = deepcopy(msg)
                m2.parts = new_parts
                prepped.append(m2)

    # 2. Split into "turns" at each user message
    turns: List[List[ModelMessage]] = []
    current: List[ModelMessage] = []
    for msg in prepped:
        is_user = any(getattr(p, "part_kind", "") == "user-prompt" for p in msg.parts)
        if is_user and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    # 3. Globally identify all paired tool calls/returns/retries across all turns
    # This prevents orphaned tool calls/returns/retries from being retained
    all_calls = set()
    all_returns = set()
    all_retries = set()
    
    # First pass: collect all tool call, return, and retry IDs globally across entire history
    for turn in turns:
        for m in turn:
            for p in m.parts:
                kind = getattr(p, "part_kind", "")
                if kind == "tool-call" and hasattr(p, 'tool_call_id'):
                    all_calls.add(p.tool_call_id)
                elif kind == "tool-return" and hasattr(p, 'tool_call_id'):
                    all_returns.add(p.tool_call_id)
                elif kind == "retry-prompt" and hasattr(p, 'tool_call_id'):
                    all_retries.add(p.tool_call_id)
    
    # Keep tool calls that have either a return OR a retry (both are valid responses)
    good_ids = all_calls & (all_returns | all_retries)
    
    # 4. Filter each turn using the global good_ids to remove orphaned tool calls/returns/retries
    clean_turns: List[List[ModelMessage]] = []
    for turn in turns:
        filtered: List[ModelMessage] = []
        for m in turn:
            kept = []
            for p in m.parts:
                # drop any part with an empty 'content' attribute
                if hasattr(p, "content") and not getattr(p, "content"):
                    continue
                kind = getattr(p, "part_kind", "")
                if kind in ("tool-call", "tool-return", "retry-prompt"):
                    # Use global good_ids to filter out orphaned tool calls/returns/retries
                    if not include_tool_calls or not hasattr(p, 'tool_call_id') or p.tool_call_id not in good_ids:
                        continue
                kept.append(p)
            if kept:
                m2 = deepcopy(m)
                m2.parts = kept
                filtered.append(m2)
        if filtered:
            clean_turns.append(filtered)

    # 5. Compute token-count per turn
    turn_tokens = [
        sum(count_tokens_for_part(p) for m in t for p in m.parts)
        for t in clean_turns
    ]

    # 6. Identify system turn and calculate its token usage
    system_turn = None
    system_turn_tokens = 0
    
    if include_system_prompts:
        # Find the first turn with system prompt parts
        for i, turn in enumerate(clean_turns):
            # First, check if this turn actually has system prompt parts
            has_system_part = any(
                isinstance(p, SystemPromptPart) 
                for m in turn 
                for p in m.parts
            )
            if has_system_part:
                system_turn = turn
                system_turn_tokens = turn_tokens[i]
                # Remove this turn from clean_turns and turn_tokens
                clean_turns = clean_turns[:i] + clean_turns[i+1:]
                turn_tokens = turn_tokens[:i] + turn_tokens[i+1:]
                break
    
    # 7. Greedily pick most-recent turns until we hit max_tokens
    remaining_tokens = max_tokens
    
    # Reduce remaining tokens if we have a system turn to include
    if system_turn is not None:
        remaining_tokens -= system_turn_tokens
        # Make sure we don't go negative
        remaining_tokens = max(0, remaining_tokens)
    
    # Select recent turns that fit in remaining token budget
    selected_turns = []
    total_tokens = 0
    
    for turn, tk in zip(reversed(clean_turns), reversed(turn_tokens)):
        if total_tokens + tk <= remaining_tokens:
            selected_turns.insert(0, turn)
            total_tokens += tk
        else:
            break
    
    # 8. Combine system turn (if any) with selected recent turns
    final_turns = []
    if system_turn is not None:
        final_turns.append(system_turn)
    final_turns.extend(selected_turns)
    
    # 9. Flatten into a single list
    trimmed = [msg for turn in final_turns for msg in turn if msg.parts]
    return trimmed
