"""Publish recorded conversations after voice/chat inactivity cutoffs."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import uuid

from app.services.telemetry_stamps import BACKGROUND_JOB, non_turn_v1_metadata

logger = logging.getLogger(__name__)
IDLE_SECONDS = {'voice': 300, 'chat': 3600}
_worker = None
_active = set()

# State and deadlines are shared by all API workers. Redis supplies the clock;
# updates and the publisher lease are atomic with respect to arriving turns.
_UPDATE = """
local now = tonumber(redis.call('time')[1])
if ARGV[1] == 'begin' then
    if ARGV[7] ~= '' and redis.call('get', KEYS[3]) ~= ARGV[7] then return 0 end
    if redis.call('exists', KEYS[1]) == 0 then
        redis.call('hset', KEYS[1], 'session_id', ARGV[3], 'environment', ARGV[4],
                   'history_complete', ARGV[6], 'surface', ARGV[9],
                   'pipeline', ARGV[10], 'persona', ARGV[11], 'idle_seconds', ARGV[12])
    end
    local seq = redis.call('hincrby', KEYS[1], 'sequence', 1)
    local turn = cjson.decode(ARGV[5])
    turn.sequence = seq
    redis.call('hset', KEYS[1], ARGV[2], cjson.encode(turn), 'latest', ARGV[2])
    redis.call('zadd', KEYS[1] .. ':active', now + 120, ARGV[2])
    redis.call('expire', KEYS[1] .. ':active', tonumber(ARGV[8]))
else
    local previous = redis.call('hget', KEYS[1], ARGV[2])
    if not previous then return 0 end
    local turn = cjson.decode(previous)
    turn.answer = ARGV[3]
    turn.finished = true
    redis.call('hset', KEYS[1], ARGV[2], cjson.encode(turn))
    redis.call('zrem', KEYS[1] .. ':active', ARGV[2])
    if redis.call('hget', KEYS[1], 'latest') ~= ARGV[2] then return 1 end
end
redis.call('hset', KEYS[1], 'last_activity', now)
redis.call('expire', KEYS[1], tonumber(ARGV[8]))
redis.call('zadd', KEYS[2], now + tonumber(redis.call('hget', KEYS[1], 'idle_seconds')), KEYS[1])
return 1
"""

_CLAIM = """
local due = tonumber(redis.call('zscore', KEYS[2], KEYS[1]))
local now = tonumber(redis.call('time')[1])
if not due or due > now then return nil end
if redis.call('exists', KEYS[1]) == 0 then
    redis.call('zrem', KEYS[2], KEYS[1]); return nil
end
local session = redis.call('hget', KEYS[1], 'session_id')
redis.call('zremrangebyscore', KEYS[1] .. ':active', '-inf', now)
if redis.call('zcard', KEYS[1] .. ':active') > 0 or
   (redis.call('hget', KEYS[1], 'surface') == 'voice' and
    redis.call('exists', ARGV[2] .. session .. '_active_request') == 1) then
    redis.call('zadd', KEYS[2], now + 30, KEYS[1]); return nil
end
if not redis.call('set', KEYS[3], ARGV[1], 'nx', 'ex', 60) then return nil end
return redis.call('hgetall', KEYS[1])
"""

_LEASE = """
local turn = redis.call('hget', KEYS[1], ARGV[1])
if turn and not cjson.decode(turn).finished then
    redis.call('zadd', KEYS[1] .. ':active', tonumber(redis.call('time')[1]) + 120, ARGV[1])
    redis.call('expire', KEYS[1] .. ':active', tonumber(ARGV[2]))
    redis.call('expire', KEYS[1], tonumber(ARGV[2]))
end
"""

_ACK = """
if redis.call('get', KEYS[3]) ~= ARGV[1] then return 0 end
if ARGV[2] ~= '' and redis.call('hget', KEYS[1], 'latest') == ARGV[2]
   and redis.call('hget', KEYS[1], 'last_activity') == ARGV[3] then
    redis.call('zrem', KEYS[2], KEYS[1])
end
redis.call('del', KEYS[3])
return 1
"""


def enabled(surface):
    from app.config import settings
    from app.observability import get_langfuse_client

    on = (settings.voice_route_enabled and settings.voice_session_outcome_enabled
          and settings.enable_voice_tracing) if surface == 'voice' else settings.chat_session_outcome_enabled
    return on and get_langfuse_client() is not None


def _store():
    from app.config import settings
    from app.core.cache import build_cache_key, redis_client

    return redis_client, build_cache_key, settings.history_cache_ttl_seconds


async def begin(session_id, query, *, surface='voice', owner=None, has_history=False,
                scope=None, pipeline='telephony', persona='farmer'):
    """Start/reset the inactivity window; failure never prevents serving a turn."""
    if not enabled(surface):
        return None
    from app.observability import surface_environment

    redis, key, ttl = _store()
    ttl = max(ttl, IDLE_SECONDS[surface] * 2)
    identity = json.dumps([surface, pipeline, scope or session_id, persona])
    state = key('session-outcome:' + hashlib.sha256(identity.encode()).hexdigest())
    turn = 'turn:' + uuid.uuid4().hex
    try:
        accepted = await asyncio.wait_for(redis.eval(
            _UPDATE, 3, state, key(f'session-outcome:{surface}:due'),
            key(session_id + '_active_request'), 'begin', turn, session_id,
            surface_environment(surface), json.dumps({'query': query, 'finished': False}),
            'false' if has_history else 'true', owner.request_token if owner else '', str(ttl),
            surface, pipeline, persona, str(IDLE_SECONDS[surface]),
        ), timeout=0.25)
        if not accepted:
            return None
        ticket = (state, turn, surface)
        _active.add(ticket)
        return ticket
    except Exception as exc:
        logger.warning('Session timer start failed (%s)', type(exc).__name__)
        return None


async def finish(ticket, answer):
    if ticket is None:
        return
    _active.discard(ticket)
    redis, key, ttl = _store()
    ttl = max(ttl, IDLE_SECONDS[ticket[2]] * 2)
    try:
        await asyncio.wait_for(redis.eval(
            _UPDATE, 3, ticket[0], key(f'session-outcome:{ticket[2]}:due'), '',
            'finish', ticket[1], answer, '', '', '', '', str(ttl)), timeout=0.25)
    except Exception as exc:
        logger.warning('Session timer finish failed (%s)', type(exc).__name__)


@asynccontextmanager
async def track(*args, **kwargs):
    ticket = await begin(*args, **kwargs)
    chunks = [] if ticket is not None else None
    try:
        yield chunks
    finally:
        await finish(ticket, ''.join(chunks or ()))


def _publish(state):
    from app.observability import get_langfuse_client, non_turn_root

    conversation = []
    turns = sorted((json.loads(v) for k, v in state.items() if k.startswith('turn:')),
                   key=lambda turn: turn['sequence'])
    complete = state['history_complete'] == 'true' and all(t['finished'] for t in turns)
    for turn in turns:
        conversation.append({'role': 'user', 'content': turn['query']})
        if turn.get('answer'):
            conversation.append({'role': 'assistant', 'content': turn['answer']})
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError('Langfuse unavailable')
    with non_turn_root(
        BACKGROUND_JOB,
        non_turn_v1_metadata(BACKGROUND_JOB, job='session_outcome',
                             pipeline=state['pipeline'], persona=state['persona']),
        session_id=state['session_id'], environment=state['environment'],
    ):
        with client.start_as_current_observation(
            name='session_outcome_ready', as_type='span',
            input={'conversation': conversation, 'history_complete': complete},
            metadata={'surface': state['surface'], 'completion_reason': 'inactivity',
                      'idle_seconds': int(state['idle_seconds'])},
        ):
            pass
    client.flush()


async def drain(surface):
    redis, key, _ = _store()
    queue = key(f'session-outcome:{surface}:due')
    now = (await redis.time())[0]
    for state_key in await redis.zrangebyscore(queue, '-inf', now, start=0, num=10):
        token = uuid.uuid4().hex
        lock = state_key + ':publishing'
        raw = await redis.eval(_CLAIM, 3, state_key, queue, lock, token, key(''))
        if not raw:
            continue
        state = dict(zip(raw[::2], raw[1::2]))
        published = False
        try:
            await asyncio.to_thread(_publish, state)
            published = True
        finally:
            await redis.eval(_ACK, 3, state_key, queue, lock, token,
                             state['latest'] if published else '', state['last_activity'])


async def _run():
    while True:
        try:
            redis, _, ttl = _store()
            # Keep live requests out of idle snapshots; leases expire if a worker dies.
            async with redis.pipeline(transaction=False) as pipe:
                for state, turn, surface in tuple(_active):
                    pipe.eval(_LEASE, 1, state, turn, str(max(ttl, IDLE_SECONDS[surface] * 2)))
                await pipe.execute()
            for surface in IDLE_SECONDS:
                if enabled(surface):
                    await drain(surface)
        except Exception as exc:
            logger.warning('Session timer check failed (%s)', type(exc).__name__)
        await asyncio.sleep(15)


async def start():
    global _worker
    if any(enabled(surface) for surface in IDLE_SECONDS) and (_worker is None or _worker.done()):
        _worker = asyncio.create_task(_run())


async def stop():
    global _worker
    if _worker is not None:
        _worker.cancel()
        try:
            await _worker
        except asyncio.CancelledError:
            pass
        finally:
            _worker = None
