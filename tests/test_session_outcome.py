"""Inactivity timing and Redis scripts, without model or Langfuse requests."""
from contextlib import contextmanager
import json
import sys
import time
from types import SimpleNamespace

import pytest

fakeredis = pytest.importorskip('fakeredis')
from app.services import session_outcome as timer


@pytest.fixture
def world(monkeypatch):
    clock = [1800000000]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(timer, '_store', lambda: (redis, lambda s: 'test-' + s, 7200))
    monkeypatch.setattr(timer, 'enabled', lambda surface: True)
    monkeypatch.setitem(sys.modules, 'app.observability', SimpleNamespace(
        surface_environment=lambda surface: surface + '-production'))
    timer._active.clear()
    snapshots = []
    monkeypatch.setattr(timer, '_publish', snapshots.append)
    return redis, clock, snapshots


@pytest.mark.asyncio
@pytest.mark.parametrize('surface,idle', [('voice', 300), ('chat', 3600)])
async def test_new_turn_resets_surface_inactivity_deadline(world, surface, idle):
    redis, clock, snapshots = world
    args = {'surface': surface, 'pipeline': 'webapp' if surface == 'chat' else 'telephony'}
    first = await timer.begin('session', 'Where is the payment?', **args)
    await timer.finish(first, 'Check Payments.')
    clock[0] += idle - 1
    await timer.drain(surface)
    assert not snapshots
    # This may be served by any worker: the deadline lives in shared Redis.
    second = await timer.begin('session', 'Found it, thanks.', has_history=True, **args)
    await timer.finish(second, 'You are welcome.')
    clock[0] += idle - 1
    await timer.drain(surface)
    assert not snapshots
    clock[0] += 1
    await timer.drain(surface)
    assert snapshots[0]['session_id'] == 'session'
    turns = [json.loads(v) for k, v in snapshots[0].items() if k.startswith('turn:')]
    assert len(turns) == 2
    assert snapshots[0]['history_complete'] == 'true'
    assert 0 < await redis.ttl(first[0]) <= 7200


@pytest.mark.asyncio
async def test_same_id_stays_separate_across_surface_channel_and_persona(world):
    tickets = [await timer.begin('same-id', 'Question', **args) for args in [
        {}, {'surface': 'chat', 'pipeline': 'webapp'},
        {'surface': 'chat', 'pipeline': 'whatsapp'},
        {'surface': 'chat', 'pipeline': 'webapp', 'persona': 'doctor'},
    ]]
    assert len({ticket[0] for ticket in tickets}) == 4


@pytest.mark.asyncio
async def test_chat_transcript_outlives_one_hour_even_with_short_history_ttl(world, monkeypatch):
    redis, _, _ = world
    monkeypatch.setattr(timer, '_store', lambda: (redis, lambda s: 'test-' + s, 600))
    ticket = await timer.begin('session', 'Question', surface='chat', pipeline='webapp')
    await timer.finish(ticket, 'Answer')
    assert await redis.ttl(ticket[0]) >= 7200


@pytest.mark.asyncio
async def test_live_chat_request_lease_postpones_cutoff(world):
    redis, clock, snapshots = world
    ticket = await timer.begin('session', 'Question', surface='chat', pipeline='webapp')
    clock[0] += 3599
    await redis.eval(timer._LEASE, 1, ticket[0], ticket[1], '7200')
    clock[0] += 1
    await timer.drain('chat')
    assert not snapshots


@pytest.mark.asyncio
async def test_active_request_postpones_idle_snapshot(world):
    redis, clock, snapshots = world
    await redis.set('test-session_active_request', 'owner')
    ticket = await timer.begin('session', 'Help', owner=SimpleNamespace(request_token='owner'))
    await timer.finish(ticket, 'Answer')
    clock[0] += 300
    await timer.drain('voice')
    assert not snapshots
    await redis.delete('test-session_active_request')
    clock[0] += 30
    await timer.drain('voice')
    assert snapshots


@pytest.mark.asyncio
async def test_expired_request_is_unanswered_not_a_fabricated_reply(world):
    _, clock, snapshots = world
    await timer.begin('session', 'Please answer this.', has_history=True)
    # A worker can restart before finish; the pending record survives in Redis.
    clock[0] += 300
    await timer.drain('voice')
    state = snapshots[0]
    turn = next(json.loads(v) for k, v in state.items() if k.startswith('turn:'))
    assert turn['query'] == 'Please answer this.'
    assert not turn['finished'] and 'answer' not in turn
    assert state['history_complete'] == 'false'


@pytest.mark.asyncio
async def test_old_owner_cannot_start_tracking_after_takeover(world):
    redis, _, _ = world
    await redis.set('test-session_active_request', 'new-owner')
    ticket = await timer.begin('session', 'Old turn', owner=SimpleNamespace(request_token='old-owner'))
    assert ticket is None


@pytest.mark.asyncio
async def test_older_turn_finishing_does_not_reset_latest_turns_deadline(world):
    redis, clock, snapshots = world
    first = await timer.begin('session', 'First question')
    clock[0] += 10
    latest = await timer.begin('session', 'Latest question', has_history=True)
    await timer.finish(latest, 'Latest reply')
    deadline = await redis.zscore('test-session-outcome:voice:due', latest[0])
    clock[0] += 100
    await timer.finish(first, 'Earlier reply')
    assert await redis.zscore('test-session-outcome:voice:due', latest[0]) == deadline
    assert not snapshots


@pytest.mark.asyncio
async def test_new_activity_during_publication_keeps_next_deadline(world, monkeypatch):
    redis, clock, _ = world
    ticket = await timer.begin('session', 'First question')
    await timer.finish(ticket, 'First answer')
    clock[0] += 300

    async def publish_then_resume(function, state):
        function(state)
        newer = await timer.begin('session', 'One more thing', has_history=True)
        await timer.finish(newer, 'Another answer')

    monkeypatch.setattr(timer.asyncio, 'to_thread', publish_then_resume)
    await timer.drain('voice')
    assert await redis.zscore('test-session-outcome:voice:due', ticket[0]) == clock[0] + 300


@pytest.mark.asyncio
async def test_redis_failure_does_not_fail_the_voice_turn(world, monkeypatch, caplog):
    redis, _, _ = world

    async def fail(*_):
        raise ConnectionError('must not log caller text')

    monkeypatch.setattr(redis, 'eval', fail)
    assert await timer.begin('session', 'private caller text') is None
    await timer.finish(('state', 'turn', 'voice'), 'private answer')
    assert 'private' not in caplog.text and 'must not log' not in caplog.text


@pytest.mark.asyncio
async def test_disabled_feature_does_not_touch_redis(world, monkeypatch):
    monkeypatch.setattr(timer, 'enabled', lambda surface: False)
    monkeypatch.setattr(timer, '_store', lambda: pytest.fail('disabled feature touched Redis'))
    assert await timer.begin('session', 'Hello') is None
    await timer.finish(None, '')


def test_snapshot_contains_only_caller_and_emitted_answer_with_session_link(monkeypatch):
    observations = []
    roots = []

    @contextmanager
    def observation(**kwargs):
        observations.append(kwargs)
        yield

    @contextmanager
    def root(*args, **kwargs):
        roots.append((args, kwargs))
        yield

    client = SimpleNamespace(start_as_current_observation=observation, flush=lambda: None)
    monkeypatch.setitem(sys.modules, 'app.observability', SimpleNamespace(
        get_langfuse_client=lambda: client, non_turn_root=root))
    timer._publish({
        'session_id': 'session', 'environment': 'voice-production', 'history_complete': 'true',
        'surface': 'voice', 'pipeline': 'telephony', 'persona': 'farmer', 'idle_seconds': '300',
        'turn:b': json.dumps({'sequence': 2, 'query': 'And this?', 'finished': False}),
        'turn:a': json.dumps({'sequence': 1, 'query': 'Where?', 'answer': 'Over there.', 'finished': True}),
    })
    assert roots[0][1] == {'session_id': 'session', 'environment': 'voice-production'}
    assert observations[0]['name'] == 'session_outcome_ready'
    assert observations[0]['input'] == {'history_complete': False, 'conversation': [
        {'role': 'user', 'content': 'Where?'}, {'role': 'assistant', 'content': 'Over there.'},
        {'role': 'user', 'content': 'And this?'},
    ]}
