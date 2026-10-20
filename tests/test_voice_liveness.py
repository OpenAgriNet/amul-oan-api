"""Voice's nudge, its side channel and the staleness check, on their own.

The nudge races the timeout against the agent's first tool call, sends at most
once through the side channel, and never after it is stopped or once the
request is stale.
"""
import asyncio
import gc
import os
import re
import shutil
import socket
import subprocess
import time
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from redis.asyncio import Redis

from app import utils
from app.channels.chat import WEB
from app.config import settings
from app.turn.types import SideChannelEmission, TelephonyCall, Turn
from app.voice import liveness as lv
from app.voice import trace as voice_trace
from app.voice.trace import VoiceTrace


def _turn(target_lang="gu"):
    return Turn(
        query="મારી ગાયને તાવ છે",
        session_id="call-1",
        source_lang="gu",
        target_lang=target_lang,
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=TelephonyCall(process_id="proc-1"),
    )


class _Sender:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send(self, emission):
        if self.fail:
            raise ConnectionError("nudge endpoint down")
        self.sent.append(emission)


def _bounded(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


@pytest.fixture
def fast_nudge(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "enable_voice_nudges", True)


# ── the messages ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pick, pool", [
    (lv.get_timeout_nudge_message, lv._TIMEOUT_NUDGE_MESSAGES),
    (lv.get_tool_nudge_message, lv._TOOL_NUDGE_MESSAGES),
])
def test_messages_come_from_the_callers_language_else_english(pick, pool):
    assert pick("gu") in pool["gu"]
    assert pick("en") in pool["en"]
    assert pick("hi") in pool["en"]


# ── the nudge ───────────────────────────────────────────────────────────────


def test_a_slow_turn_gets_one_timeout_nudge_in_the_callers_language(fast_nudge):
    sender = _Sender()

    async def _run():
        liveness = lv.VoiceLiveness(_turn(), started_at=time.monotonic(), send=sender, is_stale=None)
        await asyncio.sleep(0.1)
        await liveness.stop()

    _bounded(_run())

    (emission,) = sender.sent
    assert isinstance(emission, SideChannelEmission)
    assert emission.text in lv._TIMEOUT_NUDGE_MESSAGES["gu"]


def test_a_tool_call_sends_the_tool_nudge_at_once(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 60)
    sender = _Sender()

    async def _run():
        liveness = lv.VoiceLiveness(_turn("en"), started_at=time.monotonic(), send=sender, is_stale=None)
        lv.fire_tool_call_nudge()
        lv.fire_tool_call_nudge()
        await asyncio.sleep(0.05)
        await liveness.stop()

    _bounded(_run())

    (emission,) = sender.sent
    assert emission.text in lv._TOOL_NUDGE_MESSAGES["en"]


def test_the_deadline_counts_from_when_the_request_started(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 3.0)
    sender = _Sender()

    async def _run():
        started_long_ago = time.monotonic() - 10
        liveness = lv.VoiceLiveness(_turn(), started_at=started_long_ago, send=sender, is_stale=None)
        await asyncio.sleep(0.05)
        await liveness.stop()

    _bounded(_run())

    assert len(sender.sent) == 1


def test_stopping_first_means_no_nudge_and_nothing_left_running(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 60)
    sender = _Sender()

    async def _run():
        already_running = asyncio.all_tasks()
        liveness = lv.VoiceLiveness(_turn(), started_at=time.monotonic(), send=sender, is_stale=None)
        await asyncio.sleep(0.01)
        await liveness.stop()
        await liveness.stop()
        return [t for t in asyncio.all_tasks() - already_running if not t.done()]

    assert _bounded(_run()) == []
    assert sender.sent == []


def test_a_stale_request_is_not_nudged(fast_nudge):
    sender = _Sender()
    asked = []

    async def _stale(reason):
        asked.append(reason)
        return "stale_request"

    async def _run():
        liveness = lv.VoiceLiveness(_turn(), started_at=time.monotonic(), send=sender, is_stale=_stale)
        await asyncio.sleep(0.1)
        await liveness.stop()

    _bounded(_run())

    assert sender.sent == []
    assert asked == ["before_nudge_send"]


def test_nudges_can_be_switched_off(fast_nudge, monkeypatch):
    monkeypatch.setattr(settings, "enable_voice_nudges", False)
    sender = _Sender()

    async def _run():
        liveness = lv.VoiceLiveness(_turn(), started_at=time.monotonic(), send=sender, is_stale=None)
        lv.fire_tool_call_nudge()
        await asyncio.sleep(0.1)
        await liveness.stop()

    _bounded(_run())

    assert sender.sent == []


def test_a_failing_send_is_logged_not_raised(fast_nudge, caplog):
    sender = _Sender(fail=True)

    async def _run():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unhandled.append(context))
        liveness = lv.VoiceLiveness(_turn(), started_at=time.monotonic(), send=sender, is_stale=None)
        await asyncio.sleep(0.1)
        await liveness.stop()
        del liveness
        gc.collect()
        return unhandled

    assert _bounded(_run()) == []
    assert "Nudge task failed" in caplog.text


# ── the side channel ────────────────────────────────────────────────────────


def test_the_raya_sender_posts_the_nudge_for_this_call(monkeypatch):
    posted = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json, headers):
            posted.append((url, json))
            return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(lv.httpx, "AsyncClient", _Client)
    sender = lv.RayaNudgeSender(session_id="call-1", process_id="proc-1")

    asyncio.run(sender.send(SideChannelEmission("Please wait.")))

    assert posted == [(settings.nudge_api_url, {"message": "Please wait.", "session_id": "call-1", "process_id": "proc-1"})]


# ── the staleness check ─────────────────────────────────────────────────────


class _Request:
    def __init__(self, disconnected):
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


@pytest.fixture
def ownership(monkeypatch):
    state = {"refreshes": 0, "refresh_ok": True, "owner": True, "taken_over_by": None, "takeover_reads": 0}

    async def _refresh(owner):
        state["refreshes"] += 1
        return state["refresh_ok"]

    async def _is_owner(owner):
        return state["owner"]

    async def _taken_over_by(owner):
        state["takeover_reads"] += 1
        if isinstance(state["taken_over_by"], Exception):
            raise state["taken_over_by"]
        return state["taken_over_by"]

    monkeypatch.setattr(lv, "refresh_session_request_ownership", _refresh)
    monkeypatch.setattr(lv, "is_session_request_owner", _is_owner)
    monkeypatch.setattr(lv, "session_request_taken_over_by", _taken_over_by)
    return state


@pytest.fixture
def turn_trace():
    """The voice turn's trace the check runs in."""
    trace = VoiceTrace(session_id="call-1", user_id="", source_lang="gu", target_lang="gu", query="", enabled=False)
    token = voice_trace._current.set(trace)
    yield trace
    voice_trace._current.reset(token)


def _check(*, disconnected=False, owner=SimpleNamespace(epoch=1)):
    return lv.CallStaleness(
        http_request=_Request(disconnected), owner=owner, session_id="call-1", process_id="proc-1",
    )


def test_a_live_request_is_not_stale(ownership):
    assert asyncio.run(_check()("before_history_write")) is None


def test_a_hung_up_caller_is_stale(ownership):
    assert asyncio.run(_check(disconnected=True)("before_history_write")) == "cancelled"


def test_a_request_that_lost_the_session_is_stale(ownership):
    ownership["owner"] = False

    assert asyncio.run(_check()("before_history_write")) == "stale_request"


def test_a_failed_ownership_refresh_is_stale(ownership):
    ownership["refresh_ok"] = False

    assert asyncio.run(_check()("before_history_write")) == "stale_request"


def test_ownership_is_refreshed_at_most_once_per_interval(ownership, monkeypatch):
    monkeypatch.setattr(settings, "session_owner_refresh_interval_seconds", 60)
    check = _check()

    async def _run():
        for _ in range(3):
            assert await check("during_agent_stream") is None

    asyncio.run(_run())

    assert ownership["refreshes"] == 1


def test_without_an_owner_only_the_connection_counts(ownership):
    assert asyncio.run(_check(owner=None)("before_history_write")) is None
    assert ownership["refreshes"] == 0


# Found at the owner check, or first at the refresh (the session is another's then too).
@pytest.mark.parametrize("lost", [{"owner": False}, {"refresh_ok": False, "owner": False}])
def test_a_stale_request_tells_its_trace_which_request_took_over_once(ownership, turn_trace, lost):
    ownership.update(lost, taken_over_by="ef" * 16)
    check = _check()

    async def _run():
        first = await check("during_agent_stream")
        # Known from the check that found it, which may be the turn's last.
        known = turn_trace.superseded_by
        return [first, known, await check("before_history_write")]

    assert asyncio.run(_run()) == ["stale_request", "ef" * 16, "stale_request"]
    assert ownership["takeover_reads"] == 1


def test_a_check_that_finds_the_request_stale_while_another_reads_the_takeover_waits_for_it(ownership, turn_trace, monkeypatch):
    ownership["owner"] = False
    read = asyncio.Event()

    async def _slow(owner):
        ownership["takeover_reads"] += 1
        await read.wait()
        return "ef" * 16

    monkeypatch.setattr(lv, "session_request_taken_over_by", _slow)
    check = _check()

    async def _turns_check():
        await asyncio.sleep(0)  # the nudge's check is reading by now
        read.set()
        return await check("before_history_write"), turn_trace.superseded_by

    async def _run():
        return await asyncio.wait_for(asyncio.gather(check("before_nudge_send"), _turns_check()), timeout=5)

    assert asyncio.run(_run()) == ["stale_request", ("stale_request", "ef" * 16)]
    assert ownership["takeover_reads"] == 1


def test_a_takeover_read_cut_short_is_read_again(ownership, turn_trace, monkeypatch):
    ownership.update(owner=False, taken_over_by="ef" * 16)

    async def _run():
        check = _check()
        taken_over_by = lv.session_request_taken_over_by
        started = asyncio.Event()

        async def _hangs(owner):
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(lv, "session_request_taken_over_by", _hangs)
        nudge = asyncio.create_task(check("before_nudge_send"))
        await asyncio.wait_for(started.wait(), timeout=5)
        nudge.cancel()  # the nudge stopped as the turn ended
        with pytest.raises(asyncio.CancelledError):
            await nudge
        monkeypatch.setattr(lv, "session_request_taken_over_by", taken_over_by)
        return await check("before_history_write")

    assert asyncio.run(_run()) == "stale_request"
    assert turn_trace.superseded_by == "ef" * 16


def test_a_stale_request_whose_takeover_cant_be_read_is_still_stale(ownership, turn_trace):
    ownership["owner"] = False
    ownership["taken_over_by"] = ConnectionError("redis down")

    assert asyncio.run(_check()("before_history_write")) == "stale_request"
    assert turn_trace.superseded_by is None


def test_a_live_request_reads_no_takeover(ownership, turn_trace):
    assert asyncio.run(_check()("before_history_write")) is None
    assert asyncio.run(_check(disconnected=True)("before_history_write")) == "cancelled"
    assert ownership["takeover_reads"] == 0 and turn_trace.superseded_by is None


# ── the session's ownership, in a real Redis ────────────────────────────────


@pytest.fixture(scope="module")
def redis_port():
    """A throwaway redis-server: the claim is a Lua script, which only Redis runs."""
    if shutil.which("redis-server") is None:
        pytest.skip("needs redis-server")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        ["redis-server", "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        yield port
    finally:
        server.terminate()
        server.wait(timeout=10)


def _in_redis(port, monkeypatch, steps):
    async def _go():
        client = Redis(host="127.0.0.1", port=port, decode_responses=True)
        monkeypatch.setattr(utils, "redis_client", client)
        try:
            await client.flushall()
            return await steps(client)
        finally:
            await client.aclose()

    return asyncio.run(_go())


def test_a_request_that_comes_in_while_another_holds_the_session_takes_it_over(redis_port, monkeypatch):
    async def _steps(redis):
        first = await utils.claim_session_request_ownership("call-1")
        assert (first.epoch, first.utterance_id) == (1, first.trace_id)
        assert re.fullmatch(r"[0-9a-f]{32}", first.trace_id)
        assert await redis.get(utils._session_owner_key("call-1")) == first.request_token
        assert await utils.session_request_taken_over_by(first) is None

        second = await utils.claim_session_request_ownership("call-1")
        # The same utterance, and the first can read who took over.
        assert (second.epoch, second.utterance_id) == (2, first.trace_id)
        assert await utils.session_request_taken_over_by(first) == second.trace_id
        assert not await utils.is_session_request_owner(first) and await utils.is_session_request_owner(second)
        assert not await utils.refresh_session_request_ownership(first)
        assert not await utils.release_session_request_ownership(first)

        third = await utils.claim_session_request_ownership("call-1")
        assert third.utterance_id == first.trace_id
        assert await utils.session_request_taken_over_by(second) == third.trace_id
        # Kept past the owner's TTL: a request stuck in a long call reads it when it next checks.
        assert settings.session_owner_ttl_seconds < await redis.ttl(utils._session_takeover_key("call-1")) <= utils.DEFAULT_CACHE_TTL

        # Once the session is released, the next request is an utterance of its own.
        assert await utils.release_session_request_ownership(third)
        fourth = await utils.claim_session_request_ownership("call-1")
        assert (fourth.epoch, fourth.utterance_id) == (4, fourth.trace_id)
        assert await utils.session_request_taken_over_by(third) is None

    _in_redis(redis_port, monkeypatch, _steps)


def test_a_request_taking_over_an_older_builds_token_starts_its_own_utterance(redis_port, monkeypatch):
    async def _steps(redis):
        await redis.set(utils._session_owner_key("call-1"), "5:2b8e0c3a-1f4d-4c7e-9a3b-0d2e1f3a4b5c", ex=120)
        await redis.set(utils._session_epoch_key("call-1"), 5)
        taking_over = await utils.claim_session_request_ownership("call-1")

        assert (taking_over.epoch, taking_over.utterance_id) == (6, taking_over.trace_id)
        assert await redis.hget(utils._session_takeover_key("call-1"), "5") == taking_over.trace_id

    _in_redis(redis_port, monkeypatch, _steps)


def test_requests_claiming_at_once_leave_the_newest_owning_the_session(redis_port, monkeypatch):
    async def _steps(redis):
        owners = await asyncio.gather(*(utils.claim_session_request_ownership("call-1") for _ in range(20)))
        by_epoch = {owner.epoch: owner for owner in owners}

        assert sorted(by_epoch) == list(range(1, 21))
        assert await redis.get(utils._session_owner_key("call-1")) == by_epoch[20].request_token
        for epoch in range(1, 20):
            assert await utils.session_request_taken_over_by(by_epoch[epoch]) == by_epoch[epoch + 1].trace_id
        assert {owner.utterance_id for owner in owners} == {by_epoch[1].trace_id}

    _in_redis(redis_port, monkeypatch, _steps)
