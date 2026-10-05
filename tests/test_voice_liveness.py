"""Voice's nudge, its side channel and the staleness check, on their own.

The nudge races the timeout against the agent's first tool call, sends at most
once through the side channel, and never after it is stopped or once the
request is stale.
"""
import asyncio
import gc
import os
import time
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.chat import WEB
from app.config import settings
from app.turn.types import SideChannelEmission, TelephonyCall, Turn
from app.voice import liveness as lv


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
    state = {"refreshes": 0, "refresh_ok": True, "owner": True}

    async def _refresh(owner):
        state["refreshes"] += 1
        return state["refresh_ok"]

    async def _is_owner(owner):
        return state["owner"]

    monkeypatch.setattr(lv, "refresh_session_request_ownership", _refresh)
    monkeypatch.setattr(lv, "is_session_request_owner", _is_owner)
    return state


def _check(*, disconnected=False, owner=SimpleNamespace(epoch=1)):
    return lv.CallStaleness(
        http_request=_Request(disconnected), owner=owner, session_id="call-1", process_id="proc-1",
    )


def test_a_live_request_is_not_stale(ownership):
    assert asyncio.run(_check()("before_history_write")) is None


def test_a_hung_up_caller_is_stale(ownership):
    assert asyncio.run(_check(disconnected=True)("before_history_write")) == "client_disconnected"


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
