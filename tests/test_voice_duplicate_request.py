"""The words just answered, sent again by Raya (#582): the caller gets the same
answer, the agent does not run again, and only an answer the agent gave is kept.

The store's Redis is a dict here; the route is test_voice_route's, with its
stand-ins for the rest.
"""
import asyncio
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.voice import agent as voice_agents
from app.services import voice as voice_service
from app.turn.types import TelephonyCall
from app.voice import classifiers as vc
from app.voice import duplicates
from app.voice import liveness as lv
from tests.test_voice_classifiers import _turn
from tests.test_voice_route import _get, _outcomes, world  # noqa: F401  (world is a fixture)

_WORDS = "ગાય દૂધ ઓછું આપે છે"
_ANSWER = "પાણી આપો."


class _Redis:
    """The two calls the store makes, kept in a dict."""

    def __init__(self):
        self.values, self.ttls = {}, {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key], self.ttls[key] = value, ex


@pytest.fixture
def kept(monkeypatch):
    redis = _Redis()
    monkeypatch.setattr(duplicates, "redis_client", redis)
    return redis


@pytest.fixture(autouse=True)
def no_heat_alert(monkeypatch):
    """No call here is one of our heat-alert calls (that lookup reads Redis)."""
    async def _none(*_a):
        return None

    monkeypatch.setattr(voice_service.heat_alert_call, "context_for_call", _none)


def _remember(heard=(_ANSWER,), query=_WORDS, at=100.0, session="call-1"):
    asyncio.run(duplicates.remember_answer(session, query, list(heard), at))


def _repeat(query=_WORDS, at=101.0, session="call-1"):
    return asyncio.run(duplicates.answer_to_repeat(session, query, at))


# ── the store ───────────────────────────────────────────────────────────────


def test_the_same_words_within_five_seconds_get_the_answer_again(kept):
    _remember()

    assert _repeat(at=101.0) == _ANSWER
    assert _repeat(at=105.0) == _ANSWER
    assert _repeat(at=105.1) is None


def test_punctuation_and_spacing_are_not_new_words(kept):
    _remember()

    assert _repeat(query=f" {_WORDS.replace(' ', '  ')}? ") == _ANSWER


def test_other_words_or_another_call_are_not_a_repeat(kept):
    _remember()

    assert _repeat(query="ભેંસ ખાતી નથી") is None
    assert _repeat(session="call-2") is None
    assert _repeat(query=" ? ") is None


def test_the_answer_is_kept_as_heard_a_little_longer_than_the_window(kept):
    _remember(heard=("પાણી ", "આપો."))

    (value,) = kept.values.values()
    assert json.loads(value) == {"words": _WORDS, "answer": _ANSWER, "at": 100.0}
    assert list(kept.ttls.values()) == [6]


def test_an_answer_that_hung_up_or_said_nothing_is_not_kept(kept):
    _remember(heard=("આવજો.", " Goodbye."))
    _remember(heard=())
    _remember(heard=("  ",))
    _remember(query=" ? ")

    assert kept.values == {}


def test_a_store_that_fails_holds_up_no_turn(monkeypatch):
    class _Down:
        async def get(self, _key):
            raise ConnectionError("down")

        async def set(self, *_a, **_kw):
            raise ConnectionError("down")

    monkeypatch.setattr(duplicates, "redis_client", _Down())

    _remember()
    assert _repeat() is None


# ── the classifier ──────────────────────────────────────────────────────────


def test_the_repeat_is_said_as_heard_and_adds_nothing_to_history():
    call = TelephonyCall(process_id="proc-2", repeated_answer=_ANSWER)

    answer = asyncio.run(vc._duplicate_request_classifier(_turn(_WORDS, call=call)))

    assert (answer.canned_text, answer.label, answer.outcome) == (_ANSWER, "duplicate_request", "duplicate_request")
    assert answer.raw and answer.history_pair is None
    assert asyncio.run(vc._duplicate_request_classifier(_turn(_WORDS))) is None


# ── through the route ───────────────────────────────────────────────────────


def test_the_words_sent_again_get_the_same_answer_without_the_agent(world, kept, monkeypatch):  # noqa: F811
    outcomes = _outcomes(monkeypatch)
    runs = []
    run = voice_agents.voice_agent.iter
    monkeypatch.setattr(voice_agents.voice_agent, "iter", lambda **kw: runs.append(kw) or run(**kw))

    first = _get(query="My cow has fever", source_lang="en", target_lang="en")
    again = _get(query="My cow has fever.", source_lang="en", target_lang="en")

    assert first.text == again.text == "Give her water."
    assert len(runs) == 1 and len(world["history_writes"]) == 1
    assert outcomes == ["success", "duplicate_request"]

    _get(query="My buffalo is not eating", source_lang="en", target_lang="en")
    assert len(runs) == 2 and outcomes[-1] == "success"


def test_only_an_answer_the_agent_gave_is_kept(world, kept):  # noqa: F811
    _get(query="hello", source_lang="en", target_lang="en")

    assert kept.values == {}


def test_a_request_another_took_over_keeps_no_answer(world, kept, monkeypatch):  # noqa: F811
    async def _not_owner(_owner):
        return False

    monkeypatch.setattr(lv, "is_session_request_owner", _not_owner)
    outcomes = _outcomes(monkeypatch)

    _get(query="My cow has fever", source_lang="en", target_lang="en")

    assert outcomes == ["stale_request"] and kept.values == {}
