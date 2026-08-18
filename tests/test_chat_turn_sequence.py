"""Pins the ORDER of stages in a chat turn.

This is the invariant `run_turn` has to reproduce when the channel seam lands.
The two orchestrators sequenced their stages slightly differently, so recording
chat's order before the rewrite is what makes "behaviour-preserving" checkable
rather than asserted. A change here is a real behaviour change and needs to be
argued for, not absorbed.
"""
import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import BackgroundTasks

from app.services import chat as chat_service


class _Cache:
    async def delete(self, _key):
        return None

    async def get(self, _key):
        return None

    async def set(self, *_a, **_kw):
        return None


_PartDeltaEvent = type("PartDeltaEvent", (), {})
_TextPartDelta = type("TextPartDelta", (), {})


def _delta(chunk):
    d = _TextPartDelta()
    d.content_delta = chunk
    e = _PartDeltaEvent()
    e.delta = d
    return e


class _Node:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, _ctx):
        chunks = self._chunks

        class _S:
            async def __aenter__(s):
                return s

            async def __aexit__(s, *_e):
                return False

            async def __aiter__(s):
                for c in chunks:
                    yield _delta(c)

        return _S()


_Node.__name__ = "ModelRequestNode"


class _Run:
    def __init__(self, chunks):
        self._chunks = chunks
        self.ctx = object()
        self.result = SimpleNamespace(new_messages=lambda: [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_e):
        return False

    async def __aiter__(self):
        yield _Node(self._chunks)


def _drive(monkeypatch, *, source_lang="gu", target_lang="gu",
           fallback_enabled=False, moderation_action="allow",
           moderation_category="valid_agricultural", user_info=None,
           requested_persona=None):
    """Run one turn with every stage instrumented, and return the stage order."""
    seen: list[str] = []

    def record(name):
        def _mark(*_a, **_kw):
            seen.append(name)
        return _mark

    monkeypatch.setattr(chat_service.settings, "fallback_enabled", fallback_enabled)
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "get_langfuse_client", None)
    monkeypatch.setattr(chat_service, "cache", _Cache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_a, **_kw: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_a, **_kw: "")

    async def _pretranslate(text, *_a, **_kw):
        seen.append("pretranslation")
        return "How much water should I give my cow?"

    async def _moderate(user_message, model=None):
        seen.append("moderation")
        return SimpleNamespace(
            output=SimpleNamespace(category=moderation_category, action=moderation_action)
        )

    def _agent_iter(**_kw):
        seen.append("agent")
        return _Run(["Give clean water daily."])

    def _doctor_agent_iter(**_kw):
        seen.append("doctor_agent")
        assert _kw["deps"].persona == "doctor"
        assert _kw["deps"].farmer_info == ""
        assert _kw["deps"].mobile is None
        return _Run(["Administer only the retrieved protocol."])

    async def _farmer_context(_phone):
        seen.append("farmer_context")
        return "farmer data", [], {}

    async def _translate_stream(text, *_a, **_kw):
        seen.append("output_translation")
        yield "રોજ સ્વચ્છ પાણી આપો."

    async def _history(*_a, **_kw):
        seen.append("history_persist")

    async def _set_cache(*_a, **_kw):
        return None

    monkeypatch.setattr(chat_service, "translate_to_english_pretranslation", _pretranslate)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderate)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _agent_iter)
    monkeypatch.setattr(chat_service.doctor_agent, "iter", _doctor_agent_iter)
    monkeypatch.setattr(chat_service, "get_farmer_context_bundle_by_mobile", _farmer_context)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _translate_stream)
    monkeypatch.setattr(chat_service, "update_message_history", _history)
    monkeypatch.setattr(chat_service, "set_cache", _set_cache)
    monkeypatch.setattr(chat_service, "create_suggestions", record("suggestions"))

    async def _go():
        out = []
        async for chunk in chat_service.stream_chat_messages(
            query="મારી ગાયને કેટલું પાણી આપવું?",
            session_id="turn-sequence",
            source_lang=source_lang,
            target_lang=target_lang,
            channel="web",
            user_id="+919876543210",
            history=[],
            user_info=user_info or {},
            background_tasks=BackgroundTasks(),
            use_translation_pipeline=True,
            pipeline_profile="managed",
            requested_persona=requested_persona,
        ):
            out.append(chunk)
        return "".join(out)

    return asyncio.run(_go()), seen


_TRACKED = ("pretranslation", "moderation", "agent", "output_translation")


@pytest.mark.parametrize("fallback_enabled", [False, True])
def test_gujarati_turn_stage_order(monkeypatch, fallback_enabled):
    output, stages = _drive(monkeypatch, fallback_enabled=fallback_enabled)

    assert output, "the turn produced no output"
    # EXACT sequence, not a subsequence: a stage running twice is a duplicated
    # model call, which a first-index check cannot see.
    assert [s for s in stages if s in _TRACKED] == list(_TRACKED), f"stage order changed: {stages}"


def test_moderation_runs_after_pretranslation_not_before(monkeypatch):
    """Moderation sees the ENGLISH text, so it must follow pretranslation."""
    _, stages = _drive(monkeypatch)
    assert stages.index("pretranslation") < stages.index("moderation")


def test_agent_runs_after_moderation_allows(monkeypatch):
    _, stages = _drive(monkeypatch)
    assert stages.index("moderation") < stages.index("agent")


def test_english_turn_skips_translation_stages(monkeypatch):
    """en->en needs neither pretranslation nor output translation."""
    _, stages = _drive(monkeypatch, source_lang="en", target_lang="en")
    assert "pretranslation" not in stages
    assert "output_translation" not in stages
    assert "agent" in stages


@pytest.mark.parametrize("fallback_enabled", [False, True])
def test_blocked_moderation_never_reaches_the_agent(monkeypatch, fallback_enabled):
    """The hard gate. A blocked query must decline without running the agent.

    Without this case the lock pins only the happy path, and deleting the gate at
    the yield-decline/return in stream_chat_messages leaves the suite green.
    """
    output, stages = _drive(
        monkeypatch,
        fallback_enabled=fallback_enabled,
        moderation_action="block",
        moderation_category="non_agricultural",
    )

    assert "moderation" in stages
    assert "agent" not in stages, f"blocked query reached the agent: {stages}"
    assert output, "a blocked turn must still say something to the farmer"


def test_doctor_jwt_routes_to_doctor_agent_without_farmer_context(monkeypatch):
    output, stages = _drive(
        monkeypatch,
        source_lang="en",
        target_lang="en",
        user_info={
            "phone": "9375028676",
            "user_type": "doctor",
            "auth_type": "jwt",
        },
    )

    assert output
    assert "doctor_agent" in stages
    assert "agent" not in stages
    assert "farmer_context" not in stages
