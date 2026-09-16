"""Bengali chat uses the translation pipeline like Hindi: a bn question is
pretranslated to English for moderation and the agent, and the English answer is
translated back to Bengali. BENGALI_CHAT_ENABLED=false bypasses both gates."""
import asyncio
from types import SimpleNamespace

from fastapi import BackgroundTasks

from app.services import chat as chat_service

BENGALI_QUERY = "আমার গরুকে কতটা জল খাওয়ানো উচিত?"
ENGLISH_QUERY = "How much water should I give my cow?"
ENGLISH_ANSWER = "Give 40 to 60 liters of clean water daily."
BENGALI_ANSWER = "আপনার গরুকে প্রতিদিন 40 থেকে 60 লিটার পরিষ্কার জল খাওয়ান।"


class _DummyModerationOutput:
    category = "valid_agricultural"
    action = "allow"


class _DummyModerationRun:
    output = _DummyModerationOutput()


class _DummyCache:
    async def delete(self, _key: str):
        return None


# chat.py dispatches on type(x).__name__, so these only need matching class names.
_PartDeltaEvent = type("PartDeltaEvent", (), {})
_TextPartDelta = type("TextPartDelta", (), {})


def _text_delta_event(chunk: str):
    delta = _TextPartDelta()
    delta.content_delta = chunk
    event = _PartDeltaEvent()
    event.delta = delta
    return event


class _FakeModelRequestNode:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, _ctx):
        chunks = self._chunks

        class _EventStream:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *_exc):
                return False

            async def __aiter__(self_inner):
                for chunk in chunks:
                    yield _text_delta_event(chunk)

        return _EventStream()


_FakeModelRequestNode.__name__ = "ModelRequestNode"


class _FakeAgentRun:
    def __init__(self, chunks):
        self._chunks = chunks
        self.ctx = object()
        self.result = SimpleNamespace(new_messages=lambda: [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def __aiter__(self):
        yield _FakeModelRequestNode(self._chunks)


def _install_fakes(monkeypatch) -> dict[str, list]:
    calls: dict[str, list] = {"pretranslation": [], "moderation": [], "agent": [], "translation": []}

    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "get_langfuse_client", None)
    monkeypatch.setattr(chat_service, "cache", _DummyCache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_args, **_kwargs: "")

    async def _fake_set_cache(*_args, **_kwargs):
        return None

    async def _fake_update_message_history(*_args, **_kwargs):
        return None

    async def _fake_pretranslation(_tier, *, text: str, source_lang: str, **_kwargs):
        calls["pretranslation"].append({"text": text, "source_lang": source_lang})
        return ENGLISH_QUERY

    async def _fake_moderation_run(user_message: str, model=None):
        calls["moderation"].append(user_message)
        return _DummyModerationRun()

    async def _fake_translate_text_stream_fast(
        text: str,
        source_lang: str,
        target_lang: str,
        max_output_chars=None,
        **_kwargs,
    ):
        calls["translation"].append(
            {"text": text, "source_lang": source_lang, "target_lang": target_lang}
        )
        yield BENGALI_ANSWER

    def _fake_iter(**kwargs):
        calls["agent"].append(kwargs)
        return _FakeAgentRun([ENGLISH_ANSWER])

    monkeypatch.setattr(chat_service, "set_cache", _fake_set_cache)
    monkeypatch.setattr(chat_service, "update_message_history", _fake_update_message_history)
    monkeypatch.setattr(chat_service, "pretranslate_with_tier", _fake_pretranslation)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _fake_moderation_run)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _fake_iter)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _fake_translate_text_stream_fast)
    return calls


def _drive(query: str, lang: str) -> str:
    async def _run() -> str:
        chunks: list[str] = []
        async for chunk in chat_service.stream_chat_messages(
            query=query,
            session_id="bengali-chat-e2e",
            source_lang=lang,
            target_lang=lang,
            channel="web",
            user_id="+919876543210",
            history=[],
            user_info={},
            background_tasks=BackgroundTasks(),
            use_translation_pipeline=True,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    return asyncio.run(_run())


def test_bengali_source_uses_pretranslation_then_bengali_output(monkeypatch):
    calls = _install_fakes(monkeypatch)

    result = _drive(BENGALI_QUERY, "bn")

    assert result == BENGALI_ANSWER
    assert calls["pretranslation"] == [{"text": BENGALI_QUERY, "source_lang": "bn"}]
    assert calls["moderation"] and ENGLISH_QUERY in calls["moderation"][0]
    assert calls["agent"] and calls["agent"][0]["deps"].query == ENGLISH_QUERY
    assert calls["agent"][0]["deps"].lang_code == "en"
    assert calls["translation"] and calls["translation"][0]["source_lang"] == "english"
    assert calls["translation"][0]["target_lang"] == "bn"


def test_bengali_kill_switch_bypasses_translation_pipeline(monkeypatch):
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(chat_service.settings, "bengali_chat_enabled", False)

    result = _drive(BENGALI_QUERY, "bn")

    assert calls["pretranslation"] == []
    assert calls["translation"] == []
    assert calls["agent"] and calls["agent"][0]["deps"].query == BENGALI_QUERY
    assert result == ENGLISH_ANSWER
