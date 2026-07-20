import asyncio

from fastapi import BackgroundTasks

from app.services import chat as chat_service


class _DummyModerationOutput:
    category = "valid_agricultural"
    action = "allow"


class _DummyModerationRun:
    output = _DummyModerationOutput()


class _DummyResponseStream:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def stream_text(self, delta: bool = True):
        for chunk in self._chunks:
            yield chunk

    def new_messages(self):
        return []


class _DummyCache:
    async def delete(self, _key: str):
        return None


def test_hindi_source_uses_pretranslation_then_hindi_output(monkeypatch):
    pretranslation_calls: list[dict] = []
    moderation_messages: list[str] = []
    agent_calls: list[dict] = []
    stream_translation_calls: list[dict] = []

    monkeypatch.setattr(chat_service.settings, "llm_core_enabled", False)
    monkeypatch.setattr(chat_service.settings, "fallback_enabled", False)
    monkeypatch.setattr(chat_service, "get_model_for_variant", lambda _variant: object())
    monkeypatch.setattr(chat_service, "provider_for_variant", lambda _variant: "openai")
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "get_langfuse_client", None)
    monkeypatch.setattr(chat_service, "cache", _DummyCache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_args, **_kwargs: "")

    async def _fake_set_cache(*_args, **_kwargs):
        return None

    async def _fake_update_message_history(*_args, **_kwargs):
        return None

    async def _fake_pretranslation(text: str, source_lang: str, provider=None, **_kwargs):
        pretranslation_calls.append(
            {
                "text": text,
                "source_lang": source_lang,
                "provider": provider,
            }
        )
        return "How much water should I give my cow?"

    async def _fake_moderation_run(user_message: str, model=None):
        moderation_messages.append(user_message)
        return _DummyModerationRun()

    def _fake_run_stream(**kwargs):
        agent_calls.append(kwargs)
        return _DummyResponseStream(
            ["Give 40 to 60 liters of clean water daily."]
        )

    async def _fake_translate_text_stream_fast(
        text: str,
        source_lang: str,
        target_lang: str,
        max_output_chars=None,
        **_kwargs,
    ):
        stream_translation_calls.append(
            {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "max_output_chars": max_output_chars,
            }
        )
        yield "अपनी गाय को रोज़ 40 से 60 लीटर साफ पानी पिलाएँ।"

    monkeypatch.setattr(chat_service, "set_cache", _fake_set_cache)
    monkeypatch.setattr(chat_service, "update_message_history", _fake_update_message_history)
    monkeypatch.setattr(chat_service, "translate_to_english_pretranslation", _fake_pretranslation)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _fake_moderation_run)
    monkeypatch.setattr(chat_service.agrinet_agent, "run_stream", _fake_run_stream)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _fake_translate_text_stream_fast)

    async def _drive():
        chunks: list[str] = []
        async for chunk in chat_service.stream_chat_messages(
            query="मुझे अपनी गाय को कितना पानी पिलाना चाहिए?",
            session_id="hindi-chat-e2e",
            source_lang="hi",
            target_lang="hi",
            channel="web",
            user_id="+919876543210",
            history=[],
            user_info={},
            background_tasks=BackgroundTasks(),
            use_translation_pipeline=True,
            pipeline_variant="legacy",
        ):
            chunks.append(chunk)
        return "".join(chunks)

    result = asyncio.run(_drive())

    assert result == "अपनी गाय को रोज़ 40 से 60 लीटर साफ पानी पिलाएँ।"
    assert pretranslation_calls == [
        {
            "text": "मुझे अपनी गाय को कितना पानी पिलाना चाहिए?",
            "source_lang": "hi",
            "provider": None,
        }
    ]
    assert moderation_messages and "How much water should I give my cow?" in moderation_messages[0]
    assert agent_calls and agent_calls[0]["deps"].query == "How much water should I give my cow?"
    assert agent_calls[0]["deps"].lang_code == "en"
    assert stream_translation_calls and stream_translation_calls[0]["source_lang"] == "english"
    assert stream_translation_calls[0]["target_lang"] == "hi"
