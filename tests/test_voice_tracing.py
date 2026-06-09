import asyncio
import logging

from app.services.voice_trace import create_voice_trace, sanitize_text


def test_sanitize_text_preview_hash(monkeypatch):
    monkeypatch.setattr("app.services.voice_trace.settings.voice_trace_text_mode", "preview_hash", raising=False)
    monkeypatch.setattr("app.services.voice_trace.settings.voice_trace_preview_chars", 5, raising=False)

    payload = sanitize_text("hello farmer")

    assert payload["chars"] == 12
    assert payload["preview"] == "hello"
    assert payload["sha256"]
    assert "text" not in payload


def test_sanitize_text_full(monkeypatch):
    monkeypatch.setattr("app.services.voice_trace.settings.voice_trace_text_mode", "full", raising=False)

    payload = sanitize_text("hello farmer")

    assert payload["text"] == "hello farmer"
    assert payload["sha256"]


def test_sanitize_text_none(monkeypatch):
    monkeypatch.setattr("app.services.voice_trace.settings.voice_trace_text_mode", "none", raising=False)

    payload = sanitize_text("hello farmer")

    assert payload["chars"] == 12
    assert payload["sha256"]
    assert "preview" not in payload
    assert "text" not in payload


def test_stage_and_finish_emit_summary(caplog, monkeypatch):
    monkeypatch.setattr("app.services.voice_trace.settings.enable_voice_tracing", True, raising=False)
    monkeypatch.setattr("app.services.voice_trace.settings.voice_trace_log_summary", True, raising=False)
    caplog.set_level(logging.INFO, logger="app.services.voice_trace")

    trace = create_voice_trace(
        session_id="trace-test",
        user_id="user-1",
        query="hello",
        source_lang="en",
        target_lang="en",
        provider=None,
        process_id="proc-1",
    )

    with trace.stage("unit_stage", metadata={"example": "yes"}):
        pass
    trace.record_emit("first answer")
    trace.finish("success")
    trace.finish("ignored")

    assert trace.stage_totals_ms["unit_stage"] >= 0
    assert trace.timings_ms["ttft_ms"] >= 0
    assert trace.timings_ms["ttfr_ms"] >= 0
    assert trace.outcome == "success"
    assert sum("VOICE_TRACE_SUMMARY" in r.message for r in caplog.records) == 1


def test_greeting_stream_records_trace(monkeypatch):
    from app.services import voice as voice_module

    async def _update_message_history(*args, **kwargs):
        return None

    async def _render_text_for_caller(text_en, target_lang):
        return text_en

    monkeypatch.setattr(voice_module, "update_message_history", _update_message_history)
    monkeypatch.setattr(voice_module, "_render_text_for_caller", _render_text_for_caller)
    monkeypatch.setattr(voice_module.settings, "voice_trace_log_summary", False, raising=False)

    trace = create_voice_trace(
        session_id="greeting-trace",
        user_id="anonymous",
        query="hello",
        source_lang="en",
        target_lang="en",
        provider=None,
        process_id="proc-1",
    )

    async def _collect():
        chunks = []
        async for chunk in voice_module.stream_voice_message(
            query="hello",
            session_id="greeting-trace",
            source_lang="en",
            target_lang="en",
            user_id="anonymous",
            history=[],
            provider=None,
            process_id="proc-1",
            user_info={},
            owner=None,
            http_request=None,
            trace=trace,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    output = asyncio.run(_collect())

    assert "Hello, I am Sarlaben" in output
    assert trace.route == "greeting_fast_path"
    assert trace.outcome == "greeting_fast_path"
    assert "ttft_ms" in trace.timings_ms
    assert "ttfr_ms" in trace.timings_ms
