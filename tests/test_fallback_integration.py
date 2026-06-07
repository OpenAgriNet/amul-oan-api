"""End-to-end fault-injection: with FALLBACK_ENABLED and the OSS endpoint DEAD,
a real agent run must fall back to the managed model and complete.

SKIPPED unless RUN_FALLBACK_INTEGRATION=1 and a real OPENAI_API_KEY is set — it
makes real managed-model calls and needs network (run in staging/CI, not locally).

    RUN_FALLBACK_INTEGRATION=1 OPENAI_API_KEY=sk-... \\
        .venv/bin/python -m pytest tests/test_fallback_integration.py -o asyncio_mode=auto -s
"""

import os

import asyncio

import pytest

_KEY = os.getenv("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_FALLBACK_INTEGRATION") != "1" or not _KEY or _KEY == "test-key",
    reason="set RUN_FALLBACK_INTEGRATION=1 and a real OPENAI_API_KEY (needs network + managed model)",
)

DEAD_OSS_URL = "http://127.0.0.1:1/v1"  # port 1 -> connection refused


def _dead_oss_model():
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIModel(
        "gemma-dead",
        provider=OpenAIProvider(base_url=DEAD_OSS_URL, api_key="x"),
    )


@pytest.fixture
def oss_dead(monkeypatch):
    from app.services import fallback as fb

    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", _dead_oss_model())
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", DEAD_OSS_URL)
    events = []
    monkeypatch.setattr(fb, "emit", events.append)
    return fb, events


def test_unary_moderation_falls_back_to_managed(oss_dead):
    fb, events = oss_dead
    from agents.moderation import moderation_agent

    result = asyncio.run(
        fb.execute_with_fallback(
            pipeline="moderation",
            session_id="it-moderation",
            variant="oss",
            run=lambda a: moderation_agent.run("My cow has a fever, what should I do?", model=a.model),
        )
    )
    assert result is not None and result.output is not None  # managed produced a verdict
    assert any(e.fell_back for e in events), "expected an OSS->managed fallback event"


def test_streaming_chat_falls_back_to_managed(oss_dead):
    fb, events = oss_dead
    from agents.agrinet import agrinet_agent
    from agents.deps import FarmerContext

    deps = FarmerContext(query="Reply with a short greeting.", session_id="it-chat", farmer_info="No farmer context.")

    async def make_stream(attempt):
        async with agrinet_agent.run_stream(
            user_prompt="Reply with a short greeting.",
            message_history=[],
            deps=deps,
            model=attempt.model,
        ) as rs:
            async for c in rs.stream_text(delta=True):
                yield c

    async def drive():
        chunks = []
        async for c in fb.stream_with_fallback(
            pipeline="chat", session_id="it-chat", variant="oss", make_stream=make_stream
        ):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(drive())
    assert "".join(chunks).strip(), "expected streamed tokens from the managed model"
    assert any(e.fell_back and not e.committed for e in events), "expected a pre-first-token OSS->managed swap"
