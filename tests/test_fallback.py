"""Focused tests for the LLM core fallback state machine."""

import asyncio

import pytest

from app.llm_core import Step
from app.llm_core import execution as fb
from app.llm_core.execution import FallbackReason


class _StatusError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.status_code = code


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TimeoutError(), FallbackReason.TIMEOUT),
        (ConnectionError("refused"), FallbackReason.CONNECTION),
        (_StatusError(429), FallbackReason.RATE_LIMITED),
        (_StatusError(503), FallbackReason.HTTP_5XX),
        (_StatusError(500, "CUDA out of memory"), FallbackReason.OOM),
        (ValueError("unknown"), FallbackReason.UNKNOWN),
    ],
)
def test_classify_infrastructure_failures(error, reason):
    assert fb.classify(error) is reason


def _target(provider, *, ttft_ms=None):
    from app.llm_core.config_model import Provider, StepClientKind, Tier

    configured = Tier(
        provider=provider,
        model="test-model",
        endpoint="http://model/v1" if provider is not Provider.OPENAI else None,
        ttft_ms=ttft_ms,
    )
    return fb.ExecutionTarget(configured, StepClientKind.AGENT)


def _chain(*, ttft_ms=None):
    from app.llm_core.config_model import Provider

    return [
        _target(Provider.VLLM, ttft_ms=ttft_ms),
        _target(Provider.OPENAI),
    ]


async def _collect(stream):
    return [chunk async for chunk in stream]


def test_unary_falls_back_on_infrastructure_error(monkeypatch):
    events = []
    monkeypatch.setattr(fb, "emit", events.append)

    async def run(target):
        if target.kind == "oss":
            raise ConnectionError("primary down")
        return "managed"

    result = asyncio.run(
        fb.execute_with_fallback(
            step=Step.MODERATION,
            session_id="s1",
            run=run,
            chain=_chain(),
        )
    )
    assert result == "managed"
    assert events[0].fell_back is True


def test_unary_does_not_fallback_on_bad_output(monkeypatch):
    class UnexpectedModelBehavior(Exception):
        pass

    monkeypatch.setattr(fb, "emit", lambda event: None)

    async def run(target):
        raise UnexpectedModelBehavior("schema mismatch")

    with pytest.raises(UnexpectedModelBehavior):
        asyncio.run(
            fb.execute_with_fallback(
                step=Step.MODERATION,
                session_id="s1",
                run=run,
                chain=_chain(),
            )
        )


def test_stream_ttft_falls_back_before_commit(monkeypatch):
    events = []
    monkeypatch.setattr(fb, "emit", events.append)

    async def make_stream(target):
        if target.kind == "oss":
            await asyncio.sleep(0.05)
            yield "late"
        else:
            yield "managed"

    result = asyncio.run(
        _collect(
            fb.stream_with_fallback(
                step=Step.POST_TRANSLATION,
                session_id="s1",
                make_stream=make_stream,
                chain=_chain(ttft_ms=10),
            )
        )
    )
    assert result == ["managed"]
    assert events[0].reason is FallbackReason.TIMEOUT


def test_activity_commits_without_being_forwarded(monkeypatch):
    events = []
    calls = []
    monkeypatch.setattr(fb, "emit", events.append)

    async def make_stream(target):
        calls.append(target.kind)
        if target.kind == "oss":
            yield fb.AGENT_ACTIVITY
            await asyncio.sleep(0.02)
            raise ConnectionError("failed after tool activity")
        yield "must not rerun"

    with pytest.raises(ConnectionError):
        asyncio.run(
            _collect(
                fb.stream_with_fallback(
                    step=Step.AGENT,
                    session_id="s1",
                    make_stream=make_stream,
                    chain=_chain(ttft_ms=10),
                )
            )
        )
    assert calls == ["oss"]
    assert events[0].committed is True


def test_first_chunk_disarms_ttft(monkeypatch):
    monkeypatch.setattr(fb, "emit", lambda event: None)

    async def make_stream(target):
        yield "first"
        await asyncio.sleep(0.03)
        yield "second"

    result = asyncio.run(
        _collect(
            fb.stream_with_fallback(
                step=Step.AGENT,
                session_id="s1",
                make_stream=make_stream,
                chain=[_chain(ttft_ms=10)[0]],
            )
        )
    )
    assert result == ["first", "second"]


def test_stream_close_cancels_source_cleanly(monkeypatch):
    state = {"closed": False}
    monkeypatch.setattr(fb, "emit", lambda event: None)

    async def source(target):
        try:
            while True:
                await asyncio.sleep(0.005)
                yield "chunk"
        finally:
            state["closed"] = True

    async def drive():
        stream = fb.stream_with_fallback(
            step=Step.AGENT,
            session_id="s1",
            make_stream=source,
            chain=[_chain(ttft_ms=100)[0]],
        )
        async for _ in stream:
            await stream.aclose()
            break
        await asyncio.sleep(0.02)

    asyncio.run(drive())
    assert state["closed"] is True
