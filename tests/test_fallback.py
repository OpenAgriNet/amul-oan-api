"""Focused tests for the LLM core fallback state machine."""

import asyncio

import pytest

from app.llm_core.config_model import Step
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


def test_disabled_context_calls_primary_without_policy_walker(monkeypatch):
    from app.llm_core.config_model import NamedProfile, PipelineConfig, Provider, StepConfig, Tier

    cfg = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps={
        Step.MODERATION: StepConfig(tiers=[
            Tier(provider=Provider.OPENAI, model="gpt-4.1")
        ])
    })])
    execution = fb.ExecutionContext("s1", cfg, "managed")
    monkeypatch.setattr(
        fb,
        "execute_with_fallback",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("policy walker called")),
    )

    async def run():
        return await execution.run_adapter(Step.MODERATION, lambda target: _answer(target))

    async def _answer(target):
        return target.model_name

    assert asyncio.run(run()) == "gpt-4.1"


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
    state = {"closed": False, "produced": 0}
    monkeypatch.setattr(fb, "emit", lambda event: None)

    async def source(target):
        try:
            value = 0
            while True:
                value += 1
                state["produced"] = value
                yield f"chunk-{value}"
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
            # Let the producer fill the one-item queue and block on its next put.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await stream.aclose()
            break
        await asyncio.sleep(0.02)

    asyncio.run(drive())
    assert state["produced"] >= 3
    assert state["closed"] is True
