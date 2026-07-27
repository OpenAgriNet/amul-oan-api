"""Unit tests for the standard OSS -> managed fallback module.

Sets a dummy OPENAI_API_KEY before importing app code, because agents.models
constructs the LLM model eagerly at import.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio

import pytest

from app.services import fallback as fb
from app.services.fallback import FallbackReason, classify, execute_with_fallback


# ── classify() ──────────────────────────────────────────────────────────────

class _StatusError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg)
        self.status_code = code


def test_classify_timeout():
    assert classify(asyncio.TimeoutError()) is FallbackReason.TIMEOUT
    assert classify(TimeoutError()) is FallbackReason.TIMEOUT


def test_classify_connection():
    assert classify(ConnectionError("connection refused")) is FallbackReason.CONNECTION


def test_classify_http_status():
    assert classify(_StatusError(429)) is FallbackReason.RATE_LIMITED
    assert classify(_StatusError(503)) is FallbackReason.HTTP_5XX
    assert classify(_StatusError(500, "CUDA out of memory")) is FallbackReason.OOM


def test_classify_bad_output():
    class UnexpectedModelBehavior(Exception):
        pass

    assert classify(UnexpectedModelBehavior("schema mismatch")) is FallbackReason.BAD_OUTPUT


def test_classify_cancelled():
    assert classify(asyncio.CancelledError()) is FallbackReason.CANCELLED


def test_classify_unknown():
    assert classify(ValueError("???")) is FallbackReason.UNKNOWN


def test_bad_output_and_cancelled_not_fallbackable():
    assert FallbackReason.BAD_OUTPUT not in fb.FALLBACKABLE
    assert FallbackReason.CANCELLED not in fb.FALLBACKABLE
    assert FallbackReason.TIMEOUT in fb.FALLBACKABLE


# ── execute_with_fallback() over the config-driven chain ─────────────────────
# The chain now comes from the weighted-profile resolver (tested in test_split);
# these walker tests stub ``_resolve_chain`` with a controlled variant-keyed chain
# (via the shared ``install_chain`` fixture) so they isolate the classify /
# first-token-commit logic. variant 'oss' -> [oss, managed]; else -> [managed].

@pytest.fixture
def oss_enabled(monkeypatch, install_chain):
    """Fallback ON, controlled [oss, managed] chain; capture emitted events."""
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    install_chain()
    events = []
    monkeypatch.setattr(fb, "emit", events.append)
    return events


def test_falls_back_to_managed_on_connection_error(oss_enabled):
    async def run(attempt):
        if attempt.kind == "oss":
            raise ConnectionError("connection refused")
        return "managed-result"

    result = asyncio.run(
        execute_with_fallback(pipeline="moderation", session_id="s1", profile_name="oss", run=run)
    )
    assert result == "managed-result"
    assert len(oss_enabled) == 1
    ev = oss_enabled[0]
    assert ev.fell_back is True
    assert ev.reason is FallbackReason.CONNECTION
    assert ev.from_variant == "oss" and ev.to_variant == "managed"


def test_does_not_fall_back_on_bad_output(oss_enabled):
    class UnexpectedModelBehavior(Exception):
        pass

    async def run(attempt):
        raise UnexpectedModelBehavior("schema mismatch")

    with pytest.raises(UnexpectedModelBehavior):
        asyncio.run(
            execute_with_fallback(pipeline="moderation", session_id="s1", profile_name="oss", run=run)
        )
    # Recorded for visibility, but not a fallback.
    assert len(oss_enabled) == 1
    assert oss_enabled[0].fell_back is False
    assert oss_enabled[0].reason is FallbackReason.BAD_OUTPUT


def test_success_on_oss_emits_nothing(oss_enabled):
    async def run(attempt):
        return f"ok-{attempt.kind}"

    result = asyncio.run(
        execute_with_fallback(pipeline="moderation", session_id="s1", profile_name="oss", run=run)
    )
    assert result == "ok-oss"
    assert oss_enabled == []


def test_both_tiers_fail_raises_and_records_both(oss_enabled):
    async def run(attempt):
        raise ConnectionError("refused")

    with pytest.raises(ConnectionError):
        asyncio.run(
            execute_with_fallback(pipeline="moderation", session_id="s1", profile_name="oss", run=run)
        )
    assert len(oss_enabled) == 2
    assert oss_enabled[0].fell_back is True   # oss -> managed
    assert oss_enabled[1].fell_back is False  # managed was last tier


def test_legacy_session_no_fallback_attempted(oss_enabled):
    calls = []

    async def run(attempt):
        calls.append(attempt.kind)
        raise ConnectionError("refused")

    with pytest.raises(ConnectionError):
        asyncio.run(
            execute_with_fallback(pipeline="moderation", session_id="s1", profile_name="legacy", run=run)
        )
    assert calls == ["managed"]  # only one tier for legacy sessions
    assert len(oss_enabled) == 1 and oss_enabled[0].fell_back is False


# ── stream_with_fallback() (first-token commit) ──────────────────────────────

async def _collect(agen):
    return [c async for c in agen]


def test_stream_success_on_oss_no_fallback(oss_enabled):
    async def make_stream(attempt):
        for c in ["a", "b", "c"]:
            yield f"{attempt.kind}:{c}"

    chunks = asyncio.run(
        _collect(fb.stream_with_fallback(
            pipeline="chat", session_id="s", profile_name="oss", make_stream=make_stream))
    )
    assert chunks == ["oss:a", "oss:b", "oss:c"]
    assert oss_enabled == []


def test_stream_precommit_failure_swaps_to_managed(oss_enabled):
    async def make_stream(attempt):
        if attempt.kind == "oss":
            raise ConnectionError("refused")  # before any yield
            yield  # pragma: no cover - makes this an async generator
        for c in ["x", "y"]:
            yield f"managed:{c}"

    chunks = asyncio.run(
        _collect(fb.stream_with_fallback(
            pipeline="chat", session_id="s", profile_name="oss", make_stream=make_stream))
    )
    assert chunks == ["managed:x", "managed:y"]
    assert len(oss_enabled) == 1
    ev = oss_enabled[0]
    assert ev.committed is False and ev.fell_back is True and ev.reason is FallbackReason.CONNECTION


def test_stream_postcommit_failure_propagates(oss_enabled):
    async def make_stream(attempt):
        yield f"{attempt.kind}:first"
        raise ConnectionError("died mid-stream")

    got = []

    async def drive():
        async for c in fb.stream_with_fallback(
            pipeline="chat", session_id="s", profile_name="oss", make_stream=make_stream
        ):
            got.append(c)

    with pytest.raises(ConnectionError):
        asyncio.run(drive())
    assert got == ["oss:first"]              # committed chunk reached the client
    assert len(oss_enabled) == 1
    assert oss_enabled[0].committed is True and oss_enabled[0].fell_back is False


def test_stream_precommit_bad_output_does_not_swap(oss_enabled):
    class UnexpectedModelBehavior(Exception):
        pass

    async def make_stream(attempt):
        raise UnexpectedModelBehavior("schema")
        yield  # pragma: no cover

    with pytest.raises(UnexpectedModelBehavior):
        asyncio.run(_collect(fb.stream_with_fallback(
            pipeline="chat", session_id="s", profile_name="oss", make_stream=make_stream)))
    assert len(oss_enabled) == 1
    assert oss_enabled[0].committed is False and oss_enabled[0].fell_back is False


# ── with_first_token_deadline() (time-to-first-token bound) ──────────────────

def _attempt(timeout):
    from app.llm_core.factory import MaterializedTier
    return MaterializedTier(kind="oss", handle=object(), model_name="gemma-test",
                            provider="vllm", endpoint="http://oss:8020/v1", timeout=timeout)


def test_ttft_slow_first_token_raises_timeout():
    """First token later than the deadline -> TimeoutError (classify -> TIMEOUT)."""
    async def raw(attempt):
        await asyncio.sleep(0.20)
        yield "late"

    async def drive():
        return await _collect(fb.with_first_token_deadline(_attempt(0.05), raw(None)))

    with pytest.raises(TimeoutError):
        asyncio.run(drive())
    assert classify(TimeoutError("x")) is FallbackReason.TIMEOUT


def test_ttft_disarmed_after_first_token_allows_long_gap():
    """The deadline must bound ONLY time-to-first-token: a long gap AFTER the
    first token (e.g. a tool round-trip) must not abort the stream."""
    async def raw(attempt):
        yield "first"                 # arrives immediately, well within deadline
        await asyncio.sleep(0.15)     # > deadline, but post-first-token -> allowed
        yield "second"

    out = asyncio.run(_collect(fb.with_first_token_deadline(_attempt(0.05), raw(None))))
    assert out == ["first", "second"]


def test_ttft_none_is_noop():
    """timeout=None (fallback disabled) -> no deadline, slow first token is fine."""
    async def raw(attempt):
        await asyncio.sleep(0.05)
        yield "ok"

    out = asyncio.run(_collect(fb.with_first_token_deadline(_attempt(None), raw(None))))
    assert out == ["ok"]


def test_ttft_timeout_triggers_fallback_to_managed(oss_enabled, install_chain):
    """End to end: a silent OSS first-token hang swaps to managed before commit."""
    async def make_stream(attempt):
        async for c in fb.with_first_token_deadline(attempt, _raw(attempt)):
            yield c

    async def _raw(attempt):
        if attempt.kind == "oss":
            await asyncio.sleep(0.30)     # exceeds the oss attempt timeout
            yield "oss-never"             # pragma: no cover
        else:
            yield "managed:ok"

    # Tight oss first-token timeout so the test is fast; managed has no deadline.
    install_chain(oss_timeout=0.05)
    chunks = asyncio.run(_collect(fb.stream_with_fallback(
        pipeline="chat", session_id="s", profile_name="oss", make_stream=make_stream)))
    assert chunks == ["managed:ok"]
    reasons = [e.reason for e in oss_enabled]
    assert FallbackReason.TIMEOUT in reasons
    assert oss_enabled[0].fell_back is True and oss_enabled[0].committed is False


def test_ttft_aclose_midstream_is_clean():
    """Client disconnect mid-stream: aclosing the wrapper must not raise or leak,
    and must close the underlying source generator (regression: the old cancel-scope
    impl threw 'exit cancel scope in a different task' / 'aclose: already running')."""
    state = {"source_closed": False}

    async def raw(attempt):
        try:
            for i in range(20):
                await asyncio.sleep(0.01)
                yield f"c{i}"
        finally:
            state["source_closed"] = True

    async def go():
        agen = fb.with_first_token_deadline(_attempt(1.0), raw(None))
        out = []
        async for c in agen:
            out.append(c)
            if len(out) == 2:
                await agen.aclose()  # simulate client disconnect
                break
        await asyncio.sleep(0.05)  # let the drained task unwind
        return out

    out = asyncio.run(go())
    assert out == ["c0", "c1"]
    assert state["source_closed"] is True
