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


# ── attempt_chain() + execute_with_fallback() ────────────────────────────────

@pytest.fixture
def oss_enabled(monkeypatch):
    """Fallback ON, OSS available; capture emitted events instead of logging."""
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", object())
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma-test")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")
    events = []
    monkeypatch.setattr(fb, "emit", events.append)
    return events


def test_chain_oss_two_tiers(oss_enabled):
    chain = fb.attempt_chain("oss", "moderation")
    assert [a.kind for a in chain] == ["oss", "managed"]
    assert chain[0].endpoint == "http://oss:8020/v1"
    assert chain[0].timeout is not None


def test_chain_legacy_single_tier(oss_enabled):
    chain = fb.attempt_chain("legacy", "moderation")
    assert [a.kind for a in chain] == ["managed"]


def test_chain_disabled_single_tier_no_deadline(monkeypatch):
    monkeypatch.setattr(fb.settings, "fallback_enabled", False)
    chain = fb.attempt_chain("oss", "moderation")
    assert len(chain) == 1
    assert chain[0].timeout is None


def test_falls_back_to_managed_on_connection_error(oss_enabled):
    async def run(attempt):
        if attempt.kind == "oss":
            raise ConnectionError("connection refused")
        return "managed-result"

    result = asyncio.run(
        execute_with_fallback(pipeline="moderation", session_id="s1", variant="oss", run=run)
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
            execute_with_fallback(pipeline="moderation", session_id="s1", variant="oss", run=run)
        )
    # Recorded for visibility, but not a fallback.
    assert len(oss_enabled) == 1
    assert oss_enabled[0].fell_back is False
    assert oss_enabled[0].reason is FallbackReason.BAD_OUTPUT


def test_success_on_oss_emits_nothing(oss_enabled):
    async def run(attempt):
        return f"ok-{attempt.kind}"

    result = asyncio.run(
        execute_with_fallback(pipeline="moderation", session_id="s1", variant="oss", run=run)
    )
    assert result == "ok-oss"
    assert oss_enabled == []


def test_both_tiers_fail_raises_and_records_both(oss_enabled):
    async def run(attempt):
        raise ConnectionError("refused")

    with pytest.raises(ConnectionError):
        asyncio.run(
            execute_with_fallback(pipeline="moderation", session_id="s1", variant="oss", run=run)
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
            execute_with_fallback(pipeline="moderation", session_id="s1", variant="legacy", run=run)
        )
    assert calls == ["managed"]  # only one tier for legacy sessions
    assert len(oss_enabled) == 1 and oss_enabled[0].fell_back is False
