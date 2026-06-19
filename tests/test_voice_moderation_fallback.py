"""Tests for voice moderation matching chat: variant-routed OSS->managed
fallback + fail-closed (behind FALLBACK_ENABLED)."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
from types import SimpleNamespace

import pytest

from app.services import fallback as fb
from app.services import moderation as mod


def _resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# -- strict parser (fail-closed) ---------------------------------------------

def test_parse_strict_valid_in_scope():
    v = mod._parse_verdict_strict('{"category": "in_scope", "reason": "ok"}')
    assert v.category == "in_scope" and not v.rejected and not v.failed_closed


def test_parse_strict_valid_reject():
    v = mod._parse_verdict_strict('{"category": "offensive", "reason": "x"}')
    assert v.category == "offensive" and v.rejected


@pytest.mark.parametrize("raw", ["", "not json", '"a string"', '{"category": "weird"}'])
def test_parse_strict_malformed_fails_closed(raw):
    v = mod._parse_verdict_strict(raw)
    assert v.category == "unavailable" and v.rejected and v.failed_closed


def test_unavailable_verdict_has_generic_decline():
    v = mod._block_unavailable("all tiers down")
    assert v.rejected is True
    assert v.decline_text_en()  # a non-empty generic "try again" message


# -- check_moderation: fallback + fail-closed --------------------------------

@pytest.fixture
def oss_on(monkeypatch):
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", object())
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma-test")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")
    events = []
    monkeypatch.setattr(fb, "emit", events.append)
    # deterministic per-kind backends
    monkeypatch.setattr(mod, "_client_model_for_kind",
                        lambda kind: (f"{kind}-client", f"{kind}-model", kind))
    return events


def test_oss_failure_falls_back_to_managed(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        if model.startswith("oss"):
            raise ConnectionError("vllm refused")
        return _resp('{"category": "in_scope", "reason": "ok"}')
    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = asyncio.run(mod.check_moderation("hi", "gu", variant="oss", session_id="s"))
    assert v.category == "in_scope" and not v.rejected
    assert len(oss_on) == 1 and oss_on[0].fell_back is True


def test_both_tiers_fail_fails_closed(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        raise ConnectionError("down")
    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = asyncio.run(mod.check_moderation("hi", "gu", variant="oss", session_id="s"))
    assert v.category == "unavailable" and v.rejected and v.failed_closed


def test_valid_reject_on_oss_does_not_fall_back(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        return _resp('{"category": "offensive", "reason": "abuse"}')
    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = asyncio.run(mod.check_moderation("...", "gu", variant="oss", session_id="s"))
    assert v.category == "offensive" and v.rejected
    assert oss_on == []  # a valid verdict is success, no fallback


def test_legacy_path_used_when_disabled(monkeypatch):
    monkeypatch.setattr(fb.settings, "fallback_enabled", False)
    sentinel = mod._allow("legacy-was-called", failed_open=True)

    async def fake_legacy(text, source_lang, recent_history_text=""):
        return sentinel
    monkeypatch.setattr(mod, "_check_moderation_legacy", fake_legacy)

    v = asyncio.run(mod.check_moderation("hi", "gu", variant="oss", session_id="s"))
    assert v is sentinel  # disabled -> today's fail-open legacy path, unchanged
