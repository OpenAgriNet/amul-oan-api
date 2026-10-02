"""Voice moderation on the turn's ExecutionContext.

Carried over from voice-oan-api's tests/test_moderation_characterization.py and
tests/test_voice_moderation_fallback.py. Those call ``check_moderation`` with a
``variant=`` keyword the function no longer takes, so on amul-dev they fail
before reaching moderation; here they run against the current signature. What
they pin is unchanged: which condition allows and which blocks, on the legacy
path (fail open) and on the fallback path (fail closed).
"""
import asyncio
import dataclasses
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from openai import AsyncOpenAI

from app.llm_core import execution as llm_execution
from app.llm_core.config_model import (
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
)
from app.llm_core.execution import ExecutionContext
from app.voice import moderation as mod

_OSS = Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1")
_MANAGED = Tier(provider=Provider.OPENAI, model="gpt")


def _execution(*, fallback, profile="oss", oss_moderation=(_OSS, _MANAGED)):
    config = PipelineConfig(
        profiles=[
            NamedProfile(name="oss", weight=50, steps={Step.MODERATION: StepConfig(tiers=list(oss_moderation))}),
            NamedProfile(name="managed", weight=50, steps={Step.MODERATION: StepConfig(tiers=[_MANAGED])}),
            NamedProfile(name="agent-only", weight=0, steps={Step.AGENT: StepConfig(tiers=[_MANAGED])}),
        ],
        fallback_enabled=fallback,
    )
    return ExecutionContext(session_id="s", config=config, profile_name=profile)


def _resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _check(text, *, execution):
    return asyncio.run(mod.check_moderation(text, "gu", execution=execution))


_MALFORMED = ["", "not json", '"a string"', '{"category": "weird"}']
_VALID_IN_SCOPE = '{"category": "in_scope", "reason": "ok"}'
_VALID_REJECT = '{"category": "offensive", "reason": "x"}'
_VALID_OTHER = '{"category": "irrelevant"}'


# ── parser: fail open ───────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", _MALFORMED)
def test_parse_open_malformed_allows(raw):
    v = mod._parse_verdict(raw)
    assert v.category == "in_scope"
    assert v.failed_open is True
    assert v.failed_closed is False
    assert v.rejected is False


def test_parse_open_valid_in_scope_unchanged():
    v = mod._parse_verdict(_VALID_IN_SCOPE)
    assert v.category == "in_scope" and v.failed_open is False and not v.rejected


def test_parse_open_valid_reject_unchanged():
    v = mod._parse_verdict(_VALID_REJECT)
    assert v.category == "offensive" and v.failed_open is False and v.rejected


def test_parse_open_valid_other_category_unchanged():
    v = mod._parse_verdict(_VALID_OTHER)
    assert v.category == "irrelevant" and v.failed_open is False and v.rejected


# ── parser: fail closed ─────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", _MALFORMED)
def test_parse_strict_malformed_blocks(raw):
    for v in (mod._parse_verdict_strict(raw), mod._parse_verdict(raw, fail_closed=True)):
        assert v.category == "unavailable"
        assert v.failed_closed is True
        assert v.failed_open is False
        assert v.rejected is True


def test_parse_strict_valid_verdicts_unchanged():
    assert mod._parse_verdict_strict(_VALID_IN_SCOPE).category == "in_scope"
    assert not mod._parse_verdict_strict(_VALID_IN_SCOPE).failed_closed
    assert mod._parse_verdict_strict(_VALID_REJECT).category == "offensive"
    assert mod._parse_verdict_strict(_VALID_REJECT).rejected


def test_strict_alias_equivalent_to_parametrized():
    for raw in _MALFORMED + [_VALID_IN_SCOPE, _VALID_REJECT, _VALID_OTHER]:
        a = mod._parse_verdict_strict(raw)
        b = mod._parse_verdict(raw, fail_closed=True)
        assert (a.category, a.failed_open, a.failed_closed) == (b.category, b.failed_open, b.failed_closed)


def test_unavailable_verdict_has_generic_decline():
    v = mod._block_unavailable("all tiers down")
    assert v.rejected is True
    assert v.decline_text_en()


# ── legacy path (fallback off): fail open ───────────────────────────────────


@pytest.fixture
def _legacy(monkeypatch):
    monkeypatch.setattr(mod, "_get_langfuse", lambda: None)
    return _execution(fallback=False)


def test_legacy_path_used_when_fallback_is_off(monkeypatch):
    sentinel = mod._allow("legacy-was-called", failed_open=True)

    async def fake_legacy(text, source_lang, recent_history_text="", **kwargs):
        return sentinel

    monkeypatch.setattr(mod, "_check_moderation_legacy", fake_legacy)

    assert _check("hi", execution=_execution(fallback=False)) is sentinel


def test_legacy_path_fails_open_on_client_error(_legacy, monkeypatch):
    monkeypatch.setattr(mod, "_moderation_client_and_model", lambda execution: ("client", "model", "openai"))

    async def _boom(client, model, text, source_lang, recent_history_text=""):
        raise ConnectionError("moderation backend down")

    monkeypatch.setattr(mod, "_create_moderation_response", _boom)

    v = _check("hi", execution=_legacy)
    assert v.category == "in_scope" and v.failed_open is True and v.rejected is False


def test_legacy_path_malformed_output_fails_open(_legacy, monkeypatch):
    monkeypatch.setattr(mod, "_moderation_client_and_model", lambda execution: ("client", "model", "openai"))

    async def _garbage(client, model, text, source_lang, recent_history_text=""):
        return _resp("not-json-at-all")

    monkeypatch.setattr(mod, "_create_moderation_response", _garbage)

    v = _check("hi", execution=_legacy)
    assert v.category == "in_scope" and v.failed_open is True and not v.rejected


def test_legacy_moderation_fails_open_when_client_build_raises(_legacy, monkeypatch):
    def _boom(execution):
        raise ValueError("vllm raw-openai client requires an endpoint")

    monkeypatch.setattr(mod, "_moderation_client_and_model", _boom)

    v = _check("hi", execution=_legacy)
    assert v.category == "in_scope" and not v.rejected and v.failed_open


def test_legacy_client_failure_reports_the_model_it_meant_to_use(_legacy, monkeypatch):
    monkeypatch.setattr(mod, "_MODERATION_PROVIDER", "vllm")
    monkeypatch.setenv("OSS_PRETRANSLATION_MODEL", "gemma-moderation")

    def _boom(execution):
        raise ValueError("no client")

    monkeypatch.setattr(mod, "_moderation_client_and_model", _boom)

    v = _check("hi", execution=_legacy)
    assert (v.requested_tier, v.requested_provider, v.requested_model) == ("oss", "vllm", "gemma-moderation")
    assert v.actual_tier == "failed"


def test_legacy_success_records_requested_actual(_legacy, monkeypatch):
    monkeypatch.setattr(mod, "_moderation_client_and_model", lambda execution: ("legacy-client", "legacy-model", "openai"))

    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        return _resp('{"category": "in_scope", "reason": "ok"}')

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = _check("hi", execution=_legacy)
    assert v.category == "in_scope"
    assert v.requested_tier == "managed"
    assert v.requested_provider == "openai"
    assert v.requested_model == "legacy-model"
    assert v.actual_tier == "managed"
    assert v.actual_provider == "openai"
    assert v.actual_model == "legacy-model"
    assert v.fallback_used is False
    assert v.attempts and v.attempts[0]["status"] == "ok"


@pytest.mark.parametrize("provider, profile_model, label", [("vllm", "gemma", "vllm"), ("openai", "gpt", "openai")])
def test_legacy_client_follows_voice_moderation_provider_not_the_session(provider, profile_model, label, monkeypatch):
    monkeypatch.setattr(mod, "_MODERATION_PROVIDER", provider)

    client, model, provider_label = mod._moderation_client_and_model(
        _execution(fallback=False, profile="managed")
    )

    assert isinstance(client, AsyncOpenAI)
    assert (model, provider_label) == (profile_model, label)


# ── fallback path (fallback on): fail closed ────────────────────────────────


@pytest.fixture
def oss_on(monkeypatch):
    """Fallback on, an [oss, managed] chain, deterministic per-kind backends, and
    the fallback events captured."""
    events = []
    monkeypatch.setattr(llm_execution, "emit", events.append)
    monkeypatch.setattr(mod, "_client_model_for_kind", lambda execution, kind: (
        f"{kind}-client",
        f"{kind}-model",
        "vllm" if kind == "oss" else "openai",
    ))
    return events


def test_oss_failure_falls_back_to_managed(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        if model.startswith("oss"):
            raise ConnectionError("vllm refused")
        return _resp('{"category": "in_scope", "reason": "ok"}')

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = _check("hi", execution=_execution(fallback=True))
    assert v.category == "in_scope" and not v.rejected
    assert v.requested_tier == "oss"
    assert v.requested_provider == "vllm"
    assert v.actual_tier == "managed"
    assert v.actual_provider == "openai"
    assert v.fallback_used is True
    assert v.attempts and v.attempts[0]["status"] == "error"
    assert v.attempts[1]["status"] == "ok"
    assert len(oss_on) == 1 and oss_on[0].fell_back is True


def test_both_tiers_fail_fails_closed(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        raise ConnectionError("down")

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = _check("hi", execution=_execution(fallback=True))
    assert v.category == "unavailable" and v.rejected and v.failed_closed
    assert v.requested_tier == "oss"
    assert v.actual_tier == "failed"
    assert v.fallback_used is True
    assert v.attempts and len(v.attempts) == 2
    assert v.attempts[0]["status"] == "error"
    assert v.attempts[1]["status"] == "error"


def test_valid_reject_on_oss_does_not_fall_back(oss_on, monkeypatch):
    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        return _resp('{"category": "offensive", "reason": "abuse"}')

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = _check("...", execution=_execution(fallback=True))
    assert v.category == "offensive" and v.rejected
    assert v.fallback_used is False
    assert v.actual_tier == "oss"
    assert oss_on == []


def test_malformed_output_on_every_tier_blocks_closed(oss_on, monkeypatch):
    async def _garbage(client, model, text, source_lang, recent_history_text=""):
        return _resp("garbage")

    monkeypatch.setattr(mod, "_create_moderation_response", _garbage)

    v = _check("x", execution=_execution(fallback=True))
    assert v.category == "unavailable" and v.failed_closed and v.rejected


def test_a_profile_without_moderation_walks_the_managed_profile(oss_on, monkeypatch):
    """Voice's finding #1: a profile that omits moderation must never bypass it.
    The requested tier defaults to managed and the managed profile is walked."""
    models = []

    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        models.append(model)
        return _resp(_VALID_REJECT)

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)

    v = _check("x", execution=_execution(fallback=True, profile="agent-only"))
    assert v.requested_tier == "managed"
    assert v.category == "offensive" and v.rejected
    assert models == ["managed-model"]


def test_a_primary_that_cannot_be_built_degrades_to_managed(monkeypatch):
    """A vLLM tier with no endpoint: requested falls to managed, and the check
    still returns a verdict instead of raising into the caller."""
    monkeypatch.setattr(llm_execution, "emit", lambda event: None)

    async def _down(client, model, text, source_lang, recent_history_text=""):
        raise ConnectionError("down")

    monkeypatch.setattr(mod, "_create_moderation_response", _down)
    no_endpoint = Tier(provider=Provider.VLLM, model="gemma")

    v = _check("hi", execution=_execution(fallback=True, oss_moderation=(no_endpoint, _MANAGED)))

    assert v.requested_tier == "managed"
    assert [a["tier"] for a in v.attempts] == ["managed"]
    assert v.category == "unavailable" and v.failed_closed


# ── the real clients ────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind, model, label", [("oss", "gemma", "vllm"), ("managed", "gpt", "openai")])
def test_each_attempt_gets_a_bare_openai_client_for_its_kind(kind, model, label):
    client, got_model, got_label = mod._client_model_for_kind(_execution(fallback=True, profile="managed"), kind)

    assert isinstance(client, AsyncOpenAI)
    assert (got_model, got_label) == (model, label)


def test_the_fallback_walk_requests_raw_clients(monkeypatch):
    kinds = []

    async def fake_run_adapter(self, step, invoke, *, client_kind=None):
        kinds.append((step, client_kind))
        return mod._allow("ok")

    monkeypatch.setattr(ExecutionContext, "run_adapter", fake_run_adapter)

    _check("hi", execution=_execution(fallback=True))

    assert kinds == [(Step.MODERATION, mod.StepClientKind.RAW_OPENAI)]


# ── empty input ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fallback", [False, True])
def test_empty_input_allows_regardless_of_flag(fallback):
    v = _check("   ", execution=_execution(fallback=fallback))
    assert v.category == "in_scope" and not v.rejected


def test_a_session_on_a_named_profile_is_moderated_on_that_profile(oss_on, monkeypatch):
    requested = []

    async def fake_create(client, model, text, source_lang, recent_history_text=""):
        return _resp(_VALID_IN_SCOPE)

    monkeypatch.setattr(mod, "_create_moderation_response", fake_create)
    managed = dataclasses.replace(_execution(fallback=True), profile_name="managed")
    requested.append(_check("hi", execution=managed).requested_tier)
    requested.append(_check("hi", execution=_execution(fallback=True)).requested_tier)

    assert requested == ["managed", "oss"]
