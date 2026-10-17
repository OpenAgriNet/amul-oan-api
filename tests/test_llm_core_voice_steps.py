"""llm_core carrying voice's steps next to chat's.

Voice's config has a ``non_meaningful`` step, and voice runs moderation and its
other classifier calls as bare ``chat.completions`` requests rather than through
a pydantic-ai agent. These pin that both work here, and that a chat config and
chat's own moderation are built exactly as before.
"""
import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.llm_core import config_source, runtime, split
from app.llm_core.config_model import (
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepClientKind,
    StepConfig,
    Tier,
)
from app.llm_core.execution import ExecutionContext
from app.llm_core.factory import STEP_CLIENT_KIND, build_handle

_OSS = Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1")
_MANAGED = Tier(provider=Provider.OPENAI, model="gpt")
_ANTHROPIC = Tier(provider=Provider.ANTHROPIC, model="haiku")


def _config(*, fallback=True, moderation=(_OSS, _MANAGED), defaults=None):
    return PipelineConfig(
        profiles=[
            NamedProfile(
                name="oss",
                weight=50,
                steps={
                    Step.AGENT: StepConfig(tiers=[_OSS, _MANAGED]),
                    Step.MODERATION: StepConfig(tiers=list(moderation)),
                },
            ),
            NamedProfile(
                name="managed",
                weight=50,
                steps={
                    Step.AGENT: StepConfig(tiers=[_MANAGED]),
                    Step.MODERATION: StepConfig(tiers=[_MANAGED]),
                },
            ),
        ],
        defaults=defaults if defaults is not None else {},
        fallback_enabled=fallback,
    )


def _voice_config(**kw):
    return _config(defaults={Step.NON_MEANINGFUL: StepConfig(tiers=[_OSS])}, **kw)


def _context(config, profile="oss"):
    return ExecutionContext(session_id="s", config=config, profile_name=profile)


# ── the step ────────────────────────────────────────────────────────────────


def test_a_voice_config_with_non_meaningful_parses():
    raw = {
        "profiles": [{"name": "managed", "weight": 100, "steps": {
            "agent": {"tiers": [{"provider": "openai", "model": "gpt"}]},
        }}],
        "defaults": {"non_meaningful": {"tiers": [
            {"provider": "vllm", "model": "gemma", "endpoint": "http://oss:8020/v1"},
        ]}},
    }

    config = PipelineConfig(**raw)

    assert config.step_config(config.profiles[0], Step.NON_MEANINGFUL).tiers[0].model == "gemma"


def test_a_chat_config_has_no_non_meaningful_step():
    with pytest.raises(ValueError, match="no config for step=non_meaningful"):
        _context(_config()).target(Step.NON_MEANINGFUL)


def test_non_meaningful_is_always_a_bare_openai_client():
    assert STEP_CLIENT_KIND[Step.NON_MEANINGFUL] is StepClientKind.RAW_OPENAI

    target = _context(_voice_config()).target(Step.NON_MEANINGFUL)

    assert isinstance(target.handle, AsyncOpenAI)
    assert str(target.handle.base_url).startswith("http://oss:8020/v1")


def test_a_chat_trace_is_unchanged_and_a_voice_trace_records_the_step():
    chat = _context(_config()).begin_trace()
    voice = _context(_voice_config()).begin_trace()

    assert "non_meaningful" not in chat.to_metadata()["steps"]
    assert "non_meaningful" in voice.to_metadata()["steps"]


# ── the raw client kind ─────────────────────────────────────────────────────


def test_raw_openai_builds_openai_compatible_clients_only():
    azure = Tier(
        provider=Provider.AZURE, model="gpt", endpoint="https://x.openai.azure.com",
        api_version="2024-06-01", api_key_env="OPENAI_API_KEY",
    )

    gemini = Tier(provider=Provider.GEMINI, model="gemini-2.5-flash", api_key_env="OPENAI_API_KEY")

    assert isinstance(build_handle(_MANAGED, StepClientKind.RAW_OPENAI), AsyncOpenAI)
    assert isinstance(build_handle(azure, StepClientKind.RAW_OPENAI), AsyncAzureOpenAI)
    for tier in (_ANTHROPIC, gemini):
        with pytest.raises(ValueError, match="not valid for a RAW_OPENAI step"):
            build_handle(tier, StepClientKind.RAW_OPENAI)


def test_a_raw_vllm_tier_without_an_endpoint_refuses_to_build():
    with pytest.raises(ValueError, match="endpoint"):
        build_handle(Tier(provider=Provider.VLLM, model="gemma"), StepClientKind.RAW_OPENAI)


def test_chat_moderation_is_still_built_as_an_agent_model():
    target = _context(_config()).target(Step.MODERATION)

    assert target.client_kind is StepClientKind.AGENT
    assert not isinstance(target.handle, AsyncOpenAI)


def test_a_caller_can_ask_for_moderation_as_a_raw_client():
    target = _context(_config()).target(Step.MODERATION, client_kind=StepClientKind.RAW_OPENAI)

    assert target.client_kind is StepClientKind.RAW_OPENAI
    assert isinstance(target.handle, AsyncOpenAI)


@pytest.mark.parametrize("fallback", [True, False])
def test_run_adapter_hands_every_attempt_the_requested_kind(fallback):
    seen = []

    async def _invoke(target):
        seen.append(target.client_kind)
        return "ok"

    context = _context(_config(fallback=fallback))

    assert asyncio.run(context.run_adapter(Step.MODERATION, _invoke, client_kind=StepClientKind.RAW_OPENAI)) == "ok"
    assert asyncio.run(context.run_adapter(Step.MODERATION, _invoke)) == "ok"
    assert seen == [StepClientKind.RAW_OPENAI, StepClientKind.AGENT]


def test_split_builds_the_chain_as_the_requested_kind():
    config = _config()

    default = asyncio.run(split.resolve_chain("s", Step.MODERATION, config, profile_name="oss"))
    raw = asyncio.run(split.resolve_chain(
        "s", Step.MODERATION, config, profile_name="oss", client_kind=StepClientKind.RAW_OPENAI,
    ))

    assert [t.client_kind for t in default] == [StepClientKind.AGENT, StepClientKind.AGENT]
    assert [t.client_kind for t in raw] == [StepClientKind.RAW_OPENAI, StepClientKind.RAW_OPENAI]
    assert [t.tier for t in raw] == [t.tier for t in default]


# ── validation ──────────────────────────────────────────────────────────────


def test_non_meaningful_rejects_a_provider_without_an_openai_client():
    config = _config(defaults={Step.NON_MEANINGFUL: StepConfig(tiers=[_ANTHROPIC])})

    with pytest.raises(ValueError, match="step=non_meaningful provider=anthropic"):
        runtime.validate_config(config)


def test_moderation_on_anthropic_stays_valid_for_chat(monkeypatch):
    monkeypatch.delenv(config_source.CHANNEL_ENV, raising=False)

    runtime.validate_config(_config(moderation=(_ANTHROPIC,)))


def test_moderation_on_the_voice_channel_must_be_a_bare_openai_client(monkeypatch):
    """Voice's moderation is a chat.completions call: an anthropic tier would only
    fail per call, so a voice deployment refuses it at boot, as voice-oan-api did."""
    monkeypatch.setenv(config_source.CHANNEL_ENV, "voice")

    runtime.validate_config(_voice_config())
    with pytest.raises(ValueError, match="step=moderation provider=anthropic"):
        runtime.validate_config(_config(moderation=(_ANTHROPIC,)))
