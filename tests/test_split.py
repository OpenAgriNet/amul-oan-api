"""Unit tests for explicit profile-name chain resolution in ``app/llm_core/split.py``.

The split layer no longer bucket-routes sessions. Callers provide a profile name
(e.g. ``oss``) and ``resolve_chain`` materializes that profile's step tiers.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OSS_INFERENCE_API_KEY", "test-oss-key")

import pytest
from pydantic import ValidationError

from app.llm_core import split
from app.llm_core.config_model import (
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
)


def _oss_tier(model="gemma"):
    return Tier(
        provider=Provider.VLLM,
        model=model,
        endpoint="http://oss:8020/v1",
        api_key_env="OSS_INFERENCE_API_KEY",
        timeout_ms=8000,
    )


def _managed_tier(model="gpt-4.1"):
    return Tier(
        provider=Provider.OPENAI,
        model=model,
        api_key_env="OPENAI_API_KEY",
        timeout_ms=20000,
    )


def two_profile_config(pct: int, ttl: int = 604800) -> PipelineConfig:
    oss_steps = {
        Step.AGENT: StepConfig(tiers=[_oss_tier(), _managed_tier()]),
        Step.MODERATION: StepConfig(tiers=[_oss_tier(), _managed_tier()]),
    }
    managed_steps = {
        Step.AGENT: StepConfig(tiers=[_managed_tier()]),
        Step.MODERATION: StepConfig(tiers=[_managed_tier()]),
    }
    return PipelineConfig(
        profiles=[
            NamedProfile(name="oss", weight=pct, steps=oss_steps),
            NamedProfile(name="managed", weight=100 - pct, steps=managed_steps),
        ],
        sticky_ttl_s=ttl,
    )


def test_pipeline_config_rejects_weights_not_summing_to_100():
    with pytest.raises(ValidationError):
        PipelineConfig(
            profiles=[
                NamedProfile(
                    name="a",
                    weight=50,
                    steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])},
                ),
                NamedProfile(
                    name="b",
                    weight=30,
                    steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])},
                ),
            ]
        )


def test_resolve_chain_matches_oss_profile_tiers():
    import asyncio

    cfg = two_profile_config(100)
    chain = asyncio.run(split.resolve_chain("", Step.AGENT, cfg, profile_name="oss"))
    assert len(chain) == 2
    assert [c.model_name for c in chain] == ["gemma", "gpt-4.1"]
    assert [c.kind for c in chain] == ["oss", "managed"]
    assert [c.provider for c in chain] == ["vllm", "openai"]
    assert all(c.handle is not None for c in chain)
    assert chain[0].timeout == 8.0 and chain[1].timeout == 20.0


def test_resolve_chain_matches_managed_profile_single_tier():
    import asyncio

    cfg = two_profile_config(0)
    chain = asyncio.run(split.resolve_chain("", Step.AGENT, cfg, profile_name="managed"))
    assert [c.kind for c in chain] == ["managed"]
    assert [c.model_name for c in chain] == ["gpt-4.1"]


def test_resolve_chain_unknown_profile_failsafe_to_managed():
    import asyncio

    cfg = two_profile_config(100)
    chain = asyncio.run(split.resolve_chain("", Step.AGENT, cfg, profile_name="legacy"))
    assert [c.kind for c in chain] == ["managed"]


def test_resolve_chain_never_empty_for_configured_step():
    import asyncio

    cfg = two_profile_config(50)
    chain = asyncio.run(split.resolve_chain("", Step.MODERATION, cfg, profile_name="oss"))
    assert len(chain) >= 1


def test_fallback_chain_uses_split(monkeypatch):
    import asyncio
    from app.services import fallback as fb

    sentinel = ["MATERIALIZED_TIER"]

    async def _spy(session_id, step, pipeline=None, *, profile_name="managed"):
        assert step is Step.MODERATION
        assert profile_name == "oss"
        return sentinel

    monkeypatch.setattr(split, "resolve_chain", _spy)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", profile_name="oss"))
    assert chain is sentinel


def test_fallback_chain_degrades_to_managed_on_split_error(monkeypatch):
    import asyncio
    from app.llm_core import runtime
    from app.services import fallback as fb

    runtime.configure(run_self_check=False)

    async def _boom(session_id, step, pipeline=None, *, profile_name="managed"):
        raise RuntimeError("config blew up")

    monkeypatch.setattr(split, "resolve_chain", _boom)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", profile_name="oss"))
    assert len(chain) >= 1
    assert chain[-1].kind == "managed"


_EXAMPLE_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.example.yaml"
)


def test_nway_three_profile_yaml_serves_named_profiles(monkeypatch):
    from app.llm_core import resolver, runtime

    monkeypatch.setenv("PIPELINE_CONFIG_PATH", _EXAMPLE_YAML)
    cfg = runtime.configure(run_self_check=False)
    try:
        assert {p.name for p in cfg.profiles} == {"gemma", "qwen", "gpt"}
        assert [p.weight for p in cfg.profiles] == [5, 10, 85]

        gemma = resolver.primary_tier(Step.AGENT, "gemma")
        qwen = resolver.primary_tier(Step.AGENT, "qwen")
        gpt = resolver.primary_tier(Step.AGENT, "gpt")
        assert (gemma.model_name, gemma.kind) == ("gemma-4-31b-it", "oss")
        assert (qwen.model_name, qwen.kind) == ("qwen2.5-32b-instruct", "oss")
        assert (gpt.model_name, gpt.kind) == ("gpt-4.1", "managed")

        assert resolver.primary_tier(Step.AGENT, "does-not-exist").model_name == "gemma-4-31b-it"
    finally:
        monkeypatch.delenv("PIPELINE_CONFIG_PATH", raising=False)
        runtime.configure(run_self_check=False)


def test_self_check_reports_broken_third_profile_nonfatal(monkeypatch, caplog):
    import logging
    from app.llm_core import runtime

    broken_agent = Tier(
        provider=Provider.VLLM,
        model="broken",
        endpoint=None,
        api_key_env="OSS_INFERENCE_API_KEY",
        timeout_ms=8000,
    )
    cfg = PipelineConfig(
        profiles=[
            NamedProfile(
                name="oss",
                weight=45,
                steps={Step.AGENT: StepConfig(tiers=[_oss_tier(), _managed_tier()])},
            ),
            NamedProfile(
                name="managed",
                weight=45,
                steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])},
            ),
            NamedProfile(
                name="broken",
                weight=10,
                steps={Step.AGENT: StepConfig(tiers=[broken_agent])},
            ),
        ]
    )
    monkeypatch.setattr(runtime, "PIPELINE", cfg)
    with caplog.at_level(logging.WARNING):
        runtime.self_check()
    assert "broken/agent" in caplog.text
