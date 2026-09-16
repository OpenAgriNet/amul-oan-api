"""Unit tests for the per-turn resolved-pipeline-config tracer
(``app/llm_core/trace.py``) and its recording seams in ``split`` / ``execution``.

The bar these pin:
  (a) a stubbed turn (no network) populates the ``pipeline`` trace metadata with
      the resolved profile (name + weight) and, per executed step, the resolved
      tier (provider / model / endpoint / timeout_ms) + tier_served;
  (b) SECRETS never appear — the api-key *value* is nowhere in the emitted
      metadata nor in the startup full-config dump (only the env-var NAME is);
  (c) the fallback walker threads the actually-served tier index back;
  (d) the recorders are a no-op with no active context (cheap request-path guard).

Zero network: ``session_id=""`` avoids Redis; building a factory handle is lazy
(no model call). A KNOWN-SECRET api key is placed in the env and then asserted
absent from every emitted structure. These tests deliberately avoid
``app.services.translation`` / ``agents.tools`` (pydantic-ai version mismatch).
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
# A sentinel we assert never leaks into any trace metadata / config dump.
_SECRET = "SUPER-SECRET-KEY-VALUE-do-not-leak"
os.environ["OSS_INFERENCE_API_KEY"] = _SECRET

import json

import pytest

from app.llm_core import ExecutionContext, trace
from app.llm_core.config_model import (
    ConcurrencyGate,
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
    Triggers,
)
# NB: app.llm_core.execution is imported lazily inside the one test that needs it
# to keep this otherwise network-free tracer module narrowly scoped.


def _oss_tier(model="gemma"):
    return Tier(provider=Provider.VLLM, model=model, endpoint="http://oss:8020/v1",
                api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)


def _managed_tier(model="gpt-4.1"):
    return Tier(provider=Provider.OPENAI, model=model, api_key_env="OPENAI_API_KEY",
                timeout_ms=20000)


def _cfg(pct=100, triggers=None) -> PipelineConfig:
    oss_steps = {
        Step.AGENT: StepConfig(tiers=[_oss_tier(), _managed_tier()], triggers=triggers or Triggers()),
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
        fallback_enabled=True,
    )


def test_flags_present_in_metadata():
    flags = trace.begin("legacy").to_metadata()["flags"]
    # P4 removed the llm_core/profiles kill-switches; only the operational triggers remain.
    assert set(flags) == {
        "health_breaker_enabled", "health_poller_enabled",
    }


# ── (b) secrets never leak ─────────────────────────────────────────────────────
def test_no_api_key_value_in_metadata():
    pt = ExecutionContext("", _cfg(100), "oss").begin_trace()
    blob = json.dumps(pt.to_metadata())
    assert _SECRET not in blob
    # The env-var NAME is fine to trace; the VALUE must never appear.
    assert "OSS_INFERENCE_API_KEY" not in blob  # not even the name (metadata omits it)


def test_no_secret_in_full_config_dump(caplog):
    import logging
    overflow = Tier(provider=Provider.VLLM, model="overflow", endpoint="http://overflow/v1")
    cfg = _cfg(50, Triggers(concurrency_gate=ConcurrencyGate(
        metrics_url="http://metrics", overflow_tier=overflow
    )))
    with caplog.at_level(logging.INFO):
        trace.log_full_config(cfg)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "llm_core.full_config" in text
    assert _SECRET not in text
    # api_key_env NAME is dumped (not a secret); the value is not.
    assert "OSS_INFERENCE_API_KEY" in text
    # all profiles + steps present
    dumped = trace.config_to_dict(cfg)
    assert {p["name"] for p in dumped["profiles"]} == {"oss", "managed"}
    assert "agent" in dumped["profiles"][0]["steps"]
    gate = dumped["profiles"][0]["steps"]["agent"]["triggers"]["concurrency_gate"]
    assert gate["overflow_tier"]["model"] == "overflow"


# ── (c) fallback walker threads the served tier index ──────────────────────────
def test_fallback_walker_records_served_index(monkeypatch, materialized_tier):
    import asyncio

    fb = pytest.importorskip("app.llm_core.execution")
    oss = materialized_tier("oss", object(), model_name="gemma")
    managed = materialized_tier("managed", object(), model_name="gpt-4.1")

    pt = trace.begin("oss")
    trace.set_step_primary(pt, Step.AGENT, oss)

    async def _run(a):
        if a.kind == "oss":
            raise TimeoutError("oss down")   # fallbackable -> swap to managed
        return "answer"

    out = asyncio.run(fb.execute_with_fallback(
        step=Step.AGENT, session_id="s", run=_run, chain=[oss, managed],
        trace_state=pt,
    ))
    assert out == "answer"
    served = pt.to_metadata()["steps"]["agent"]["tier_served"]
    assert served == {
        "route": "openai:gpt-4.1",
        "index": 1,
    }


# ── (d) populate + COMPACT flat metadata keys (the path that lands) ───────────
def test_execution_context_separates_configured_and_served_tiers():
    import asyncio

    cfg = _cfg(100)
    execution = ExecutionContext("", cfg, "oss")
    pt = execution.begin_trace()
    md = pt.to_metadata()
    assert md["profile"] == {"name": "oss", "weight": 100}
    assert md["steps"]["agent"]["provider"] == "vllm"
    assert md["steps"]["agent"]["model"] == "gemma"
    assert md["steps"]["moderation"]["model"] == "gemma"
    assert trace.served_summary(pt) is None
    asyncio.run(execution.run_adapter(Step.AGENT, lambda target: _result(target)))
    assert trace.served_summary(pt) == "agent=vllm:gemma[0]"


async def _result(target):
    return target.model_name


def test_compact_metadata_produces_short_flat_keys():
    """The keys that actually land on the trace: `pipeline_profile`, `pipeline_flags`,
    and one `pc_<step>` per step — short flat strings, under the OTEL attribute cap."""
    cfg = _cfg(100)
    pt = ExecutionContext("", cfg, "oss").begin_trace()

    m = trace.compact_metadata(pt)
    assert m["pipeline_profile"] == "oss"
    # P4 removed the llm_core/profiles kill-switches; concurrency is config-driven.
    assert set(m["pipeline_flags"].split(",")) == {
        "health_breaker", "health_poller",
    }
    assert m["pc_agent"] == "vllm:gemma@http://oss:8020/v1#oss(8000ms)"
    assert m["pc_moderation"] == "vllm:gemma@http://oss:8020/v1#oss(8000ms)"
    # every value is a short string, safely under the ~128-256 char OTEL cap.
    assert all(isinstance(v, str) or v is None for v in m.values())
    assert all(v is None or len(v) < 120 for v in m.values())


def test_compact_metadata_hard_caps_long_values():
    """A long configured endpoint/model must NOT push a pc_<step> value over the
    OTEL attribute cap (which would silently DROP the key). Values are truncated to
    _ATTR_CAP; the full untruncated config is always in the boot full_config dump."""
    from app.llm_core.config_model import (
        NamedProfile, PipelineConfig, Provider, StepConfig, Tier,
    )

    long_ep = "http://" + "x" * 500 + ":8020/v1"
    cfg = PipelineConfig(profiles=[NamedProfile(name="oss", weight=100, steps={
        Step.AGENT: StepConfig(tiers=[Tier(
            provider=Provider.VLLM, model="m" * 300, endpoint=long_ep,
            api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)]),
    })])
    pt = ExecutionContext("", cfg, "oss").begin_trace()

    m = trace.compact_metadata(pt)
    assert len(m["pc_agent"]) == trace._ATTR_CAP           # truncated -> the key still LANDS
    assert all(v is None or len(v) <= trace._ATTR_CAP for v in m.values())


def test_add_compact_metadata_merges_into_request_metadata_dict():
    """The request path merges the compact keys into the SAME dict it already hands
    to propagate_attributes / VoiceTrace.metadata — existing keys preserved."""
    cfg = _cfg(100)
    pt = ExecutionContext("", cfg, "oss").begin_trace()

    langfuse_metadata = {"pipeline": "translation", "variant": "oss"}  # pre-existing keys
    trace.add_compact_metadata(pt, langfuse_metadata)

    assert langfuse_metadata["pipeline"] == "translation"   # existing key untouched
    assert langfuse_metadata["variant"] == "oss"
    assert langfuse_metadata["pipeline_profile"] == "oss"
    assert langfuse_metadata["pc_agent"].startswith("vllm:gemma@")


def test_compact_metadata_empty_pt_is_empty_dict():
    """A None/empty pt yields an empty dict (never raises) — nothing to add."""
    assert trace.compact_metadata(None) == {}
    d = {"keep": 1}
    trace.add_compact_metadata(None, d)
    assert d == {"keep": 1}


def test_no_update_current_trace_symbol_remains():
    """Guard: the dead SDK-incompatible machinery is gone (no update_current_trace,
    no emit_to_trace) — the module must not reference them again."""
    assert not hasattr(trace, "emit_to_trace")
    assert not hasattr(trace, "_get_langfuse_client")
    src = __import__("inspect").getsource(trace)
    assert "update_current_trace(" not in src   # no CALL to the missing SDK method
    assert "import get_client" not in src
