"""Unit tests for the per-turn resolved-pipeline-config tracer
(``app/llm_core/trace.py``) and its recording seams in
``split`` / ``resolver`` / ``health`` / ``concurrency`` / ``fallback``.

The bar these pin:
  (a) a stubbed turn (no network) populates the ``pipeline`` trace metadata with
      the resolved profile (name + weight) and, per executed step, the resolved
      tier (provider / model / endpoint / timeout_ms) + tier_served;
  (b) SECRETS never appear — the api-key *value* is nowhere in the emitted
      metadata nor in the startup full-config dump (only the env-var NAME is);
  (c) the fallback walker threads the actually-served tier index back;
  (d) the health-prune and concurrency-deprioritize trigger outcomes are recorded
      when those filters fire;
  (e) the recorders are a no-op with no active context (cheap request-path guard).

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

from app.llm_core import trace, split, resolver, health, concurrency
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
# NB: app.services.fallback is imported LAZILY (via importorskip) inside the one
# test that needs it — importing it at module top pulls in agents.models, which
# fails under the local pydantic-ai 0.2.4 vs pinned 1.x mismatch and would break
# collection of the whole (otherwise network-free) module. Mirrors test_split.py.


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
    return PipelineConfig(profiles=[
        NamedProfile(name="oss", weight=pct, steps=oss_steps),
        NamedProfile(name="managed", weight=100 - pct, steps=managed_steps),
    ])


@pytest.fixture(autouse=True)
def _fresh_ctx():
    trace.clear()
    yield
    trace.clear()


# ── (a) resolved profile + per-step tier populate the pipeline metadata ────────
def test_resolve_chain_populates_profile_and_step(monkeypatch):
    import asyncio
    trace.begin("oss")
    asyncio.run(split.resolve_chain("", Step.AGENT, _cfg(100)))

    md = trace.current().to_metadata()
    assert md["profile"] == {"name": "oss", "weight": 100}
    assert md["variant"] == "oss"
    step = md["steps"]["agent"]
    assert step["provider"] == "vllm"
    assert step["model"] == "gemma"
    assert step["endpoint"] == "http://oss:8020/v1"
    assert step["timeout_ms"] == 8000
    # Default served = primary (index 0) until a fallback overwrites it.
    assert step["tier_served"] == {"kind": "oss", "index": 0}
    assert [t["kind"] for t in step["chain"]] == ["oss", "managed"]


def test_flags_present_in_metadata():
    trace.begin("legacy")
    flags = trace.current().to_metadata()["flags"]
    assert set(flags) == {
        "llm_core_enabled", "profiles_enabled", "health_breaker_enabled",
        "health_poller_enabled", "concurrency_gauge_enabled",
    }


def test_resolver_seam_records_step(monkeypatch):
    """The non-fallback primary-tier seam records too."""
    monkeypatch.setattr(resolver.runtime, "get_pipeline", lambda: _cfg(100))
    trace.begin("oss")
    resolver.resolve_chain(Step.AGENT, "oss")
    md = trace.current().to_metadata()
    assert md["profile"]["name"] == "oss"
    assert md["steps"]["agent"]["model"] == "gemma"


# ── (b) secrets never leak ─────────────────────────────────────────────────────
def test_no_api_key_value_in_metadata():
    import asyncio
    trace.begin("oss")
    asyncio.run(split.resolve_chain("", Step.AGENT, _cfg(100)))
    blob = json.dumps(trace.current().to_metadata())
    assert _SECRET not in blob
    # The env-var NAME is fine to trace; the VALUE must never appear.
    assert "OSS_INFERENCE_API_KEY" not in blob  # not even the name (metadata omits it)


def test_no_secret_in_full_config_dump(caplog):
    import logging
    cfg = _cfg(50)
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


# ── (c) fallback walker threads the served tier index ──────────────────────────
def test_fallback_walker_records_served_index(monkeypatch):
    import asyncio

    fb = pytest.importorskip("app.services.fallback")
    oss = fb.Attempt("oss", object(), "gemma", "vllm", "http://oss:8020/v1", None)
    managed = fb.Attempt("managed", object(), "gpt-4.1", "openai", "managed", None)

    async def _chain(**kw):
        return [oss, managed]

    monkeypatch.setattr(fb, "_resolve_chain", _chain)

    trace.begin("oss")
    trace.record_step_chain(Step.AGENT, [oss, managed])  # seed primary=index0

    async def _run(a):
        if a.kind == "oss":
            raise TimeoutError("oss down")   # fallbackable -> swap to managed
        return "answer"

    out = asyncio.run(fb.execute_with_fallback(
        pipeline="chat", session_id="s", variant="oss", run=_run))
    assert out == "answer"
    served = trace.current().to_metadata()["steps"]["agent"]["tier_served"]
    assert served == {"kind": "managed", "index": 1}


# ── (d) trigger outcomes recorded when the filters fire ────────────────────────
def test_health_prune_trigger_recorded(monkeypatch):
    monkeypatch.setattr(health.settings, "health_breaker_enabled", True)
    reg = health.reset()
    # Trip the oss endpoint open at real monotonic time so it stays open (within
    # cooldown) when prune_unhealthy consults is_open with the real clock.
    for _ in range(reg.config.fail_threshold):
        reg.record_failure("http://oss:8020/v1")

    trace.begin("oss")
    tiers = [_oss_tier(), _managed_tier()]
    kept = health.prune_unhealthy(Step.AGENT, tiers)
    assert len(kept) == 1  # oss pruned
    h = trace.current().to_metadata()["steps"]["agent"]["triggers"]["health"]
    assert h["pruned"] == ["http://oss:8020/v1"]
    assert h["breaker_states"]["http://oss:8020/v1"] == "open"
    health.reset()


def test_concurrency_deprioritize_trigger_recorded(monkeypatch):
    import asyncio
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", True)

    async def _fake_gauge(url):
        return 99  # saturated

    monkeypatch.setattr(concurrency, "get_concurrency", _fake_gauge)
    gate = ConcurrencyGate(metrics_url="http://oss:8020/metrics", max_concurrency=10)

    trace.begin("oss")
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, gate))
    assert out[-1].provider is Provider.VLLM  # vLLM deprioritized to the back
    c = trace.current().to_metadata()["steps"]["agent"]["triggers"]["concurrency"]
    assert c["gauge"] == 99 and c["max_concurrency"] == 10 and c["deprioritized"] is True
    assert c["metrics_url"] == "http://oss:8020/metrics"


# ── (f) populate + COMPACT flat metadata keys (the path that lands) ───────────
def test_populate_sets_profile_and_per_step_primary_tiers(monkeypatch):
    """`populate` (the explicit, contextvar-independent path the request path uses)
    fills profile + each step's PRIMARY tier via the SYNC resolver.primary_tier —
    exactly as chat.py/voice.py call it (`resolver.primary_tier`)."""
    cfg = _cfg(100)
    monkeypatch.setattr(resolver.runtime, "get_pipeline", lambda: cfg)

    pt = trace.begin("oss")
    trace.populate(pt, cfg, resolver.primary_tier, "oss", (Step.AGENT, Step.MODERATION))
    md = pt.to_metadata()
    assert md["profile"] == {"name": "oss", "weight": 100}
    assert md["steps"]["agent"]["provider"] == "vllm"
    assert md["steps"]["agent"]["model"] == "gemma"
    assert md["steps"]["moderation"]["model"] == "gemma"
    assert len(md["flags"]) == 5


def test_compact_metadata_produces_short_flat_keys(monkeypatch):
    """The keys that actually land on the trace: `pipeline_profile`, `pipeline_flags`,
    and one `pc_<step>` per step — short flat strings, under the OTEL attribute cap."""
    cfg = _cfg(100)
    monkeypatch.setattr(resolver.runtime, "get_pipeline", lambda: cfg)
    pt = trace.begin("oss")
    trace.populate(pt, cfg, resolver.primary_tier, "oss", (Step.AGENT, Step.MODERATION))

    m = trace.compact_metadata(pt)
    assert m["pipeline_profile"] == "oss"
    # all-on flags (test env) -> the 5 short names
    assert set(m["pipeline_flags"].split(",")) == {
        "llm_core", "profiles", "health_breaker", "health_poller", "concurrency_gauge",
    }
    assert m["pc_agent"] == "vllm:gemma@http://oss:8020/v1#oss(8000ms)"
    assert m["pc_moderation"] == "vllm:gemma@http://oss:8020/v1#oss(8000ms)"
    # every value is a short string, safely under the ~128-256 char OTEL cap.
    assert all(isinstance(v, str) or v is None for v in m.values())
    assert all(v is None or len(v) < 120 for v in m.values())


def test_add_compact_metadata_merges_into_request_metadata_dict(monkeypatch):
    """The request path merges the compact keys into the SAME dict it already hands
    to propagate_attributes / VoiceTrace.metadata — existing keys preserved."""
    cfg = _cfg(100)
    monkeypatch.setattr(resolver.runtime, "get_pipeline", lambda: cfg)
    pt = trace.begin("oss")
    trace.populate(pt, cfg, resolver.primary_tier, "oss", (Step.AGENT,))

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


# ── (e) no active context -> recorders are a cheap no-op ───────────────────────
def test_recorders_noop_without_context():
    import asyncio
    trace.clear()
    assert trace.current() is None
    # None of these should raise or set anything.
    trace.record_profile("oss", 100)
    trace.record_step_chain(Step.AGENT, [])
    trace.record_served(Step.AGENT, "managed", 1)
    # split resolving with no context is still a clean no-op for tracing.
    asyncio.run(split.resolve_chain("", Step.AGENT, _cfg(100)))
    assert trace.current() is None
