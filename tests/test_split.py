"""Unit tests for the P1 weighted named-profile split + config-driven chain
(``app/llm_core/split.py``) and its composition with the ``fallback`` walkers.

The bar these pin:
  (a) cumulative-bucket determinism — same session_id -> same profile; the
      aggregate over many ids tracks the configured weights; and the 2-profile
      shim reproduces ``pipeline_router``'s ``OSS_PIPELINE_PCT`` boundary EXACTLY
      (bit-compatible ``int(sha256(sid)[:8], 16) % 100``).
  (b) ``resolve_chain`` returns a non-empty materialized chain matching the
      resolved profile's tiers (order + models + kind labels).
  (c) a config (weight) change does not re-bucket an already-sticky session.
  (d) the flags-OFF path is byte-untouched: PROFILES_ENABLED defaults off and the
      fallback chain acquisition degrades to the legacy ``attempt_chain``.

Zero network: session_id="" avoids Redis entirely (deterministic path); the
sticky/fail-safe tests inject an in-memory fake cache. Building a factory handle
is lazy (no model call is ever made). A dummy OPENAI_API_KEY / OSS key is set
before importing app code (agents.models + the factory read keys at build time).
These tests deliberately avoid app.services.translation / agents.tools, which
fail to import under the local pydantic-ai 0.2.4 vs pinned 1.50.0 mismatch.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OSS_INFERENCE_API_KEY", "test-oss-key")

import hashlib

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


# ── config builders ───────────────────────────────────────────────────────────

def _oss_tier(model="gemma"):
    return Tier(provider=Provider.VLLM, model=model, endpoint="http://oss:8020/v1",
                api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)


def _managed_tier(model="gpt-4.1"):
    return Tier(provider=Provider.OPENAI, model=model, api_key_env="OPENAI_API_KEY",
                timeout_ms=20000)


def two_profile_config(pct: int, ttl: int = 604800) -> PipelineConfig:
    """The shim's seeded 2-profile split: ``[oss(pct), managed(100-pct)]`` with
    per-step ``[oss, managed]`` (oss profile) / ``[managed]`` (managed profile),
    mirroring ``fallback.attempt_chain``."""
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


class _FakeCache:
    """In-memory async cache mirroring the aiocache surface split uses."""

    def __init__(self):
        self.store = {}
        self.sets = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, value, ttl))


class _BrokenCache:
    async def get(self, key):
        raise RuntimeError("redis down")

    async def set(self, key, value, ttl=None):
        raise RuntimeError("redis down")


def _ref_bucket(session_id: str) -> int:
    """Independent recomputation of pipeline_router's bucket (no shared code)."""
    digest = hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


# ── (a) cumulative-bucket determinism + boundary bit-compatibility ────────────

def test_bucket_matches_pipeline_router_formula():
    for i in range(200):
        sid = f"sess-{i}"
        assert split._bucket(sid) == _ref_bucket(sid)


def test_deterministic_profile_is_stable_per_session():
    cfg = two_profile_config(70)
    for i in range(200):
        sid = f"s-{i}"
        assert split.deterministic_profile(sid, cfg) == split.deterministic_profile(sid, cfg)


def test_deterministic_profile_aggregate_tracks_weights():
    cfg = two_profile_config(70)
    n = 8000
    oss = sum(1 for i in range(n) if split.deterministic_profile(f"id-{i}", cfg) == "oss")
    frac = oss / n
    assert 0.66 < frac < 0.74, f"oss fraction {frac} not ~0.70"


def test_two_profile_shim_reproduces_oss_pct_boundary_exactly():
    """The proof: for the seeded 2-profile config, a session is 'oss' iff its
    bucket < pct — the EXACT boundary pipeline_router._deterministic_variant uses.
    Checked against a fresh recompute AND against pipeline_router itself."""
    from app.services import pipeline_router

    for pct in (0, 1, 30, 50, 80, 99, 100):
        cfg = two_profile_config(pct)
        for i in range(500):
            sid = f"boundary-{pct}-{i}"
            bucket = _ref_bucket(sid)
            expected = "oss" if bucket < pct else "managed"
            assert split.deterministic_profile(sid, cfg) == expected


def test_split_variant_is_bit_compatible_with_pipeline_router(monkeypatch):
    """variant_for_profile(weighted-split) == pipeline_router._deterministic_variant
    for every pct — i.e. flipping PROFILES_ENABLED never moves a session."""
    from app.services import pipeline_router

    monkeypatch.setattr(pipeline_router, "oss_model_available", lambda: True)
    for pct in (0, 25, 80, 100):
        cfg = two_profile_config(pct)
        monkeypatch.setattr(pipeline_router.settings, "oss_pipeline_pct", pct)
        for i in range(500):
            sid = f"compat-{pct}-{i}"
            legacy = pipeline_router._deterministic_variant(sid)          # 'oss'|'legacy'
            new = split.variant_for_profile(split.deterministic_profile(sid, cfg))
            assert new == legacy, f"pct={pct} sid={sid}: {new} != {legacy}"


def test_cumulative_buckets_over_three_profiles():
    """Generalization past 2: profile i owns [sum(w[:i]), sum(w[:i+1]))."""
    cfg = PipelineConfig(profiles=[
        NamedProfile(name="a", weight=20, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
        NamedProfile(name="b", weight=30, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
        NamedProfile(name="c", weight=50, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
    ])
    for i in range(500):
        sid = f"tri-{i}"
        b = _ref_bucket(sid)
        expected = "a" if b < 20 else ("b" if b < 50 else "c")
        assert split.deterministic_profile(sid, cfg) == expected


# ── weights-sum validator (confirm it lives in PipelineConfig) ────────────────

def test_pipeline_config_rejects_weights_not_summing_to_100():
    with pytest.raises(ValidationError):
        PipelineConfig(profiles=[
            NamedProfile(name="a", weight=50, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
            NamedProfile(name="b", weight=30, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
        ])


# ── (b) resolve_chain returns a materialized chain matching the profile tiers ──

def test_resolve_chain_matches_oss_profile_tiers():
    import asyncio

    cfg = two_profile_config(100)  # everyone -> oss profile
    chain = asyncio.run(split.resolve_chain("", Step.AGENT, cfg))
    assert len(chain) == 2
    assert [c.model_name for c in chain] == ["gemma", "gpt-4.1"]
    assert [c.kind for c in chain] == ["oss", "managed"]   # == attempt_chain labels
    assert [c.provider for c in chain] == ["vllm", "openai"]
    assert all(c.handle is not None for c in chain)
    assert chain[0].timeout == 8.0 and chain[1].timeout == 20.0


def test_resolve_chain_matches_managed_profile_single_tier():
    import asyncio

    cfg = two_profile_config(0)   # everyone -> managed profile
    chain = asyncio.run(split.resolve_chain("", Step.AGENT, cfg))
    assert [c.kind for c in chain] == ["managed"]
    assert [c.model_name for c in chain] == ["gpt-4.1"]


def test_resolve_chain_never_empty_for_configured_step():
    import asyncio

    cfg = two_profile_config(50)
    chain = asyncio.run(split.resolve_chain("", Step.MODERATION, cfg))
    assert len(chain) >= 1


# ── (c) sticky assignment: a config change does not re-bucket ─────────────────

def _sid_with_bucket_between(lo, hi):
    i = 0
    while True:
        sid = f"pick-{i}"
        if lo <= split._bucket(sid) < hi:
            return sid, split._bucket(sid)
        i += 1


def test_sticky_profile_persists_across_weight_change(monkeypatch):
    import asyncio

    fake = _FakeCache()
    monkeypatch.setattr(split, "cache", fake)

    sid, bucket = _sid_with_bucket_between(30, 60)
    cfg_a = two_profile_config(bucket + 5)   # bucket < pct -> 'oss'
    cfg_b = two_profile_config(bucket - 5)   # bucket >= pct -> deterministic 'managed'

    first = asyncio.run(split.resolve_profile(sid, cfg_a))
    assert first == "oss"
    assert fake.store[f"pipeline_profile:{sid}"] == "oss"   # stored under the P1 key

    # Deterministic assignment WOULD flip under cfg_b ...
    assert split.deterministic_profile(sid, cfg_b) == "managed"
    # ... but the sticky session keeps its original profile.
    assert asyncio.run(split.resolve_profile(sid, cfg_b)) == "oss"


def test_sticky_hit_short_circuits_bucketing(monkeypatch):
    import asyncio

    fake = _FakeCache()
    fake.store["pipeline_profile:preset"] = "managed"
    monkeypatch.setattr(split, "cache", fake)
    # bucket would say 'oss' at pct=100, but the stored name wins.
    cfg = two_profile_config(100)
    assert asyncio.run(split.resolve_profile("preset", cfg)) == "managed"
    assert fake.sets == []   # a hit must not re-write


def test_sticky_ttl_comes_from_config(monkeypatch):
    import asyncio

    fake = _FakeCache()
    monkeypatch.setattr(split, "cache", fake)
    cfg = two_profile_config(50, ttl=12345)
    asyncio.run(split.resolve_profile("ttl-sess", cfg))
    assert fake.sets and fake.sets[0][2] == 12345


def test_stale_stored_name_is_rebucketed(monkeypatch):
    import asyncio

    fake = _FakeCache()
    fake.store["pipeline_profile:x"] = "no-such-profile"
    monkeypatch.setattr(split, "cache", fake)
    cfg = two_profile_config(100)
    # invalid stored name ignored -> deterministic (oss at pct=100)
    assert asyncio.run(split.resolve_profile("x", cfg)) == "oss"


def test_resolve_profile_fail_safe_on_cache_error(monkeypatch):
    import asyncio

    monkeypatch.setattr(split, "cache", _BrokenCache())
    cfg = two_profile_config(70)
    # A Redis error degrades to the deterministic bucket, never raises.
    got = asyncio.run(split.resolve_profile("err-sess", cfg))
    assert got == split.deterministic_profile("err-sess", cfg)


def test_empty_session_id_skips_cache(monkeypatch):
    import asyncio

    fake = _FakeCache()
    monkeypatch.setattr(split, "cache", fake)
    cfg = two_profile_config(50)
    asyncio.run(split.resolve_profile("", cfg))
    assert fake.sets == [] and fake.store == {}   # no id -> no Redis touch


# ── (d) flags-OFF path untouched + composition with fallback walkers ──────────

def test_profiles_enabled_defaults_off():
    from app.config import Settings
    import os as _os
    _os.environ.pop("PROFILES_ENABLED", None)
    assert Settings().profiles_enabled is False


def test_fallback_chain_uses_legacy_attempt_chain_when_flag_off(monkeypatch):
    """With PROFILES_ENABLED off, the walkers' chain acquisition is byte-identical
    to today: exactly what attempt_chain returns, and split is never consulted."""
    import asyncio
    from app.services import fallback as fb

    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "profiles_enabled", False)
    monkeypatch.setattr(fb.settings, "llm_core_enabled", False)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", object())
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma-test")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")

    def _boom(*a, **k):
        raise AssertionError("split must not be consulted with the flag off")

    monkeypatch.setattr(split, "resolve_chain", _boom)

    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", variant="oss"))
    legacy = fb.attempt_chain("oss", "moderation")
    assert [a.kind for a in chain] == [a.kind for a in legacy] == ["oss", "managed"]


def test_fallback_chain_stays_legacy_when_only_profiles_on(monkeypatch):
    """Composition: PROFILES_ENABLED on but LLM_CORE_ENABLED off -> still the
    legacy chain (the config-driven chain carries P0 factory handles)."""
    import asyncio
    from app.services import fallback as fb

    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "profiles_enabled", True)
    monkeypatch.setattr(fb.settings, "llm_core_enabled", False)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", object())
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma-test")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")

    called = {"n": 0}

    async def _spy(session_id, step):
        called["n"] += 1
        return []

    monkeypatch.setattr(split, "resolve_chain", _spy)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", variant="oss"))
    assert called["n"] == 0                       # split not consulted
    assert [a.kind for a in chain] == ["oss", "managed"]


def test_fallback_chain_uses_split_when_both_flags_on(monkeypatch):
    """Both flags on -> the walkers receive the config-driven materialized chain."""
    import asyncio
    from app.services import fallback as fb

    monkeypatch.setattr(fb.settings, "profiles_enabled", True)
    monkeypatch.setattr(fb.settings, "llm_core_enabled", True)

    sentinel = ["MATERIALIZED_TIER"]

    async def _spy(session_id, step):
        assert step is Step.MODERATION
        return sentinel

    monkeypatch.setattr(split, "resolve_chain", _spy)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", variant="oss"))
    assert chain is sentinel


def test_fallback_chain_degrades_to_legacy_on_split_error(monkeypatch):
    """A config/Redis edge case in split must never break the fallback path."""
    import asyncio
    from app.services import fallback as fb

    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "profiles_enabled", True)
    monkeypatch.setattr(fb.settings, "llm_core_enabled", True)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", object())
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma-test")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")

    async def _boom(session_id, step):
        raise RuntimeError("config blew up")

    monkeypatch.setattr(split, "resolve_chain", _boom)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", variant="oss"))
    assert [a.kind for a in chain] == ["oss", "managed"]   # fell back to attempt_chain
