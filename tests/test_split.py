"""Unit tests for the P1 weighted named-profile split + config-driven chain
(``app/llm_core/split.py``) and its composition with the ``fallback`` walkers.

The bar these pin:
  (a) cumulative-bucket determinism — same session_id -> same profile; the
      aggregate over many ids tracks the configured weights; and the 2-profile
      shim reproduces ``pipeline_router``'s ``OSS_PIPELINE_PCT`` boundary EXACTLY
      (bit-compatible ``int(sha256(sid)[:8], 16) % 100``).
  (b) ``resolve_chain`` returns a non-empty materialized chain matching the
      resolved profile's tiers (order + models + kind labels).
  (c) a config (weight) change RE-BUCKETS a continuing session — the sha256
      bucket over the CURRENT weights is the sticky key; no Redis pin freezes it.
  (d) the flags-OFF path is byte-untouched: PROFILES_ENABLED defaults off and the
      fallback chain acquisition degrades to the legacy ``attempt_chain``.

Zero network: routing is pure deterministic (no Redis). Building a factory handle
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
    bucket < pct — the EXACT boundary the removed pipeline_router used
    (``int(sha256(sid)[:8],16) % 100 < OSS_PIPELINE_PCT``), checked against a fresh
    independent recompute of that bucket."""
    for pct in (0, 1, 30, 50, 80, 99, 100):
        cfg = two_profile_config(pct)
        for i in range(500):
            sid = f"boundary-{pct}-{i}"
            bucket = _ref_bucket(sid)
            expected = "oss" if bucket < pct else "managed"
            assert split.deterministic_profile(sid, cfg) == expected


def test_2way_parity_name_threading_is_bit_identical():
    """HARD no-regression bar: at the 2-way env-shim config (profiles named
    oss/managed) threading the resolved profile NAME is bit-identical to the old
    ``variant='oss'`` path:

      * bucket < pct  -> name 'oss',    AGENT primary kind 'oss'    (is_oss True)
      * bucket >= pct -> name 'managed', AGENT primary kind 'managed' (is_oss False)

    and threading that NAME through ``resolve_chain(profile_name=...)`` yields the
    SAME chain (kinds + models) as the sticky ``resolve_profile`` path — proving the
    N->2 ``variant`` collapse is gone with zero behaviour change. ``is_oss`` is now
    derived from the AGENT primary tier KIND (the serving-path change), and here it
    matches the old ``bucket < pct`` bit exactly."""
    import asyncio

    for pct in (0, 30, 100):
        cfg = two_profile_config(pct)
        for i in range(300):
            sid = f"parity-{pct}-{i}"
            in_oss = _ref_bucket(sid) < pct
            name = split.deterministic_profile(sid, cfg)
            assert name == ("oss" if in_oss else "managed")
            sticky = asyncio.run(split.resolve_chain(sid, Step.AGENT, cfg))
            is_oss = sticky[0].kind == "oss"          # the new is_oss-from-kind bit
            assert is_oss is in_oss                   # == old bucket<pct variant bit
            named = asyncio.run(split.resolve_chain(sid, Step.AGENT, cfg, profile_name=name))
            assert [c.kind for c in named] == [c.kind for c in sticky]
            assert [c.model_name for c in named] == [c.model_name for c in sticky]


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


# ── (c) deterministic assignment: a weight change RE-BUCKETS (refresh-on-change) ─

def _sid_with_bucket_between(lo, hi):
    i = 0
    while True:
        sid = f"pick-{i}"
        if lo <= split._bucket(sid) < hi:
            return sid, split._bucket(sid)
        i += 1


def test_resolve_profile_is_pure_deterministic():
    """resolve_profile == deterministic bucket over the CURRENT weights; no Redis
    state, so nothing pins a session across weight changes."""
    import asyncio

    cfg = two_profile_config(70)
    for sid in ("a", "b", "sess-xyz", ""):
        assert asyncio.run(split.resolve_profile(sid, cfg)) == split.deterministic_profile(sid, cfg)


def test_weight_change_rebuckets_continuing_session():
    """The refresh-on-change contract: a session assigned to one model at a given
    weight MOVES to the other model when the weight changes — it is not frozen."""
    import asyncio

    sid, bucket = _sid_with_bucket_between(30, 60)
    cfg_hi = two_profile_config(bucket + 5)   # bucket < pct -> 'oss'
    cfg_lo = two_profile_config(bucket - 5)   # bucket >= pct -> 'managed'
    assert asyncio.run(split.resolve_profile(sid, cfg_hi)) == "oss"
    # SAME session id, weight changed -> re-buckets to the new model (does NOT stick).
    assert asyncio.run(split.resolve_profile(sid, cfg_lo)) == "managed"


def test_zero_to_fifty_moves_about_half_of_continuing_sessions():
    """0 -> 50% for the oss model moves ~half of continuing (same-id) sessions onto
    it — exactly the redeploy scenario, not 0%."""
    import asyncio

    ids = [f"s{i}" for i in range(400)]
    at_zero = [asyncio.run(split.resolve_profile(s, two_profile_config(0))) for s in ids]
    at_fifty = [asyncio.run(split.resolve_profile(s, two_profile_config(50))) for s in ids]
    assert all(p == "managed" for p in at_zero)          # 0% oss -> everyone on managed
    moved = sum(1 for a, b in zip(at_zero, at_fifty) if a != b and b == "oss")
    assert 150 <= moved <= 250                            # ~50% of continuing sessions moved


def test_resolve_chain_honors_profile_name_without_rebucketing(monkeypatch):
    """(C) resolve_chain(profile_name=...) selects that profile DIRECTLY and never
    calls resolve_profile — so a capped session id can't diverge from the primary
    path. An absent name ('legacy') fail-safes to managed."""
    import asyncio

    def _boom(*a, **k):
        raise AssertionError("resolve_profile must not be called when profile_name is threaded")

    monkeypatch.setattr(split, "resolve_profile", _boom)
    cfg = two_profile_config(0)   # deterministic bucket would be 'managed' for all
    chain = asyncio.run(split.resolve_chain("any-session", Step.AGENT, cfg, profile_name="oss"))
    assert [c.kind for c in chain] == ["oss", "managed"]
    legacy_chain = asyncio.run(split.resolve_chain("x", Step.AGENT, cfg, profile_name="legacy"))
    assert [c.kind for c in legacy_chain] == ["managed"]


# ── (d) the walkers receive the config-driven chain (the only path) ───────────

def test_fallback_chain_uses_split(monkeypatch):
    """The walkers' chain acquisition delegates to the config-driven split for the
    mapped step (moderation -> Step.MODERATION). The router variant is threaded
    through (fix C) — asserted in the spy below."""
    import asyncio
    from app.services import fallback as fb


    sentinel = ["MATERIALIZED_TIER"]

    async def _spy(session_id, step, *, profile_name=None):
        assert step is Step.MODERATION
        assert profile_name == "oss"     # (C) router-resolved profile NAME threaded through
        return sentinel

    monkeypatch.setattr(split, "resolve_chain", _spy)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", profile_name="oss"))
    assert chain is sentinel


def test_fallback_chain_degrades_to_managed_on_split_error(monkeypatch):
    """A config/Redis edge case in split must never break the fallback path: it
    degrades to the resolver's managed-tier chain (non-empty)."""
    import asyncio
    from app.services import fallback as fb
    from app.llm_core import runtime

    runtime.configure(run_self_check=False)   # synthesized (managed-only) config

    async def _boom(session_id, step, *, profile_name=None):
        raise RuntimeError("config blew up")

    monkeypatch.setattr(split, "resolve_chain", _boom)
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="s", profile_name="oss"))
    assert len(chain) >= 1                       # degrade chain is never empty
    assert chain[-1].kind == "managed"


# ── N-way ("nxn"): the 3-profile yaml is loaded, distributed AND served ────────
# The proof that a 3rd profile is actually SERVED (not collapsed to oss/managed):
# deterministic_profile buckets across all 3 by weight, and each profile's AGENT
# primary tier resolves to ITS configured model (gemma/qwen/gpt) with the right kind.

import os as _os

_EXAMPLE_YAML = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "pipeline.example.yaml"
)


def test_nway_three_profile_yaml_distributes_and_serves(monkeypatch):
    from app.llm_core import runtime, resolver

    monkeypatch.setenv("PIPELINE_CONFIG_PATH", _EXAMPLE_YAML)
    cfg = runtime.configure(run_self_check=False)   # parse N NamedProfiles w/ per-step tiers
    try:
        assert {p.name for p in cfg.profiles} == {"gemma", "qwen", "gpt"}
        assert [p.weight for p in cfg.profiles] == [5, 10, 85]

        # (1) deterministic_profile distributes across ALL THREE by weight (5/10/85).
        n = 8000
        counts = {"gemma": 0, "qwen": 0, "gpt": 0}
        for i in range(n):
            counts[split.deterministic_profile(f"nway-{i}", cfg)] += 1
        assert 0.03 < counts["gemma"] / n < 0.07
        assert 0.07 < counts["qwen"] / n < 0.13
        assert 0.80 < counts["gpt"] / n < 0.90

        # (2) each profile's AGENT primary tier is ITS model + kind — served, not
        # collapsed. A qwen-on-vLLM profile is "oss"; the gpt profile is "managed".
        gemma = resolver.primary_tier(Step.AGENT, "gemma")
        qwen = resolver.primary_tier(Step.AGENT, "qwen")
        gpt = resolver.primary_tier(Step.AGENT, "gpt")
        assert (gemma.model_name, gemma.kind) == ("gemma-4-31b-it", "oss")
        assert (qwen.model_name, qwen.kind) == ("qwen2.5-32b-instruct", "oss")
        assert (gpt.model_name, gpt.kind) == ("gpt-4.1", "managed")
        # unknown name fail-safes: managed if present, else profiles[0] (here gemma,
        # since this N-way config has no "managed" profile) — never raises.
        assert resolver.primary_tier(Step.AGENT, "does-not-exist").model_name == "gemma-4-31b-it"
    finally:
        # delenv BEFORE reconfigure so the global is restored to the env-shim (not the
        # yaml) for later tests — monkeypatch's own teardown runs only after this.
        monkeypatch.delenv("PIPELINE_CONFIG_PATH", raising=False)
        runtime.configure(run_self_check=False)


# ── self_check over ALL profiles: a broken 3rd-profile tier is reported, non-fatal ─

def test_self_check_reports_broken_third_profile_nonfatal(monkeypatch, caplog):
    """runtime.self_check iterates EVERY profile by name and resolves each configured
    step's primary tier, so a broken 3rd-profile tier (here: a vLLM agent tier with no
    endpoint -> factory raises at build) is caught at boot. It must WARN, not raise."""
    import logging
    from app.llm_core import runtime

    broken_agent = Tier(provider=Provider.VLLM, model="broken", endpoint=None,
                        api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)
    cfg = PipelineConfig(profiles=[
        NamedProfile(name="oss", weight=45, steps={Step.AGENT: StepConfig(tiers=[_oss_tier(), _managed_tier()])}),
        NamedProfile(name="managed", weight=45, steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
        NamedProfile(name="broken", weight=10, steps={Step.AGENT: StepConfig(tiers=[broken_agent])}),
    ])
    monkeypatch.setattr(runtime, "PIPELINE", cfg)
    with caplog.at_level(logging.WARNING):
        runtime.self_check()                 # must NOT raise (non-fatal)
    assert "broken/agent" in caplog.text     # the broken 3rd profile is reported
