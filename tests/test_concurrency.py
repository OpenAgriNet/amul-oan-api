"""Unit tests for the P3 concurrency-gauge REORDER filter
(``app/llm_core/concurrency.py``) and its composition with the P2 health prune in
``split.resolve_chain``.

The bar these pin:
  (a) reorder = DEPRIORITIZE, not drop — a saturated (gauge >= threshold) vLLM
      tier is moved BEHIND the managed tier; below-threshold leaves order
      unchanged; the chain is never emptied and no tier is dropped.
  (b) fail-open — unreadable metrics (gauge ``None``) leave order unchanged (NOT a
      forced flip to managed); a step without a configured gate is untouched;
      flags-off is identity.
  (c) the metrics scrape sums ``num_requests_running + num_requests_waiting`` and
      is Redis-cached; a fetch failure -> ``None`` (the fail-open signal).
  (d) composition: ``resolve_chain`` runs health-prune THEN concurrency-reorder,
      so a DOWN tier pruned by P2 is gone and can never be reordered to the front,
      while a saturated-but-UP vLLM primary is deprioritized (not dropped).
  (e) inverted-semantics note (plan §2): "primary" is only tier index 0; the
      filter reproduces bh's "flip to closed-source when gemma busy" as a pure
      reorder — no separate inversion.

Zero network: the gauge value is injected (``get_concurrency`` monkeypatched) for
the reorder/composition tests; the scrape test stubs ``httpx.AsyncClient`` + an
in-memory cache. Dummy OPENAI/OSS keys are set before importing app code (the
factory reads keys at materialize time).
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OSS_INFERENCE_API_KEY", "test-oss-key")

import asyncio

import pytest

from app.llm_core import concurrency, health, split
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

OSS_EP = "http://oss:8020/v1"
METRICS_URL = "http://oss:8020/metrics"


def _oss_tier(ep=OSS_EP):
    return Tier(provider=Provider.VLLM, model="gemma", endpoint=ep,
                api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)


def _managed_tier():
    return Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY",
                timeout_ms=20000)


def _gate(max_concurrency=10):
    return ConcurrencyGate(metrics_url=METRICS_URL, max_concurrency=max_concurrency)


@pytest.fixture
def gauge_on(monkeypatch):
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", True)
    return monkeypatch


def _inject_gauge(monkeypatch, value):
    """Replace the (Redis+HTTP) gauge read with a synthetic value / None."""
    async def _fake(metrics_url):
        return value
    monkeypatch.setattr(concurrency, "get_concurrency", _fake)


# ── (a) reorder = deprioritize, not drop ──────────────────────────────────────

def test_saturated_vllm_moved_behind_managed(gauge_on):
    _inject_gauge(gauge_on, 12)                                     # >= 10 -> saturated
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert [t.provider for t in out] == [Provider.OPENAI, Provider.VLLM]  # managed first
    assert len(out) == 2                                            # nothing dropped


def test_at_threshold_is_saturated(gauge_on):
    _inject_gauge(gauge_on, 10)                                     # gauge == threshold
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert [t.provider for t in out] == [Provider.OPENAI, Provider.VLLM]


def test_below_threshold_leaves_order_unchanged(gauge_on):
    _inject_gauge(gauge_on, 3)                                      # < 10 -> not saturated
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert out is tiers                                             # identity, primary stays primary


def test_reorder_is_stable_across_multiple_managed_tiers(gauge_on):
    """Deprioritize moves vLLM to the back but preserves the relative order of the
    non-vLLM tiers (stable partition)."""
    _inject_gauge(gauge_on, 50)
    a, b = _managed_tier(), Tier(provider=Provider.ANTHROPIC, model="claude",
                                 api_key_env="ANTHROPIC_API_KEY", timeout_ms=20000)
    tiers = [_oss_tier(), a, b]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert out == [a, b, tiers[0]]                                 # both managed ahead, vLLM last


def test_never_empty_all_vllm_chain(gauge_on):
    """Saturated but the ONLY tiers are vLLM -> nothing to reorder behind; the
    chain is returned unchanged (never emptied, tier not dropped)."""
    _inject_gauge(gauge_on, 99)
    tiers = [_oss_tier(), _oss_tier("http://oss2:8020/v1")]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert out is tiers
    assert len(out) == 2


# ── (b) fail-open / no-gate / flag-off = identity ─────────────────────────────

def test_unreadable_metrics_fail_open(gauge_on):
    _inject_gauge(gauge_on, None)                                  # metrics unreadable
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert out is tiers                                            # order unchanged, NOT flipped


def test_step_without_gate_untouched(gauge_on):
    _inject_gauge(gauge_on, 99)                                    # would saturate, but...
    tiers = [_oss_tier(), _managed_tier()]
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, None))  # no gate
    assert out is tiers


def test_flag_off_is_identity(monkeypatch):
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", False)
    _inject_gauge(monkeypatch, 99)                                 # saturated in the registry...
    tiers = [_oss_tier(), _managed_tier()]
    # ... but the filter is inert while the flag is off.
    out = asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, tiers, _gate(10)))
    assert out is tiers


def test_empty_tiers_returned_as_is(gauge_on):
    _inject_gauge(gauge_on, 99)
    assert asyncio.run(concurrency.reprioritize_by_load(Step.AGENT, [], _gate(10))) == []


# ── (c) the metrics scrape (stubbed httpx + in-memory cache, zero network) ─────

_METRICS_BODY = """# HELP vllm:num_requests_running Number of running requests.
vllm:num_requests_running{model_name="gemma"} 7.0
vllm:num_requests_waiting{model_name="gemma"} 5.0
vllm:num_requests_swapped{model_name="gemma"} 3.0
some_other_metric{foo="bar"} 100.0
"""


class _FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self._status = status

    def raise_for_status(self):
        if self._status != 200:
            raise RuntimeError(f"HTTP {self._status}")


class _FakeAsyncClient:
    def __init__(self, resp=None, exc=None, **kwargs):
        self._resp = resp
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if self._exc is not None:
            raise self._exc
        return self._resp


class _FakeCache:
    def __init__(self):
        self.store = {}
        self.sets = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, value, ttl))


def _patch_httpx(monkeypatch, resp=None, exc=None):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: _FakeAsyncClient(resp=resp, exc=exc))


def test_fetch_sums_running_and_waiting(monkeypatch):
    _patch_httpx(monkeypatch, resp=_FakeResp(_METRICS_BODY))
    total = asyncio.run(concurrency._fetch_concurrency(METRICS_URL))
    assert total == 12                                             # 7 running + 5 waiting (swapped ignored)


def test_fetch_failure_returns_none_fail_open(monkeypatch):
    _patch_httpx(monkeypatch, exc=RuntimeError("connection refused"))
    assert asyncio.run(concurrency._fetch_concurrency(METRICS_URL)) is None


def test_get_concurrency_caches_and_reuses(monkeypatch):
    fake = _FakeCache()
    monkeypatch.setattr(concurrency, "cache", fake)
    _patch_httpx(monkeypatch, resp=_FakeResp(_METRICS_BODY))
    v1 = asyncio.run(concurrency.get_concurrency(METRICS_URL))
    assert v1 == 12
    assert len(fake.sets) == 1                                     # cached once

    # Second read hits the cache -> httpx would raise if called; it must not be.
    _patch_httpx(monkeypatch, exc=RuntimeError("should not fetch — cache hit"))
    v2 = asyncio.run(concurrency.get_concurrency(METRICS_URL))
    assert v2 == 12
    assert len(fake.sets) == 1                                     # no new write


def test_get_concurrency_cache_error_degrades_to_fetch(monkeypatch):
    class _BrokenCache:
        async def get(self, key):
            raise RuntimeError("redis down")
        async def set(self, key, value, ttl=None):
            raise RuntimeError("redis down")

    monkeypatch.setattr(concurrency, "cache", _BrokenCache())
    _patch_httpx(monkeypatch, resp=_FakeResp(_METRICS_BODY))
    assert asyncio.run(concurrency.get_concurrency(METRICS_URL)) == 12  # direct fetch, not broken


# ── (d) composition: health-prune THEN concurrency-reorder in resolve_chain ────

def _gated_config():
    """Single OSS profile whose AGENT step carries a ConcurrencyGate + [oss, managed]."""
    oss_steps = {
        Step.AGENT: StepConfig(
            tiers=[_oss_tier(), _managed_tier()],
            triggers=Triggers(concurrency_gate=_gate(10)),
        ),
    }
    return PipelineConfig(profiles=[NamedProfile(name="oss", weight=100, steps=oss_steps)])


def test_resolve_chain_deprioritizes_saturated_up_primary(monkeypatch):
    """Saturated but UP vLLM primary: health leaves it (closed), concurrency moves
    it behind managed -> managed tried first, but the vLLM tier is NOT dropped."""
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", True)
    monkeypatch.setattr(health.settings, "health_breaker_enabled", False)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    _inject_gauge(monkeypatch, 20)                                 # saturated

    chain = asyncio.run(split.resolve_chain("", Step.AGENT, _gated_config()))
    assert [c.provider for c in chain] == ["openai", "vllm"]       # managed first, vLLM kept last
    assert [c.kind for c in chain] == ["managed", "oss"]


def test_resolve_chain_health_prune_then_concurrency_compose(monkeypatch):
    """The composition proof: a DOWN vLLM tier is pruned by P2 FIRST, so it is
    already gone when the concurrency reorder runs and can NEVER be reordered back
    to the front — even though the box also reads as saturated."""
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", True)
    monkeypatch.setattr(health.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    _inject_gauge(monkeypatch, 99)                                 # also saturated

    from app.llm_core.health import BreakerConfig
    health.reset(BreakerConfig(fail_threshold=1, cooldown_s=1e12, healthy_polls_required=2))
    health._registry.record_failure(OSS_EP)                        # OSS box DOWN -> pruned

    chain = asyncio.run(split.resolve_chain("", Step.AGENT, _gated_config()))
    # prune -> [managed]; reorder sees no vLLM -> [managed]. Down OSS never at front.
    assert [c.provider for c in chain] == ["openai"]
    assert all(c.provider != "vllm" for c in chain)
    health.reset()


def test_resolve_chain_untouched_when_gauge_off(monkeypatch):
    """Flags off -> resolve_chain is byte-identical to the un-reordered chain."""
    monkeypatch.setattr(concurrency.settings, "concurrency_gauge_enabled", False)
    monkeypatch.setattr(health.settings, "health_breaker_enabled", False)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    _inject_gauge(monkeypatch, 99)                                 # would saturate if on

    chain = asyncio.run(split.resolve_chain("", Step.AGENT, _gated_config()))
    assert [c.provider for c in chain] == ["vllm", "openai"]       # primary stays primary
    assert [c.kind for c in chain] == ["oss", "managed"]
