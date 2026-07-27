"""Unit tests for the P2 health filter (``app/llm_core/health.py`` +
``app/tasks/health_poller.py``) and its composition with the fallback walkers.

The bar these pin:
  (a) passive breaker — trips after N consecutive failures, half-opens after the
      cooldown, resets on a live success; a half-open probe failure re-opens;
      a success interrupts a failure streak.
  (b) poller hysteresis — an ``open`` endpoint needs K consecutive healthy polls
      to fail back to ``closed``; a failure resets that progress; a failed poll
      trips the breaker like a request failure.
  (c) ``prune_unhealthy`` — drops exactly the open-endpoint tiers, NEVER returns
      empty, and is per-endpoint isolated (an open agent endpoint does not prune
      the pre-translation tier).
  (d) flags-OFF paths untouched — record_* no-op and prune is identity when the
      HEALTH_* flags are off; the poller helpers (health-url, endpoint discovery,
      one poll sweep) work against a stubbed HTTP client (zero network).

Zero network: the breaker/registry mechanics use an injected ``now`` (no sleeps);
the poller sweep uses a fake async client returning synthetic statuses. Dummy
OPENAI/OSS keys are set before importing app code (the factory reads keys at
build time, though these tests never build a real client).
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OSS_INFERENCE_API_KEY", "test-oss-key")

import asyncio

import pytest

from app.llm_core import health
from app.llm_core.health import BreakerConfig, BreakerState, HealthRegistry
from app.llm_core.config_model import Provider, Step, Tier

AGENT_EP = "http://10.185.25.197:8020/v1"
PRE_EP = "http://10.185.25.198:8021/v1"
TG_EP = "http://localhost:18002/v1"


def _cfg(n=3, cooldown=10.0, k=2) -> BreakerConfig:
    return BreakerConfig(fail_threshold=n, cooldown_s=cooldown, healthy_polls_required=k)


# ── (a) passive breaker mechanics ─────────────────────────────────────────────

def test_breaker_trips_after_n_consecutive_failures():
    r = HealthRegistry(_cfg(n=3))
    r.record_failure(AGENT_EP, now=0.0)
    r.record_failure(AGENT_EP, now=0.0)
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED
    assert r.is_open(AGENT_EP, now=0.0) is False
    r.record_failure(AGENT_EP, now=0.0)   # Nth failure
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    assert r.is_open(AGENT_EP, now=0.0) is True


def test_breaker_half_opens_after_cooldown():
    r = HealthRegistry(_cfg(n=2, cooldown=10.0))
    r.record_failure(AGENT_EP, now=100.0)
    r.record_failure(AGENT_EP, now=100.0)
    assert r.is_open(AGENT_EP, now=105.0) is True          # still cooling -> pruned
    # cooldown elapsed -> lazily half-opens on read; probe allowed (NOT pruned).
    assert r.is_open(AGENT_EP, now=110.0) is False
    assert r.state_of(AGENT_EP) is BreakerState.HALF_OPEN


def test_breaker_resets_on_live_success():
    r = HealthRegistry(_cfg(n=2))
    r.record_failure(AGENT_EP, now=0.0)
    r.record_failure(AGENT_EP, now=0.0)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    r.record_success(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED
    assert r.is_open(AGENT_EP, now=0.0) is False


def test_half_open_probe_failure_reopens():
    r = HealthRegistry(_cfg(n=2, cooldown=10.0))
    r.record_failure(AGENT_EP, now=0.0)
    r.record_failure(AGENT_EP, now=0.0)
    assert r.is_open(AGENT_EP, now=20.0) is False          # -> half_open
    assert r.state_of(AGENT_EP) is BreakerState.HALF_OPEN
    r.record_failure(AGENT_EP, now=21.0)                   # probe failed
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    assert r.is_open(AGENT_EP, now=22.0) is True           # cooldown restarts from 21.0


def test_success_interrupts_failure_streak():
    r = HealthRegistry(_cfg(n=3))
    r.record_failure(AGENT_EP, now=0.0)
    r.record_failure(AGENT_EP, now=0.0)
    r.record_success(AGENT_EP)                             # streak cleared
    r.record_failure(AGENT_EP, now=0.0)
    r.record_failure(AGENT_EP, now=0.0)
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED     # 2 < 3 after reset


# ── (a') half-open probe token: release + time-box (Finding #3) ───────────────

def test_release_probe_frees_slot_idempotently():
    """release_probe frees ONLY the probe slot (state/counters untouched) and is a
    no-op when no probe is in flight or the endpoint is unknown; once freed, the
    NEXT caller may probe again."""
    r = HealthRegistry(_cfg(n=1, cooldown=10.0))
    r.record_failure(AGENT_EP, now=0.0)                    # -> OPEN
    assert r.is_open(AGENT_EP, now=20.0) is False          # -> HALF_OPEN + probe granted
    assert r.snapshot()[AGENT_EP]["probe_in_flight"] is True

    r.release_probe(AGENT_EP)                              # frees the slot
    assert r.snapshot()[AGENT_EP]["probe_in_flight"] is False
    assert r.state_of(AGENT_EP) is BreakerState.HALF_OPEN  # only the slot freed; state unchanged

    r.release_probe(AGENT_EP)                              # idempotent no-op
    r.release_probe("http://never-seen:9/v1")             # unknown endpoint -> no-op
    assert r.state_of(AGENT_EP) is BreakerState.HALF_OPEN

    # A freed slot -> the next caller re-probes instead of being pruned forever.
    assert r.is_open(AGENT_EP, now=21.0) is False
    assert r.snapshot()[AGENT_EP]["probe_in_flight"] is True


def test_probe_time_box_auto_releases_lost_token():
    """Backstop: a probe token never released by a walker finally (lost token) is
    auto-released by ``is_open`` after ``probe_max_s`` so a recovered endpoint is
    re-probed rather than pinned HALF_OPEN (pruned) forever."""
    r = HealthRegistry(BreakerConfig(fail_threshold=1, cooldown_s=10.0, probe_max_s=30.0))
    r.record_failure(AGENT_EP, now=0.0)                    # -> OPEN at t=0
    assert r.is_open(AGENT_EP, now=20.0) is False          # -> HALF_OPEN + probe granted at t=20
    assert r.is_open(AGENT_EP, now=25.0) is True           # 25-20 < 30 -> probe still held -> pruned
    # Token leaked (no finally, no success/failure ever cleared it). Past probe_max_s
    # the backstop auto-releases AND re-grants, so THIS caller probes again.
    assert r.is_open(AGENT_EP, now=51.0) is False          # 51-20 >= 30 -> auto-release + re-grant
    assert r.state_of(AGENT_EP) is BreakerState.HALF_OPEN
    assert r.snapshot()[AGENT_EP]["probe_in_flight"] is True


# ── (b) poller hysteresis ─────────────────────────────────────────────────────

def test_poller_hysteresis_requires_k_healthy_polls():
    r = HealthRegistry(_cfg(n=1, k=3))
    r.record_failure(AGENT_EP, now=0.0)                    # -> OPEN (n=1)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    r.record_healthy_poll(AGENT_EP)
    r.record_healthy_poll(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN       # 2 < K -> still open
    r.record_healthy_poll(AGENT_EP)                        # Kth healthy poll
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED


def test_failure_resets_healthy_poll_progress():
    r = HealthRegistry(_cfg(n=1, k=3))
    r.record_failure(AGENT_EP, now=0.0)                    # OPEN
    r.record_healthy_poll(AGENT_EP)
    r.record_healthy_poll(AGENT_EP)                        # 2 healthy polls banked
    r.record_failure(AGENT_EP, now=0.0)                    # resets healthy progress
    r.record_healthy_poll(AGENT_EP)
    r.record_healthy_poll(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN       # only 2 since reset
    r.record_healthy_poll(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED


def test_failed_poll_trips_breaker_like_a_request_failure():
    r = HealthRegistry(_cfg(n=2))
    r.record_failed_poll(AGENT_EP, now=0.0)
    r.record_failed_poll(AGENT_EP, now=0.0)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN


# ── (c) prune_unhealthy: drops open tiers, never empty, per-endpoint isolated ──

def _oss_tier(ep=AGENT_EP):
    return Tier(provider=Provider.VLLM, model="gemma", endpoint=ep,
                api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)


def _managed_tier():
    return Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY", timeout_ms=20000)


@pytest.fixture
def breaker_on(monkeypatch):
    """Activate the health filter with a fresh, small-threshold global registry.

    Cooldown is set effectively infinite so a tripped endpoint stays ``open``
    under real-``monotonic()`` ``is_open`` reads inside ``prune_unhealthy`` (these
    tests assert the pruned steady state, not the cooldown transition — that is
    covered separately with injected time)."""
    monkeypatch.setattr(health.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    r = health.reset(_cfg(n=1, cooldown=1e12))
    yield r
    health.reset()  # restore a clean global for other tests


def test_prune_drops_open_endpoint_tier(breaker_on):
    breaker_on.record_failure(AGENT_EP)                    # AGENT_EP -> OPEN (n=1)
    tiers = [_oss_tier(AGENT_EP), _managed_tier()]
    kept = health.prune_unhealthy(Step.AGENT, tiers)
    assert len(kept) == 1
    assert kept[0].provider is Provider.OPENAI             # only the managed tier survives


def test_prune_never_returns_empty(breaker_on):
    breaker_on.record_failure(AGENT_EP)                    # the ONLY tier's endpoint is open
    tiers = [_oss_tier(AGENT_EP)]
    kept = health.prune_unhealthy(Step.AGENT, tiers)
    assert kept == tiers                                   # degrade-safe: input unchanged, not []


def test_prune_never_empties_when_all_endpoints_open(breaker_on):
    breaker_on.record_failure(AGENT_EP)
    breaker_on.record_failure(PRE_EP)
    tiers = [_oss_tier(AGENT_EP), _oss_tier(PRE_EP)]        # both open
    kept = health.prune_unhealthy(Step.AGENT, tiers)
    assert kept == tiers


def test_prune_per_endpoint_isolation(breaker_on):
    """An open AGENT endpoint must NOT prune a healthy pre-translation tier."""
    breaker_on.record_failure(AGENT_EP)                    # agent box down
    # pre-translation chain keyed on a DIFFERENT endpoint -> untouched.
    pre_tiers = [_oss_tier(PRE_EP), _managed_tier()]
    kept = health.prune_unhealthy(Step.PRE_TRANSLATION, pre_tiers)
    assert len(kept) == 2                                  # nothing pruned
    assert kept[0].endpoint == PRE_EP


def test_prune_keeps_healthy_and_managed(breaker_on):
    # nothing tripped -> full chain returned unchanged.
    tiers = [_oss_tier(AGENT_EP), _managed_tier()]
    kept = health.prune_unhealthy(Step.AGENT, tiers)
    assert kept == tiers


# ── (d) flags-OFF: record_* no-op + prune is identity ─────────────────────────

def test_prune_is_identity_when_flags_off(monkeypatch):
    monkeypatch.setattr(health.settings, "health_breaker_enabled", False)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    r = health.reset(_cfg(n=1))
    r.record_failure(AGENT_EP, now=0.0)                    # endpoint IS open in the registry
    tiers = [_oss_tier(AGENT_EP), _managed_tier()]
    # ... but the filter is inert while both flags are off.
    assert health.prune_unhealthy(Step.AGENT, tiers) is tiers
    health.reset()


def test_module_record_failure_noop_when_breaker_disabled(monkeypatch):
    monkeypatch.setattr(health.settings, "health_breaker_enabled", False)
    r = health.reset(_cfg(n=1))
    for _ in range(10):
        health.record_failure(AGENT_EP)                    # module fn: gated off
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED     # registry untouched
    health.reset()


def test_module_record_failure_feeds_registry_when_enabled(monkeypatch):
    monkeypatch.setattr(health.settings, "health_breaker_enabled", True)
    r = health.reset(_cfg(n=2))
    health.record_failure(AGENT_EP)
    health.record_failure(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    health.record_success(AGENT_EP)
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED
    health.reset()


def test_module_healthy_poll_noop_when_poller_disabled(monkeypatch):
    monkeypatch.setattr(health.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(health.settings, "health_poller_enabled", False)
    r = health.reset(_cfg(n=1, k=1))
    health.record_failure(AGENT_EP)                        # OPEN
    health.record_healthy_poll(AGENT_EP)                   # poller gated off -> no failback
    assert r.state_of(AGENT_EP) is BreakerState.OPEN
    health.reset()


def test_endpoint_of_skips_managed_and_none():
    assert health._endpoint_of(_managed_tier()) is None    # openai tier -> endpoint None
    from app.llm_core.factory import MaterializedTier
    managed_mt = MaterializedTier(kind="managed", handle=object(), model_name="gpt",
                                  provider="openai", endpoint="managed", timeout=None)
    assert health._endpoint_of(managed_mt) is None         # "managed" sentinel -> not tracked


# ── poller helpers (zero network) ─────────────────────────────────────────────

def test_health_url_strips_v1():
    from app.tasks import health_poller as hp
    assert hp._health_url("http://10.185.25.197:8020/v1") == "http://10.185.25.197:8020/health"
    assert hp._health_url("http://10.185.25.197:8020/v1/") == "http://10.185.25.197:8020/health"
    assert hp._health_url("http://host:9000") == "http://host:9000/health"


def test_distinct_endpoints_collects_self_hosted_only():
    from app.tasks import health_poller as hp
    from app.llm_core.config_model import (
        ApiStyle, NamedProfile, PipelineConfig, StepConfig,
    )

    oss_steps = {
        Step.AGENT: StepConfig(tiers=[_oss_tier(AGENT_EP), _managed_tier()]),
        Step.PRE_TRANSLATION: StepConfig(tiers=[_oss_tier(PRE_EP), _managed_tier()]),
    }
    managed_steps = {Step.AGENT: StepConfig(tiers=[_managed_tier()])}
    tg = Tier(provider=Provider.TRANSLATEGEMMA, model="tg", endpoint=TG_EP,
              api_style=ApiStyle.TEXT_COMPLETION, timeout_ms=60000)
    cfg = PipelineConfig(
        profiles=[
            NamedProfile(name="oss", weight=60, steps=oss_steps),
            NamedProfile(name="managed", weight=40, steps=managed_steps),
        ],
        defaults={Step.POST_TRANSLATION: StepConfig(tiers=[tg])},
    )
    eps = set(hp._distinct_endpoints(cfg))
    assert eps == {AGENT_EP, PRE_EP, TG_EP}                # 3 independent boxes, no openai


class _FakeResp:
    def __init__(self, status):
        self.status_code = status


class _FakeClient:
    """Async client stub mapping url substrings -> status (or raising)."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def get(self, url, timeout=None):
        self.calls.append(url)
        val = None
        for key, v in self.mapping.items():
            if key in url:
                val = v
                break
        if isinstance(val, Exception):
            raise val
        return _FakeResp(val if val is not None else 200)


def test_poll_once_updates_breaker_state(monkeypatch):
    from app.tasks import health_poller as hp

    monkeypatch.setattr(health.settings, "health_poller_enabled", True)
    r = health.reset(_cfg(n=1, k=2))

    # AGENT box dead (500), PRE box unreachable (exception), TG healthy (200).
    client = _FakeClient({
        "10.185.25.197": 500,
        "10.185.25.198": ConnectionError("refused"),
        "18002": 200,
    })
    asyncio.run(hp._poll_once(client, [AGENT_EP, PRE_EP, TG_EP], timeout_s=2.0))

    assert r.state_of(AGENT_EP) is BreakerState.OPEN        # 500 trips (n=1)
    assert r.state_of(PRE_EP) is BreakerState.OPEN          # unreachable trips
    assert r.state_of(TG_EP) is BreakerState.CLOSED         # 200 stays closed
    # one more healthy sweep on the (already down) agent box does NOT flip it yet (K=2)
    client2 = _FakeClient({"10.185.25.197": 200})
    asyncio.run(hp._poll_once(client2, [AGENT_EP], timeout_s=2.0))
    assert r.state_of(AGENT_EP) is BreakerState.OPEN        # 1 < K
    asyncio.run(hp._poll_once(client2, [AGENT_EP], timeout_s=2.0))
    assert r.state_of(AGENT_EP) is BreakerState.CLOSED      # Kth healthy poll -> failback
    health.reset()


def test_start_health_poller_noop_when_disabled(monkeypatch):
    from app.tasks import health_poller as hp

    monkeypatch.setattr(hp.settings, "health_poller_enabled", False)
    asyncio.run(hp.start_health_poller())
    assert hp._worker_task is None                          # never created while off


# ── composition with the fallback walkers ─────────────────────────────────────

def _oss_moderation_pipeline():
    """A 2-profile config whose oss profile's MODERATION step is [oss, managed],
    with the oss endpoint at ``http://oss:8020/v1`` (so it can be tripped)."""
    from app.llm_core.config_model import (
        NamedProfile, PipelineConfig, StepConfig, Tier,
    )
    oss = Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1",
               api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=5000)
    managed = Tier(provider=Provider.OPENAI, model="gpt-4.1",
                   api_key_env="OPENAI_API_KEY", timeout_ms=20000)
    return PipelineConfig(profiles=[
        NamedProfile(name="oss", weight=100,
                     steps={Step.MODERATION: StepConfig(tiers=[oss, managed])}),
        NamedProfile(name="managed", weight=0,
                     steps={Step.MODERATION: StepConfig(tiers=[managed])}),
    ])


def test_fallback_resolve_chain_prunes_when_breaker_on(monkeypatch):
    """HEALTH_BREAKER_ENABLED prunes the config-driven chain, so the OSS timeout
    tax is skipped during an outage (the whole win of the pre-flight filter)."""
    from app.services import fallback as fb
    from app.llm_core import runtime

    monkeypatch.setattr(runtime, "PIPELINE", _oss_moderation_pipeline())
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(fb.settings, "health_poller_enabled", False)

    health.reset(_cfg(n=1, cooldown=1e12))
    health._registry.record_failure("http://oss:8020/v1")            # OSS endpoint down

    # session_id="" -> deterministic profile (no Redis); oss weight 100 -> [oss, managed]
    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="", profile_name="oss"))
    assert [a.kind for a in chain] == ["managed"]                    # oss pruned
    health.reset()


def test_fallback_resolve_chain_untouched_when_health_off(monkeypatch):
    """Flags off -> the config-driven chain is intact even with a tripped endpoint
    in the registry (the filter is inert)."""
    from app.services import fallback as fb
    from app.llm_core import runtime

    monkeypatch.setattr(runtime, "PIPELINE", _oss_moderation_pipeline())
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "health_breaker_enabled", False)
    monkeypatch.setattr(fb.settings, "health_poller_enabled", False)

    health.reset(_cfg(n=1))
    health._registry.record_failure("http://oss:8020/v1", now=0.0)

    chain = asyncio.run(fb._resolve_chain(pipeline="moderation", session_id="", profile_name="oss"))
    assert [a.kind for a in chain] == ["oss", "managed"]
    health.reset()


# ── walker frees the half-open probe on ANY terminal outcome (Finding #3) ─────
# A probe token is granted to a tier's endpoint during chain resolution
# (prune_unhealthy -> is_open). It is released by record_success / a BREAKER_EVIDENCE
# record_failure — but NOT by a non-evidence failure (UNKNOWN / BAD_OUTPUT) nor by a
# tier a reorder left unexecuted. The walkers' finally sweep must free it regardless,
# or a recovered endpoint stays pruned forever.

OSS_EP = "http://oss:8020/v1"


def _grant_probe(r, ep=OSS_EP):
    """Trip ``ep`` then let the cooldown elapse so is_open grants it the probe."""
    r.record_failure(ep, now=0.0)                          # -> OPEN (n=1)
    assert r.is_open(ep, now=100.0) is False               # cooldown elapsed -> HALF_OPEN + probe
    assert r.snapshot()[ep]["probe_in_flight"] is True


def test_walker_releases_probe_on_non_evidence_failure(monkeypatch, install_chain):
    """OSS probe fails on UNKNOWN (a caller bug / 4xx) -> fallbackable but NOT breaker
    evidence, so record_failure never fires; only the finally can free the probe."""
    from app.services import fallback as fb
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(fb.settings, "health_poller_enabled", False)
    monkeypatch.setattr(fb, "emit", lambda e: None)
    r = health.reset(_cfg(n=1, cooldown=10.0))
    install_chain()                                        # profile 'oss' -> [oss, managed]
    _grant_probe(r)

    async def run(attempt):
        if attempt.kind == "oss":
            raise ValueError("caller bug")                 # classify -> UNKNOWN
        return "managed-ok"

    result = asyncio.run(fb.execute_with_fallback(
        pipeline="moderation", session_id="s", profile_name="oss", run=run))
    assert result == "managed-ok"
    assert r.snapshot()[OSS_EP]["probe_in_flight"] is False   # freed by the finally
    assert r.state_of(OSS_EP) is BreakerState.HALF_OPEN       # only the slot freed
    health.reset()


def test_walker_releases_probe_on_bad_output_raise(monkeypatch, install_chain):
    """BAD_OUTPUT is non-fallbackable, so the walker RAISES on the first tier; the
    probe must still be freed on the way out."""
    from app.services import fallback as fb
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(fb.settings, "health_poller_enabled", False)
    monkeypatch.setattr(fb, "emit", lambda e: None)
    r = health.reset(_cfg(n=1, cooldown=10.0))
    install_chain()
    _grant_probe(r)

    class UnexpectedModelBehavior(Exception):
        pass

    async def run(attempt):
        raise UnexpectedModelBehavior("schema mismatch")   # classify -> BAD_OUTPUT

    with pytest.raises(UnexpectedModelBehavior):
        asyncio.run(fb.execute_with_fallback(
            pipeline="moderation", session_id="s", profile_name="oss", run=run))
    assert r.snapshot()[OSS_EP]["probe_in_flight"] is False   # freed despite the raise
    health.reset()


def test_walker_releases_probe_on_not_run_reordered_tier(monkeypatch, materialized_tier):
    """The crux: a concurrency reorder moved the just-probed OSS tier off index 0,
    so the managed tier returns first and the OSS tier NEVER executes — yet its probe
    must STILL be freed (only the whole-chain finally sweep does this)."""
    from app.services import fallback as fb
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    monkeypatch.setattr(fb.settings, "health_breaker_enabled", True)
    monkeypatch.setattr(fb.settings, "health_poller_enabled", False)
    monkeypatch.setattr(fb, "emit", lambda e: None)
    r = health.reset(_cfg(n=1, cooldown=10.0))

    async def _reordered_chain(*, pipeline, session_id, profile_name):
        return [
            materialized_tier("managed", None),                       # index 0: runs first
            materialized_tier("oss", None, endpoint=OSS_EP),          # index 1: never reached
        ]
    monkeypatch.setattr(fb, "_resolve_chain", _reordered_chain)
    _grant_probe(r)                                        # OSS probed, then reorder shifted it

    async def run(attempt):
        return f"ok-{attempt.kind}"                        # managed (index 0) succeeds

    result = asyncio.run(fb.execute_with_fallback(
        pipeline="chat", session_id="s", profile_name="x", run=run))
    assert result == "ok-managed"                          # OSS tier never ran
    assert r.snapshot()[OSS_EP]["probe_in_flight"] is False   # ...yet its probe was freed
    health.reset()
