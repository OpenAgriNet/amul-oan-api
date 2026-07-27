"""Unit tests for M2: the redis-backed LIVE pipeline-config source
(``app/llm_core/config_source.py``) + its ``runtime.get_pipeline`` wiring.

The bar these pin:
  (a) DEFAULT OFF is behaviorally identical to boot-config-only — with
      ``PIPELINE_CONFIG_REDIS_ENABLED`` unset, ``maybe_refresh`` is an identity
      no-op that never even builds a redis client, and ``get_pipeline`` returns
      the boot config unchanged.
  (b) Enabled + a VALID config in a fake redis -> ``get_pipeline`` returns the new
      config after the TTL, and a WEIGHT change re-buckets a fixed session
      (``deterministic_profile`` maps it differently before/after) — the hot-reload
      + refresh-on-change contract.
  (c) FAIL-SAFE: missing key / invalid JSON / weights != 100 / redis GET raising
      all keep the last-good config, log a WARNING, and NEVER raise.
  (d) TTL: two calls within one window hit redis AT MOST once (call counter on the
      fake), so calling ``maybe_refresh`` per request is cheap.

Zero real network: the redis client seam (``config_source._get_redis``) is
monkeypatched to an in-memory fake; no test contacts a real redis. Dummy keys are
set before importing app code to match the other llm_core test modules (the
factory reads keys at build time), though these tests never materialize a tier.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OSS_INFERENCE_API_KEY", "test-oss-key")

import json

import pytest

from app.llm_core import config_source, split
from app.llm_core.config_model import (
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
)


# ── config builders (independent of test_split's) ─────────────────────────────

def _oss_tier():
    return Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1",
                api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)


def _managed_tier():
    return Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY",
                timeout_ms=20000)


def _two_profile(pct: int) -> PipelineConfig:
    """``[oss(pct), managed(100-pct)]`` — the seeded 2-profile split."""
    return PipelineConfig(profiles=[
        NamedProfile(name="oss", weight=pct,
                     steps={Step.AGENT: StepConfig(tiers=[_oss_tier(), _managed_tier()])}),
        NamedProfile(name="managed", weight=100 - pct,
                     steps={Step.AGENT: StepConfig(tiers=[_managed_tier()])}),
    ])


def _content_invalid() -> PipelineConfig:
    """SCHEMA-valid (weight sums to 100, unique names) but UNBUILDABLE: a vllm AGENT
    tier with NO endpoint. ``PipelineConfig(**data)`` accepts it, but the factory
    raises at materialize time (``resolver.primary_tier`` -> build_handle), so the
    boot-parity content probe in ``runtime.validate_content`` rejects it. AGENT is a
    RAW-independent step configured identically in chat and voice, so this fixture is
    byte-identical across the two repos."""
    return PipelineConfig(profiles=[
        NamedProfile(name="oss", weight=100,
                     steps={Step.AGENT: StepConfig(tiers=[
                         Tier(provider=Provider.VLLM, model="gemma", endpoint=None,
                              api_key_env="OSS_INFERENCE_API_KEY", timeout_ms=8000)])}),
    ])


def _json_of(cfg: PipelineConfig) -> str:
    """Exactly what the ops script / config_source round-trip through redis."""
    return json.dumps(cfg.model_dump(mode="json"))


# ── fake redis (in-memory, with a GET call counter) ───────────────────────────

class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.get_calls = 0
        self.raise_on_get = False

    def get(self, key):
        self.get_calls += 1
        if self.raise_on_get:
            raise RuntimeError("redis down")
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


@pytest.fixture(autouse=True)
def _clean_source(monkeypatch):
    """Reset module state + strip M2 env before each test so tests don't leak
    cached last-good / last-refresh state into each other."""
    monkeypatch.delenv(config_source.ENABLED_ENV, raising=False)
    monkeypatch.delenv(config_source.CHANNEL_ENV, raising=False)
    monkeypatch.delenv(config_source.REFRESH_ENV, raising=False)
    config_source.reset()
    yield
    config_source.reset()


@pytest.fixture
def fake(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(config_source, "_get_redis", lambda: r)
    return r


def _enable(monkeypatch, *, refresh="0"):
    monkeypatch.setenv(config_source.ENABLED_ENV, "true")
    monkeypatch.setenv(config_source.REFRESH_ENV, refresh)


# ── (a) default OFF is identity — never builds a client, returns current ───────

def test_disabled_is_identity_noop(monkeypatch):
    def _boom():
        raise AssertionError("_get_redis must not be called when the source is disabled")
    monkeypatch.setattr(config_source, "_get_redis", _boom)

    boot = _two_profile(30)
    assert config_source.maybe_refresh(boot) is boot   # same object, no redis touched


def test_channel_default_self_identifies_and_key_format(monkeypatch):
    # The default channel is derived from the repo's Step enum (voice has
    # non_meaningful; chat has suggestions) — identical assertion in both repos.
    expected = "voice" if hasattr(Step, "NON_MEANINGFUL") else "chat"
    assert config_source.channel() == expected
    assert config_source.key() == f"llm_pipeline_config:{expected}"
    # explicit override + explicit-channel key
    monkeypatch.setenv(config_source.CHANNEL_ENV, "staging")
    assert config_source.channel() == "staging"
    assert config_source.key() == "llm_pipeline_config:staging"
    assert config_source.key("other") == "llm_pipeline_config:other"


def test_refresh_interval_default_and_bad_value(monkeypatch):
    assert config_source.refresh_interval_s() == 10.0
    monkeypatch.setenv(config_source.REFRESH_ENV, "3")
    assert config_source.refresh_interval_s() == 3.0
    monkeypatch.setenv(config_source.REFRESH_ENV, "not-a-number")
    assert config_source.refresh_interval_s() == 10.0   # degrades, never raises


def test_refresh_interval_clamps_nonpositive_to_default(monkeypatch):
    # A 0/negative window would GET redis on EVERY request (blocking hot-path read);
    # clamp non-positive values to the default instead.
    monkeypatch.setenv(config_source.REFRESH_ENV, "0")
    assert config_source.refresh_interval_s() == 10.0
    monkeypatch.setenv(config_source.REFRESH_ENV, "-5")
    assert config_source.refresh_interval_s() == 10.0


def test_redis_client_uses_short_dedicated_socket_timeout(monkeypatch):
    # The config-source client gets a short dedicated connect+socket timeout
    # (default 0.5s), NOT the app-wide redis_socket_timeout (up to 10s), so a
    # slow/down redis on the hot path costs <=0.5s.
    import sys

    captured: dict = {}

    class _FakeRedisModule:
        class Redis:
            def __init__(self, **kwargs):
                captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule)
    assert config_source.redis_timeout_s() == 0.5             # default
    client = config_source.build_redis_client()
    assert client is not None
    assert captured["socket_timeout"] == 0.5
    assert captured["socket_connect_timeout"] == 0.5
    # env override is honored
    captured.clear()
    monkeypatch.setenv(config_source.TIMEOUT_ENV, "2")
    assert config_source.redis_timeout_s() == 2.0
    config_source.build_redis_client()
    assert captured["socket_timeout"] == 2.0
    assert captured["socket_connect_timeout"] == 2.0


# ── enabled + key absent -> current unchanged ─────────────────────────────────

def test_enabled_key_absent_reverts_to_boot(monkeypatch, fake):
    from app.llm_core import runtime
    _enable(monkeypatch)
    boot = _two_profile(30)
    monkeypatch.setattr(runtime, "BOOT_PIPELINE", boot)  # captured at configure()
    assert config_source.maybe_refresh(boot) is boot   # key absent -> BOOT config
    assert fake.get_calls == 1   # it DID consult redis (past TTL), found nothing


def test_clear_or_absent_reverts_to_boot_not_last_live(monkeypatch, fake):
    """`clear` (and a never-set key) reverts to the BOOT config captured at
    configure() — NOT the last LIVE config that was serving. Emergency rollback."""
    from app.llm_core import runtime
    _enable(monkeypatch)
    boot = _two_profile(30)
    live = _two_profile(70)
    monkeypatch.setattr(runtime, "BOOT_PIPELINE", boot)  # boot captured at configure()

    # 1) a live config is present and loads (distinct from boot)
    fake.store[config_source.key()] = _json_of(live)
    loaded = config_source.maybe_refresh(boot)
    assert loaded is not boot
    assert loaded.by_name("oss").weight == 70            # last-LIVE is serving

    # 2) operator clears the key -> next refresh reverts to BOOT, not last-live
    del fake.store[config_source.key()]
    monkeypatch.setattr(config_source, "_last_refresh_monotonic", 0.0)  # force TTL elapse
    reverted = config_source.maybe_refresh(loaded)       # current is the last-LIVE cfg
    assert reverted is boot                              # BOOT, not last-live
    assert reverted.by_name("oss").weight == 30


# ── (b) hot reload + re-bucket ────────────────────────────────────────────────

def test_enabled_loads_new_config_and_rebuckets_fixed_session(monkeypatch, fake):
    _enable(monkeypatch, refresh="0")   # every call past TTL -> reload
    # find a session whose bucket sits in [30,60) so a 40->20 pct flip crosses it
    sid = next(f"pick-{i}" for i in range(10_000) if 30 <= split._bucket(f"pick-{i}") < 60)
    bucket = split._bucket(sid)

    boot = _two_profile(bucket + 5)   # bucket < pct -> 'oss'
    live = _two_profile(bucket - 5)   # bucket >= pct -> 'managed'
    fake.store[config_source.key()] = _json_of(live)

    assert split.deterministic_profile(sid, boot) == "oss"      # before
    refreshed = config_source.maybe_refresh(boot)
    assert split.deterministic_profile(sid, refreshed) == "managed"  # after (re-bucketed)
    assert [(p.name, p.weight) for p in refreshed.profiles] == [("oss", bucket - 5), ("managed", 100 - (bucket - 5))]


def test_get_pipeline_returns_boot_when_disabled(monkeypatch):
    from app.llm_core import runtime
    boot = _two_profile(30)
    monkeypatch.setattr(runtime, "PIPELINE", boot)
    # disabled (default) -> get_pipeline serves the boot config, byte-identical
    assert runtime.get_pipeline() is boot


def test_get_pipeline_live_after_ttl_via_runtime(monkeypatch, fake):
    from app.llm_core import runtime
    _enable(monkeypatch, refresh="0")
    sid = next(f"rt-{i}" for i in range(10_000) if 30 <= split._bucket(f"rt-{i}") < 60)
    bucket = split._bucket(sid)

    boot = _two_profile(bucket + 5)   # sid -> 'oss'
    live = _two_profile(bucket - 5)   # sid -> 'managed'
    monkeypatch.setattr(runtime, "PIPELINE", boot)
    fake.store[config_source.key()] = _json_of(live)

    assert split.deterministic_profile(sid, runtime.get_pipeline()) == "managed"
    # and runtime's stored PIPELINE is now the live config (get_pipeline stores it)
    assert runtime.PIPELINE.by_name("oss").weight == bucket - 5


# ── (c) fail-safe: invalid JSON / weights!=100 / redis raising -> keep last-good ─

def test_invalid_json_keeps_last_good_and_warns(monkeypatch, fake, caplog):
    import logging
    _enable(monkeypatch, refresh="0")
    boot = _two_profile(30)
    fake.store[config_source.key()] = "{ this is not valid json"
    with caplog.at_level(logging.WARNING):
        out = config_source.maybe_refresh(boot)   # must NOT raise
    assert out is boot
    assert "invalid live config" in caplog.text


def test_weights_not_100_keeps_last_good(monkeypatch, fake):
    _enable(monkeypatch, refresh="0")
    boot = _two_profile(30)
    # craft raw JSON whose weights sum to 110 (bypasses the model builder)
    bad = _two_profile(40).model_dump(mode="json")
    bad["profiles"][1]["weight"] = 70          # 40 + 70 = 110 != 100
    fake.store[config_source.key()] = json.dumps(bad)
    out = config_source.maybe_refresh(boot)     # PipelineConfig validator rejects -> last-good
    assert out is boot


def test_redis_get_raising_keeps_last_good(monkeypatch, fake):
    _enable(monkeypatch, refresh="0")
    fake.raise_on_get = True
    boot = _two_profile(30)
    out = config_source.maybe_refresh(boot)     # GET raises -> caught -> last-good
    assert out is boot


def test_no_redis_client_keeps_last_good(monkeypatch):
    _enable(monkeypatch, refresh="0")
    monkeypatch.setattr(config_source, "_get_redis", lambda: None)  # e.g. redis import failed
    boot = _two_profile(30)
    assert config_source.maybe_refresh(boot) is boot


def test_content_invalid_live_config_kept_last_good_and_warns(monkeypatch, fake, caplog):
    """FAIL-CLOSED on bad CONTENT: a schema-valid but UNBUILDABLE live config
    (validate_content raises via the resolvability probe) is treated like a read
    failure — last-good kept, WARNING logged, never applied, never raised."""
    import logging
    _enable(monkeypatch, refresh="0")
    boot = _two_profile(30)
    fake.store[config_source.key()] = _json_of(_content_invalid())
    with caplog.at_level(logging.WARNING):
        out = config_source.maybe_refresh(boot)   # must NOT raise; must NOT apply
    assert out is boot                             # kept last-good (fail-closed)
    assert "content-invalid" in caplog.text


# ── (d) TTL: two calls within one window hit redis at most once ────────────────

def test_ttl_hits_redis_at_most_once_per_window(monkeypatch, fake):
    _enable(monkeypatch, refresh="1000")   # huge window -> second call must not re-read
    boot = _two_profile(30)
    live = _two_profile(50)
    fake.store[config_source.key()] = _json_of(live)

    first = config_source.maybe_refresh(boot)
    second = config_source.maybe_refresh(first)
    assert fake.get_calls == 1              # only the first call past TTL touched redis
    # first call loaded the live config; second returned it unchanged (within window)
    assert first.by_name("oss").weight == 50
    assert second is first


# ── ops script round-trips through the same fake redis ────────────────────────

def _load_ops_module():
    import importlib.util
    from pathlib import Path
    root = Path(config_source.__file__).resolve().parents[2]
    path = root / "scripts" / "set_pipeline_config.py"
    spec = importlib.util.spec_from_file_location("set_pipeline_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ops_script_set_get_clear_roundtrip(monkeypatch, tmp_path, fake, capsys):
    ops = _load_ops_module()
    monkeypatch.setattr(config_source, "build_redis_client", lambda: fake)

    good = tmp_path / "good.json"
    good.write_text(_json_of(_two_profile(50)), encoding="utf-8")
    assert ops.cmd_set("chat", str(good)) == 0
    assert fake.store["llm_pipeline_config:chat"]

    assert ops.cmd_get("chat") == 0
    assert '"weight": 50' in capsys.readouterr().out

    assert ops.cmd_clear("chat") == 0
    assert "llm_pipeline_config:chat" not in fake.store


def test_ops_script_refuses_invalid_config(monkeypatch, tmp_path, fake):
    ops = _load_ops_module()
    monkeypatch.setattr(config_source, "build_redis_client", lambda: fake)
    bad = _two_profile(40).model_dump(mode="json")
    bad["profiles"][1]["weight"] = 70          # sum 110 -> invalid
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    assert ops.cmd_set("chat", str(p)) == 1     # refused
    assert "llm_pipeline_config:chat" not in fake.store   # nothing written


def test_ops_script_refuses_content_invalid_config(monkeypatch, tmp_path, fake):
    """The ops script applies the SAME content gate as the app: a schema-valid but
    UNBUILDABLE config is refused (content probe raises) and nothing is written."""
    ops = _load_ops_module()
    monkeypatch.setattr(config_source, "build_redis_client", lambda: fake)
    p = tmp_path / "content_bad.json"
    p.write_text(_json_of(_content_invalid()), encoding="utf-8")
    assert ops.cmd_set("chat", str(p)) == 1     # refused on the content probe
    assert "llm_pipeline_config:chat" not in fake.store   # nothing written
