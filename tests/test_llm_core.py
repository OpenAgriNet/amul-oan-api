"""Unit tests for the unified LLM pipeline core (app/llm_core), P0.

Covers the provider factory, env-config synthesis, startup validation, and the
default-OFF fallback posture.

Zero network: building a pydantic-ai Model / AsyncOpenAI client is lazy (no call
is made), and no test invokes a model. The dummy key is read only by factory
handles built inside these tests.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from app.llm_core.config_model import (
    Provider,
    Step,
    StepClientKind,
    Tier,
)
from app.llm_core.factory import TGDescriptor, build_handle
from app.llm_core.legacy_shim import synthesize_from_env
from app.llm_core import runtime


def _openai_model_types() -> tuple[str, ...]:
    # pydantic-ai 1.x -> OpenAIChatModel; older -> OpenAIModel.
    return ("OpenAIChatModel", "OpenAIModel")


# ── factory superset ──────────────────────────────────────────────────────────

def test_factory_vllm_agent_builds_openai_model_with_base_url():
    tier = Tier(provider=Provider.VLLM, model="gemma-4-31b-it",
                endpoint="http://10.0.0.1:8020/v1", api_key_env="OSS_INFERENCE_API_KEY")
    os.environ["OSS_INFERENCE_API_KEY"] = "dummy-oss"
    handle = build_handle(tier, StepClientKind.AGENT)
    assert type(handle).__name__ in _openai_model_types()
    assert str(handle.base_url).rstrip("/") == "http://10.0.0.1:8020/v1"
    assert handle.model_name == "gemma-4-31b-it"


def test_factory_openai_agent_targets_openai_default():
    tier = Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY")
    handle = build_handle(tier, StepClientKind.AGENT)
    assert type(handle).__name__ in _openai_model_types()
    assert "openai.com" in str(handle.base_url)


def test_factory_anthropic_agent_builds_anthropic_model():
    # AnthropicModel reads ANTHROPIC_API_KEY from env at construction.
    os.environ["ANTHROPIC_API_KEY"] = "anthropic-dummy"
    tier = Tier(provider=Provider.ANTHROPIC, model="claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY")
    handle = build_handle(tier, StepClientKind.AGENT)
    assert type(handle).__name__ == "AnthropicModel"


def test_factory_azure_agent_builds_model():
    tier = Tier(provider=Provider.AZURE, model="my-deploy",
                endpoint="https://example.openai.azure.com", api_version="2024-02-01",
                api_key_env="AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_API_KEY"] = "azure-dummy"
    handle = build_handle(tier, StepClientKind.AGENT)
    assert type(handle).__name__ in _openai_model_types()


def test_factory_gemini_agent_builds_model():
    tier = Tier(provider=Provider.GEMINI, model="gemini-2.5-flash", api_key_env="GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "gemini-dummy"
    try:
        handle = build_handle(tier, StepClientKind.AGENT)
    except RuntimeError as exc:  # SDK genuinely unavailable
        pytest.skip(f"gemini SDK unavailable: {exc}")
    assert type(handle).__name__ in ("GoogleModel", "GeminiModel")


def test_factory_pretranslation_client_carries_api_key_and_base_url():
    os.environ["OSS_INFERENCE_API_KEY"] = "dummy-oss"
    tier = Tier(provider=Provider.VLLM, model="gemma-4-31b-it",
                endpoint="http://10.0.0.1:8020/v1", api_key_env="OSS_INFERENCE_API_KEY")
    client = build_handle(tier, StepClientKind.PRE_TRANSLATION)
    assert type(client).__name__ == "AsyncOpenAI"
    assert str(client.base_url).rstrip("/") == "http://10.0.0.1:8020/v1"
    assert client.api_key == "dummy-oss"


def test_factory_translategemma_builds_descriptor():
    tier = Tier(provider=Provider.TRANSLATEGEMMA, model="translategemma-27b-base",
                endpoint="http://localhost:18002/v1")
    desc = build_handle(tier, StepClientKind.TRANSLATEGEMMA)
    assert isinstance(desc, TGDescriptor)
    assert desc.completions_url == "http://localhost:18002/v1/completions"
    assert desc.model_id == "translategemma-27b-base"


# ── legality enforcement ──────────────────────────────────────────────────────

def test_factory_builds_anthropic_pretranslation_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tier = Tier(
        provider=Provider.ANTHROPIC,
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    )
    assert type(build_handle(tier, StepClientKind.PRE_TRANSLATION)).__name__ == "AsyncAnthropic"


def test_factory_rejects_translategemma_for_agent():
    tier = Tier(provider=Provider.TRANSLATEGEMMA, model="tg", endpoint="http://x/v1")
    with pytest.raises(ValueError):
        build_handle(tier, StepClientKind.AGENT)


def test_factory_rejects_openai_for_translategemma_kind():
    tier = Tier(provider=Provider.OPENAI, model="gpt-4.1")
    with pytest.raises(ValueError):
        build_handle(tier, StepClientKind.TRANSLATEGEMMA)


# ── materialize ───────────────────────────────────────────────────────────────

# ── shim identity ─────────────────────────────────────────────────────────────

def test_shim_managed_only_when_oss_unconfigured(monkeypatch):
    for k in ("OSS_INFERENCE_ENDPOINT_URL", "OSS_PIPELINE_PCT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4")
    cfg = synthesize_from_env()
    assert [(p.name, p.weight) for p in cfg.profiles] == [("managed", 100)]
    managed = cfg.by_name("managed")
    agent = cfg.step_config(managed, Step.AGENT).tiers[0]
    assert agent.provider is Provider.OPENAI and agent.model == "gpt-4"
    assert agent.api_key_env == "OPENAI_API_KEY" and agent.endpoint is None
    assert agent.timeout_ms == 20000  # FALLBACK_MANAGED_TIMEOUT_MS default


def test_shim_two_profiles_when_oss_configured(monkeypatch):
    monkeypatch.setenv("OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")
    monkeypatch.setenv("OSS_LLM_MODEL_NAME", "gemma-4-31b-it")
    monkeypatch.setenv("OSS_PIPELINE_PCT", "80")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4.1")
    cfg = synthesize_from_env()
    assert {p.name: p.weight for p in cfg.profiles} == {"oss": 80, "managed": 20}
    oss = cfg.by_name("oss")
    # OSS agent step mirrors attempt_chain: [oss, managed]
    oss_agent = cfg.step_config(oss, Step.AGENT).tiers
    assert [t.provider for t in oss_agent] == [Provider.VLLM, Provider.OPENAI]
    assert oss_agent[0].endpoint == "http://oss:8020/v1"
    assert oss_agent[0].model == "gemma-4-31b-it"
    assert oss_agent[0].timeout_ms == 8000  # FALLBACK_CHAT_OSS_TIMEOUT_MS default


def test_shim_pretranslation_and_post_translation(monkeypatch):
    monkeypatch.delenv("OSS_INFERENCE_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("PRETRANSLATION_PROVIDER", raising=False)
    monkeypatch.delenv("PRETRANSLATION_MODEL", raising=False)
    monkeypatch.setenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1")
    cfg = synthesize_from_env()
    managed = cfg.by_name("managed")
    pre = cfg.step_config(managed, Step.PRE_TRANSLATION).tiers[0]
    assert pre.provider is Provider.OPENAI and pre.model == "gpt-4.1-mini"
    post = cfg.defaults[Step.POST_TRANSLATION].tiers[0]
    assert post.provider is Provider.TRANSLATEGEMMA
    assert post.endpoint == "http://localhost:18002/v1"
    assert post.model == "translategemma-27b-base"


def test_shim_post_translation_tg_ttft_deadline(monkeypatch):
    """TG carries a distinct SHORT first-token deadline (ttft_ms) separate from its
    60s overall/total cap (timeout_ms); default 5000ms, overridable via env. The
    managed overflow tier has none (it keeps only its own timeout)."""
    monkeypatch.delenv("OSS_INFERENCE_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1")

    monkeypatch.delenv("FALLBACK_POST_TRANSLATION_TG_TTFT_MS", raising=False)
    tiers = synthesize_from_env().defaults[Step.POST_TRANSLATION].tiers
    tg, overflow = tiers[0], tiers[1]
    assert tg.timeout_ms == 60000       # overall/total cap unchanged
    assert tg.ttft_ms == 5000           # distinct short first-token deadline (default)
    assert overflow.ttft_ms is None     # overflow keeps only its own timeout

    monkeypatch.setenv("FALLBACK_POST_TRANSLATION_TG_TTFT_MS", "3500")
    tg2 = synthesize_from_env().defaults[Step.POST_TRANSLATION].tiers[0]
    assert tg2.ttft_ms == 3500 and tg2.timeout_ms == 60000


def test_yaml_config_does_not_enable_fallback_implicitly():
    from app.llm_core.config_model import NamedProfile, PipelineConfig

    cfg = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100)])
    assert cfg.fallback_enabled is False
    with pytest.raises(ValueError, match="no agent plan"):
        runtime.validate_content(cfg)


# ── (D) vLLM/OSS tier without an endpoint must RAISE (not silently build OpenAI) ─

def test_vllm_pretranslation_without_endpoint_raises():
    """A vLLM PRE_TRANSLATION tier missing its endpoint raises instead of building an
    OpenAI-default client — preserving the legacy fail-OPEN behaviour when OSS is
    unconfigured (moderation/pretranslation catch the raise and fail open)."""
    tier = Tier(provider=Provider.VLLM, model="gemma-4-31b-it",
                api_key_env="OSS_INFERENCE_API_KEY")  # endpoint omitted
    with pytest.raises(ValueError, match="endpoint"):
        build_handle(tier, StepClientKind.PRE_TRANSLATION)


def test_vllm_agent_without_endpoint_raises():
    """Same guard on the AGENT builder."""
    tier = Tier(provider=Provider.VLLM, model="gemma-4-31b-it",
                api_key_env="OSS_INFERENCE_API_KEY")  # endpoint omitted
    with pytest.raises(ValueError, match="endpoint"):
        build_handle(tier, StepClientKind.AGENT)


def test_openai_raw_without_endpoint_is_fine():
    """An OpenAI (managed) PRE_TRANSLATION tier legitimately has no endpoint (base_url
    None => OpenAI proper) and must NOT raise."""
    tier = Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY")
    client = build_handle(tier, StepClientKind.PRE_TRANSLATION)
    assert client is not None


# ── startup config validation for pretranslation adapters ────────────────────

def _cfg_with_pretranslation_provider(provider: Provider) -> "object":
    from app.llm_core.config_model import (
        NamedProfile, PipelineConfig, StepConfig, Tier as _Tier,
    )
    pre = _Tier(provider=provider, model="some-model", api_key_env="X")
    agent = _Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY")
    steps = {
        Step.AGENT: StepConfig(tiers=[agent]),
        Step.PRE_TRANSLATION: StepConfig(tiers=[pre]),
    }
    return PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps=steps)])


def test_validate_config_accepts_anthropic_pretranslation():
    cfg = _cfg_with_pretranslation_provider(Provider.ANTHROPIC)
    runtime.validate_config(cfg)


def test_validate_config_rejects_gemini_pretranslation_when_enforced():
    cfg = _cfg_with_pretranslation_provider(Provider.GEMINI)
    with pytest.raises(ValueError, match="pretranslation"):
        runtime.validate_config(cfg)


def test_validate_config_accepts_openai_and_vllm_raw_pretranslation():
    cfg = _cfg_with_pretranslation_provider(Provider.OPENAI)
    runtime.validate_config(cfg)  # openai is PRE_TRANSLATION-legal
    from app.llm_core.config_model import (
        NamedProfile, PipelineConfig, StepConfig, Tier as _Tier,
    )
    vllm_pre = _Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1",
                     api_key_env="OSS_INFERENCE_API_KEY")
    agent = _Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY")
    cfg2 = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps={
        Step.AGENT: StepConfig(tiers=[agent]),
        Step.PRE_TRANSLATION: StepConfig(tiers=[vllm_pre]),
    })])
    runtime.validate_config(cfg2)  # vllm is PRE_TRANSLATION-legal


# ── (ENABLE) concurrency gate is attached from AGENT_CONCURRENCY_METRICS_URL ────

def test_concurrency_gate_attached_from_env(monkeypatch):
    """When AGENT_CONCURRENCY_METRICS_URL is set (with OSS configured), the shim
    attaches a ConcurrencyGate to the OSS AGENT step; CONCURRENCY_MAX sets the
    threshold. Unset => no gate (harmless no-op)."""
    from app.llm_core.config_model import ConcurrencyGate

    monkeypatch.setenv("OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")
    monkeypatch.setenv("OSS_PIPELINE_PCT", "80")
    monkeypatch.setenv("AGENT_CONCURRENCY_METRICS_URL", "http://oss:8020/metrics")
    monkeypatch.setenv("CONCURRENCY_MAX", "7")

    cfg = synthesize_from_env()
    oss = cfg.by_name("oss")
    gate = oss.steps[Step.AGENT].triggers.concurrency_gate
    assert isinstance(gate, ConcurrencyGate)
    assert gate.metrics_url == "http://oss:8020/metrics"
    assert gate.max_concurrency == 7


def test_no_concurrency_gate_without_env(monkeypatch):
    monkeypatch.setenv("OSS_INFERENCE_ENDPOINT_URL", "http://oss:8020/v1")
    monkeypatch.setenv("OSS_PIPELINE_PCT", "80")
    monkeypatch.delenv("AGENT_CONCURRENCY_METRICS_URL", raising=False)

    cfg = synthesize_from_env()
    oss = cfg.by_name("oss")
    assert oss.steps[Step.AGENT].triggers.concurrency_gate is None


def test_omitted_capabilities_preserve_vllm_gemma_behavior():
    from app.llm_core.config_model import NamedProfile, PipelineConfig, StepConfig
    from app.llm_core.execution import ExecutionContext

    cfg = PipelineConfig(profiles=[NamedProfile(name="old-yaml", weight=100, steps={
        Step.AGENT: StepConfig(tiers=[Tier(
            provider=Provider.VLLM,
            model="gemma-4-31b-it",
            endpoint="http://oss:8020/v1",
        )]),
    })])

    capabilities = ExecutionContext("s", cfg, "old-yaml").capabilities
    assert capabilities.requires_translation is True
    assert capabilities.history_max_tokens == 10_000


def test_partial_capabilities_merge_with_reachable_overflow():
    from app.llm_core.config_model import (
        ConcurrencyGate, NamedProfile, PipelineConfig, ProfileCapabilities,
        StepConfig, Triggers,
    )
    from app.llm_core.execution import ExecutionContext

    overflow = Tier(
        provider=Provider.VLLM, model="gemma", endpoint="http://overflow/v1"
    )
    agent = StepConfig(
        tiers=[
            Tier(provider=Provider.OPENAI, model="gpt"),
            Tier(provider=Provider.VLLM, model="qwen", endpoint="http://qwen/v1"),
        ],
        triggers=Triggers(concurrency_gate=ConcurrencyGate(
            metrics_url="http://metrics", overflow_tier=overflow
        )),
    )
    cfg = PipelineConfig(profiles=[NamedProfile(
        name="mixed",
        weight=100,
        capabilities=ProfileCapabilities(history_max_tokens=20_000),
        steps={Step.AGENT: agent},
    )], fallback_enabled=True)
    capabilities = ExecutionContext("s", cfg, "mixed").capabilities
    assert capabilities.requires_translation is True
    assert capabilities.history_max_tokens == 20_000

    inactive = cfg.model_copy(update={
        "fallback_enabled": False,
        "profiles": [cfg.profiles[0].model_copy(update={"capabilities": None})],
    })
    capabilities = ExecutionContext("s", inactive, "mixed").capabilities
    assert capabilities.requires_translation is False
    assert capabilities.history_max_tokens == 80_000

    cfg = cfg.model_copy(update={"profiles": [cfg.profiles[0].model_copy(update={
        "capabilities": ProfileCapabilities(requires_translation=False)
    })]})
    capabilities = ExecutionContext("s", cfg, "mixed").capabilities
    assert capabilities.requires_translation is False
    assert capabilities.history_max_tokens == 10_000


def test_admission_auto_preserves_managed_provider_policy():
    from app.llm_core.config_model import AdmissionPolicy
    from app.llm_core.execution import ExecutionTarget

    managed = ExecutionTarget(
        Tier(provider=Provider.OPENAI, model="gpt-4.1"), StepClientKind.AGENT
    )
    local = ExecutionTarget(
        Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss/v1"),
        StepClientKind.AGENT,
    )
    translation = ExecutionTarget(
        Tier(provider=Provider.TRANSLATEGEMMA, model="tg", endpoint="http://tg/v1"),
        StepClientKind.TRANSLATEGEMMA,
    )
    assert managed.admission is AdmissionPolicy.MANAGED
    assert local.admission is AdmissionPolicy.NONE
    assert translation.kind == "oss"


def test_content_validation_checks_every_fallback_tier():
    from app.llm_core.config_model import (
        ConcurrencyGate, NamedProfile, PipelineConfig, StepConfig, Triggers,
    )

    cfg = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps={
        Step.AGENT: StepConfig(tiers=[
            Tier(provider=Provider.OPENAI, model="gpt-4.1", api_key_env="OPENAI_API_KEY"),
            Tier(provider=Provider.AZURE, model="broken-fallback"),
        ]),
    })], fallback_enabled=True)
    with pytest.raises(ValueError, match="azure-openai"):
        runtime.validate_content(cfg)
    runtime.validate_content(cfg.model_copy(update={"fallback_enabled": False}))

    overflow = Tier(provider=Provider.VLLM, model="broken-overflow")
    cfg = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps={
        Step.AGENT: StepConfig(
            tiers=[
                Tier(provider=Provider.OPENAI, model="gpt-4.1"),
                Tier(provider=Provider.VLLM, model="qwen", endpoint="http://qwen/v1"),
            ],
            triggers=Triggers(concurrency_gate=ConcurrencyGate(
                metrics_url="http://metrics", overflow_tier=overflow
            )),
        )
    })], fallback_enabled=True)
    with pytest.raises(ValueError, match="endpoint"):
        runtime.validate_content(cfg)


def test_validation_rejects_unsupported_posttranslation_provider():
    from app.llm_core.config_model import NamedProfile, PipelineConfig, StepConfig

    cfg = PipelineConfig(
        profiles=[NamedProfile(name="managed", weight=100)],
        defaults={Step.POST_TRANSLATION: StepConfig(tiers=[
            Tier(provider=Provider.ANTHROPIC, model="claude-haiku")
        ])},
    )
    with pytest.raises(ValueError, match="post-translation"):
        runtime.validate_config(cfg)


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_agent_provider_does_not_set_posttranslation_protocol(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL_NAME", "agent-only-model")
    monkeypatch.setenv("FALLBACK_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("OSS_INFERENCE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("PIPELINE_CONFIG_PATH", raising=False)
    monkeypatch.setattr(runtime, "PIPELINE", None)
    monkeypatch.setattr(runtime, "BOOT_PIPELINE", None)

    cfg = runtime.configure()
    post = cfg.defaults[Step.POST_TRANSLATION].tiers
    assert [tier.provider for tier in post] == [Provider.TRANSLATEGEMMA]

    monkeypatch.setenv("FALLBACK_ENABLED", "true")
    post = runtime.configure().defaults[Step.POST_TRANSLATION].tiers
    assert [tier.provider for tier in post] == [Provider.TRANSLATEGEMMA]

    monkeypatch.setenv("POST_TRANSLATION_LLM_PROVIDER", "openai")
    post = runtime.configure().defaults[Step.POST_TRANSLATION].tiers
    assert [tier.provider for tier in post] == [Provider.TRANSLATEGEMMA, Provider.OPENAI]
    assert post[1].model == "gpt-4.1"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("azure-openai", Provider.AZURE), ("vllm", Provider.VLLM)],
)
def test_posttranslation_preserves_compatible_agent_provider(monkeypatch, provider, expected):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("FALLBACK_ENABLED", "true")
    monkeypatch.delenv("POST_TRANSLATION_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("INFERENCE_ENDPOINT_URL", "http://vllm/v1")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    post = synthesize_from_env().defaults[Step.POST_TRANSLATION].tiers
    assert post[1].provider is expected


def test_invalid_boot_config_is_not_published(monkeypatch, tmp_path):
    previous = synthesize_from_env()
    path = tmp_path / "pipeline.yaml"
    path.write_text("""
fallback_enabled: true
profiles:
  - name: managed
    weight: 100
    steps:
      agent:
        tiers:
          - {provider: openai, model: gpt-4.1}
          - {provider: azure-openai, model: broken}
""")
    monkeypatch.setattr(runtime, "PIPELINE", previous)
    monkeypatch.setattr(runtime, "BOOT_PIPELINE", previous)
    monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(path))

    with pytest.raises(runtime.BootRefused):
        runtime.configure()
    assert runtime.PIPELINE is previous
    assert runtime.BOOT_PIPELINE is previous

    monkeypatch.delenv("PIPELINE_CONFIG_PATH")
    monkeypatch.setenv("FALLBACK_ENABLED", "false")
    monkeypatch.setenv("REQUIRE_OVERFLOW_ARMED", "true")
    with pytest.raises(runtime.BootRefused):
        runtime.configure()
    assert runtime.PIPELINE is previous
    assert runtime.BOOT_PIPELINE is previous
