"""Unit tests for the unified LLM pipeline core (app/llm_core), P0.

Covers the provider factory, env-config synthesis, startup validation, and the
default-OFF fallback posture.

Zero network: building a pydantic-ai Model / AsyncOpenAI client is lazy (no call
is made), and no test invokes a model. The dummy key is read only by factory
handles materialized inside these tests.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from app.llm_core import (
    Provider,
    Step,
    StepClientKind,
    Tier,
    ApiStyle,
    build_handle,
    synthesize_from_env,
    runtime,
)
from app.llm_core.factory import TGDescriptor


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
                endpoint="http://localhost:18002/v1", api_style=ApiStyle.TEXT_COMPLETION)
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
    assert post.api_style is ApiStyle.TEXT_COMPLETION
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


# ── resolver ──────────────────────────────────────────────────────────────────

# ── self-check (resolvability, non-fatal) ─────────────────────────────────────

def test_self_check_is_non_fatal_on_unresolvable_step(monkeypatch):
    """A client construction failure is logged rather than blocking startup."""
    from app.llm_core import factory

    runtime.configure(run_self_check=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("cannot build handle in this env")

    monkeypatch.setattr(factory, "build_handle", _boom)
    runtime.self_check()


def test_configure_runs_self_check_without_raising():
    """Startup config load + self-check must be robust and return a valid config."""
    cfg = runtime.configure()  # run_self_check defaults True
    assert cfg is not None and len(cfg.profiles) >= 1


def test_yaml_config_does_not_enable_fallback_implicitly():
    from app.llm_core.config_model import NamedProfile, PipelineConfig

    cfg = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100)])
    assert cfg.fallback_enabled is False


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
    runtime.validate_config(cfg, enforce=True)


def test_validate_config_rejects_gemini_pretranslation_when_enforced():
    cfg = _cfg_with_pretranslation_provider(Provider.GEMINI)
    with pytest.raises(ValueError, match="pretranslation"):
        runtime.validate_config(cfg, enforce=True)


def test_validate_config_warns_not_raises_when_flag_off():
    cfg = _cfg_with_pretranslation_provider(Provider.GEMINI)
    runtime.validate_config(cfg, enforce=False)  # no raise


def test_validate_config_accepts_openai_and_vllm_raw_pretranslation():
    cfg = _cfg_with_pretranslation_provider(Provider.OPENAI)
    runtime.validate_config(cfg, enforce=True)  # openai is PRE_TRANSLATION-legal
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
    runtime.validate_config(cfg2, enforce=True)  # vllm is PRE_TRANSLATION-legal


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
