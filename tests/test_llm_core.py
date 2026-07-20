"""Unit tests for the unified LLM pipeline core (app/llm_core), P0.

Covers: factory superset (each provider builds the right handle kind + carries
base_url/key), shim identity (synthesize_from_env reproduces the legacy env
wiring), resolver returns a non-empty chain, and the default-OFF flag posture.

Zero network: building a pydantic-ai Model / AsyncOpenAI client is lazy (no call
is made), and no test invokes a model. Sets a dummy OPENAI_API_KEY before
importing app code because agents.models constructs the managed model eagerly.

NOTE: these tests deliberately avoid importing app.services.translation /
agents.tools, which fail to import under the locally-installed pydantic-ai
(0.2.4) vs the repo-pinned 1.50.0 — a pre-existing environment mismatch unrelated
to llm_core. The AGENT-step identity check compares against agents.models, which
imports cleanly.
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
    materialize,
    synthesize_from_env,
    resolver,
    runtime,
)
from app.llm_core.factory import TGDescriptor, MaterializedTier


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
    # AnthropicModel reads ANTHROPIC_API_KEY from env at construction (matching
    # the legacy agents/models anthropic arm).
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


def test_factory_raw_openai_client_carries_api_key_and_base_url():
    os.environ["OSS_INFERENCE_API_KEY"] = "dummy-oss"
    tier = Tier(provider=Provider.VLLM, model="gemma-4-31b-it",
                endpoint="http://10.0.0.1:8020/v1", api_key_env="OSS_INFERENCE_API_KEY")
    client = build_handle(tier, StepClientKind.RAW_OPENAI)
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

def test_factory_rejects_anthropic_for_raw_openai():
    tier = Tier(provider=Provider.ANTHROPIC, model="claude-haiku-4-5")
    with pytest.raises(ValueError):
        build_handle(tier, StepClientKind.RAW_OPENAI)


def test_factory_rejects_translategemma_for_agent():
    tier = Tier(provider=Provider.TRANSLATEGEMMA, model="tg", endpoint="http://x/v1")
    with pytest.raises(ValueError):
        build_handle(tier, StepClientKind.AGENT)


def test_factory_rejects_openai_for_translategemma_kind():
    tier = Tier(provider=Provider.OPENAI, model="gpt-4.1")
    with pytest.raises(ValueError):
        build_handle(tier, StepClientKind.TRANSLATEGEMMA)


# ── materialize ───────────────────────────────────────────────────────────────

def test_materialize_preserves_order_and_timeout():
    tiers = [
        Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1", timeout_ms=8000),
        Tier(provider=Provider.OPENAI, model="gpt-4.1", timeout_ms=20000),
    ]
    mts = materialize(StepClientKind.AGENT, tiers)
    assert len(mts) == 2
    assert isinstance(mts[0], MaterializedTier)
    assert mts[0].timeout == 8.0 and mts[1].timeout == 20.0
    # .model back-compat property returns the handle
    assert mts[0].model is mts[0].handle
    assert mts[0].provider == "vllm" and mts[1].provider == "openai"


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


def test_shim_identity_agent_matches_legacy_singleton():
    """Resolver's AGENT primary == agents.models legacy singleton (real identity)."""
    from agents.models import get_model_for_variant, provider_for_variant, LLM_MODEL_NAME

    runtime.configure(run_self_check=False)
    mt = resolver.primary_tier(Step.AGENT, "legacy")
    legacy = get_model_for_variant("legacy")
    assert mt.model_name == getattr(legacy, "model_name", LLM_MODEL_NAME)
    assert mt.provider == provider_for_variant("legacy")
    # base_url identity (both build the same OpenAI-compatible client)
    assert str(mt.handle.base_url).rstrip("/") == str(legacy.base_url).rstrip("/")


# ── resolver ──────────────────────────────────────────────────────────────────

def test_resolver_returns_non_empty_chain():
    runtime.configure(run_self_check=False)
    chain = resolver.resolve_chain(Step.AGENT, "legacy")
    assert len(chain) >= 1
    assert chain[0].handle is not None
    # post-translation resolves to a TG descriptor
    post = resolver.resolve_chain(Step.POST_TRANSLATION, "legacy")
    assert isinstance(post[0].handle, TGDescriptor)


def test_resolver_falls_back_to_managed_when_oss_profile_absent():
    runtime.configure(run_self_check=False)
    # current env has no OSS profile -> asking for oss variant still resolves.
    chain = resolver.resolve_chain(Step.AGENT, "oss")
    assert len(chain) >= 1


# ── default-OFF flag posture ──────────────────────────────────────────────────

def test_llm_core_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("LLM_CORE_ENABLED", raising=False)
    from app.config import Settings
    assert Settings().llm_core_enabled is False


def test_self_check_enforces_identity_when_flag_on(monkeypatch):
    """With the flag on, a resolve-vs-legacy mismatch must raise AssertionError."""
    runtime.configure(run_self_check=False)
    monkeypatch.setattr("app.config.settings.llm_core_enabled", True, raising=False)

    real_primary_tier = resolver.primary_tier

    def _wrong(step, variant="legacy"):
        mt = real_primary_tier(step, variant)
        if step is Step.AGENT:
            return MaterializedTier(
                kind=mt.kind, handle=mt.handle, model_name="not-the-legacy-model",
                provider=mt.provider, endpoint=mt.endpoint, timeout=mt.timeout,
            )
        return mt

    monkeypatch.setattr(resolver, "primary_tier", _wrong)
    with pytest.raises(AssertionError):
        runtime.self_check()


def test_configure_does_not_raise_with_flag_off():
    """Flag-off startup must be robust even if the self-check can't import a
    legacy module in this env (pre-existing pydantic-ai mismatch) — configure()
    swallows non-assertion errors from the self-check when the flag is off."""
    cfg = runtime.configure()  # run_self_check defaults True; flag off => no raise
    assert cfg is not None and len(cfg.profiles) >= 1
