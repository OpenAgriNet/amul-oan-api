"""Agent definitions are behavior-only; llm_core owns executable models."""

from agents.agrinet import _agrinet_max_output_tokens, agrinet_agent
from agents.moderation import moderation_agent
from agents.suggestions import suggestions_agent
from app.config import settings


def test_agents_have_no_construction_time_model():
    """A missing per-turn resolver result must fail, not use a hidden singleton."""
    assert agrinet_agent.model is None
    assert moderation_agent.model is None
    assert suggestions_agent.model is None


def test_agrinet_gemma_token_cap_still_uses_startup_settings(monkeypatch):
    monkeypatch.delenv("AGRINET_MAX_TOKENS", raising=False)
    monkeypatch.setattr(settings, "llm_provider", "vllm")
    monkeypatch.setattr(settings, "llm_model_name", "gemma-4-31b-it")

    assert _agrinet_max_output_tokens() == 2048


def test_agrinet_explicit_token_cap_still_wins(monkeypatch):
    monkeypatch.setenv("AGRINET_MAX_TOKENS", "1234")
    monkeypatch.setattr(settings, "llm_provider", "vllm")
    monkeypatch.setattr(settings, "llm_model_name", "gemma-4-31b-it")

    assert _agrinet_max_output_tokens() == 1234
