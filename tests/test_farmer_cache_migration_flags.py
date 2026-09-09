"""Rollout flags for farmer cache consolidation (Layer 2-first migration).

Step 1 only: flags exist with safe defaults; wiring happens in later steps.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.config import Settings


def test_farmer_cache_migration_flags_default_off_except_fallback(monkeypatch):
    monkeypatch.delenv("FARMER_LAYER2_CHAT_CONTEXT_ENABLED", raising=False)
    monkeypatch.delenv("FARMER_LAYER2_FALLBACK_TO_LEGACY_ENABLED", raising=False)
    monkeypatch.delenv("FARMER_LAYER1_MOBILE_CACHE_BYPASS_ENABLED", raising=False)

    cfg = Settings()
    assert cfg.farmer_layer2_chat_context_enabled is False
    assert cfg.farmer_layer2_fallback_to_legacy_enabled is True
    assert cfg.farmer_layer1_mobile_cache_bypass_enabled is False


@pytest.mark.parametrize(
    "env_name,field_name,default",
    [
        ("FARMER_LAYER2_CHAT_CONTEXT_ENABLED", "farmer_layer2_chat_context_enabled", False),
        ("FARMER_LAYER2_FALLBACK_TO_LEGACY_ENABLED", "farmer_layer2_fallback_to_legacy_enabled", True),
        ("FARMER_LAYER1_MOBILE_CACHE_BYPASS_ENABLED", "farmer_layer1_mobile_cache_bypass_enabled", False),
    ],
)
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("TRUE", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_farmer_cache_migration_flag_parsing(monkeypatch, env_name, field_name, default, raw, expected):
    monkeypatch.setenv(env_name, raw)
    cfg = Settings()
    assert getattr(cfg, field_name) is expected


@pytest.mark.parametrize(
    "env_name,field_name,default",
    [
        ("FARMER_LAYER2_CHAT_CONTEXT_ENABLED", "farmer_layer2_chat_context_enabled", False),
        ("FARMER_LAYER2_FALLBACK_TO_LEGACY_ENABLED", "farmer_layer2_fallback_to_legacy_enabled", True),
        ("FARMER_LAYER1_MOBILE_CACHE_BYPASS_ENABLED", "farmer_layer1_mobile_cache_bypass_enabled", False),
    ],
)
@pytest.mark.parametrize("raw", ["", "nonsense"])
def test_farmer_cache_migration_invalid_bool_falls_back_to_default(
    monkeypatch, env_name, field_name, default, raw, caplog
):
    import logging

    monkeypatch.setenv(env_name, raw)
    with caplog.at_level(logging.WARNING):
        cfg = Settings()
    assert getattr(cfg, field_name) is default
    assert env_name in caplog.text
