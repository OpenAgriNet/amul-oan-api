"""Punjabi enablement gating: the PUNJABI_CHAT_ENABLED kill switch and the
Gujarati-only scoping of _fix_dandas (the danda ``।`` is a valid sentence
terminator in Gurmukhi text and must survive on Punjabi output)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import _get_bool_env, settings
from app.services.translation import _fix_dandas


def test_fix_dandas_preserves_danda_for_punjabi():
    text = "ਇਹ ਇੱਕ ਵਾਕ ਹੈ। ਇਹ ਦੂਜਾ ਵਾਕ ਹੈ।"
    assert _fix_dandas(text, "pa") == text
    assert _fix_dandas(text, "punjabi") == text


def test_punjabi_chat_enabled_setting_defaults_on():
    assert settings.punjabi_chat_enabled is True


def test_punjabi_chat_enabled_env_off(monkeypatch):
    monkeypatch.setenv("PUNJABI_CHAT_ENABLED", "false")
    assert _get_bool_env("PUNJABI_CHAT_ENABLED", default=True) is False
