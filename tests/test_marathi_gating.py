"""Marathi enablement gating: the MARATHI_CHAT_ENABLED kill switch and the
Gujarati-only scoping of _fix_dandas (the danda ``।`` is a valid Devanagari
sentence terminator and must survive on Marathi output)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import _get_bool_env, settings
from app.services.translation import _fix_dandas


def test_fix_dandas_preserves_danda_for_marathi():
    text = "हे एक वाक्य आहे। हे दुसरे वाक्य आहे।"
    assert _fix_dandas(text, "mr") == text
    assert _fix_dandas(text, "marathi") == text


def test_marathi_chat_enabled_setting_defaults_on():
    assert settings.marathi_chat_enabled is True


def test_marathi_chat_enabled_env_off(monkeypatch):
    monkeypatch.setenv("MARATHI_CHAT_ENABLED", "false")
    assert _get_bool_env("MARATHI_CHAT_ENABLED", default=True) is False
