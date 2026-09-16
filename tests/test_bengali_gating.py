"""Bengali enablement gating: the BENGALI_CHAT_ENABLED kill switch and the
Gujarati-only scoping of _fix_dandas (the danda ``।`` is the Bengali sentence
terminator and must survive)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import _get_bool_env, settings
from app.services.translation import _fix_dandas


def test_fix_dandas_preserves_danda_for_bengali():
    text = "এটি একটি বাক্য। আরেকটি বাক্য।"
    assert _fix_dandas(text, "bn") == text
    assert _fix_dandas(text, "bengali") == text


def test_bengali_chat_enabled_setting_defaults_on():
    assert settings.bengali_chat_enabled is True


def test_bengali_chat_enabled_env_off(monkeypatch):
    monkeypatch.setenv("BENGALI_CHAT_ENABLED", "false")
    assert _get_bool_env("BENGALI_CHAT_ENABLED", default=True) is False
