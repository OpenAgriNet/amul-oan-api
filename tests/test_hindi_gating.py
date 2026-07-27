"""Hindi enablement gating: the HINDI_CHAT_ENABLED kill switch and the
Gujarati-only scoping of _fix_dandas (the danda ``।`` must survive in Hindi)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import _get_bool_env
from app.services.translation import _fix_dandas


# ── _fix_dandas is Gujarati-only ─────────────────────────────────────────────
# ``।`` is a TranslateGemma artifact in Gujarati output but the correct sentence
# terminator in Hindi, so the fix must never run on Hindi (or any non-Gujarati)
# output.

def test_fix_dandas_strips_danda_for_gujarati():
    assert _fix_dandas("આ વાક્ય છે। બીજું।", "gu") == "આ વાક્ય છે. બીજું."


def test_fix_dandas_strips_danda_for_gujarati_fullword():
    assert "।" not in _fix_dandas("વાક્ય।", "gujarati")


def test_fix_dandas_preserves_danda_for_hindi():
    text = "यह एक वाक्य है। दूसरा वाक्य।"
    assert _fix_dandas(text, "hi") == text
    assert _fix_dandas(text, "hindi") == text


def test_fix_dandas_default_target_keeps_gujarati_behavior():
    # A caller that omits target_lang retains the original Gujarati cleaning.
    assert "।" not in _fix_dandas("વાક્ય।")


def test_fix_dandas_other_language_preserves_danda():
    text = "sentence। here।"
    assert _fix_dandas(text, "en") == text


# ── HINDI_CHAT_ENABLED kill switch ───────────────────────────────────────────

def test_hindi_chat_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("HINDI_CHAT_ENABLED", raising=False)
    assert _get_bool_env("HINDI_CHAT_ENABLED", default=True) is True


def test_hindi_chat_enabled_env_off(monkeypatch):
    monkeypatch.setenv("HINDI_CHAT_ENABLED", "false")
    assert _get_bool_env("HINDI_CHAT_ENABLED", default=True) is False


def test_hindi_chat_enabled_env_on(monkeypatch):
    monkeypatch.setenv("HINDI_CHAT_ENABLED", "true")
    assert _get_bool_env("HINDI_CHAT_ENABLED", default=True) is True
