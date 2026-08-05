"""TranslateGemma confuses the digit ૫ with the letter પ.

The repair lives in GU_POST_REPLACEMENTS, which runs on EVERY channel — chat
renders Gujarati through the same model, so chat needs it too. Voice has carried
this fix since its own copy diverged; chat never received it.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import _post_normalize_gu_translation


def _norm(text: str) -> str:
    return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


@pytest.mark.parametrize("raw,expected", [
    ("૧પ", "૧૫"),      # 15, letter after digit
    ("પ૦", "૫૦"),      # 50, letter before digit
    ("૨પ લિટર", "૨૫ લિટર"),
])
def test_pa_adjacent_to_gujarati_digit_becomes_five(raw, expected):
    assert expected in _norm(raw)


@pytest.mark.parametrize("word", ["પાણી", "પશુ", "પીઠ"])
def test_standalone_pa_is_untouched(word):
    """No digit adjacency means no repair — these are ordinary words."""
    assert word in _norm(word)
