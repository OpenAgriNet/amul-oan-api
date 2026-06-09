"""Inc 7.4a — voice pretranslation subsystem ported alongside chat's. These pin
the network-free paths (english/empty short-circuit before any API call) plus the
pure glossary-transliteration helpers, and prove the 3 public entry points import.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import re
import asyncio

from app.services.translation import (
    translate_to_english_with_gpt5_mini,
    translate_to_english_with_oss_vllm,
    translate_to_english_with_structured_fallback,
    _apply_exact_glossary_transliteration_replacements,
    _whole_ascii_token_pattern,
)

_PUBLIC = (
    translate_to_english_with_gpt5_mini,
    translate_to_english_with_oss_vllm,
    translate_to_english_with_structured_fallback,
)


def test_english_source_passthrough_no_network():
    for fn in _PUBLIC:
        assert asyncio.run(fn("already english", "english")) == "already english"
        assert asyncio.run(fn("already english", "en")) == "already english"


def test_empty_input_passthrough_no_network():
    for fn in _PUBLIC:
        assert asyncio.run(fn("", "gu")) == ""
        assert asyncio.run(fn("   ", "gu")) == "   "


def test_transliteration_replacement_noop_on_empty():
    assert _apply_exact_glossary_transliteration_replacements("", "x") == "x"
    assert _apply_exact_glossary_transliteration_replacements("src", "") == ""


def test_whole_ascii_token_pattern_matches_whole_token_only():
    pat = _whole_ascii_token_pattern("bloat")
    assert re.search(pat, "the cow has bloat now", flags=re.IGNORECASE)
    assert not re.search(pat, "bloating issue", flags=re.IGNORECASE)  # substring, not whole token
