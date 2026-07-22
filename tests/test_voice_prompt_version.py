"""
Regression tests for the VOICE_AGENT_PROMPT_VERSION selector and the
three prompt variants (mixed, gpt-5.1, gemma4).

The legacy `mixed` variant is already covered exhaustively by
tests/test_voice_regressions_apr11_12.py and tests/test_prompt_behavior.py.
This file pins:
  - the env-var dispatch in agents.voice._resolve_voice_prompt_name
  - that each variant file exists, loads, and covers every live use case
    (tools, routing intents, persona, hardcoded facts, vague-query rule,
    voice contract)

The new variants are intentionally NOT required to contain the exact
locked substrings the legacy tests pin on mixed — they are tuned to the
GPT-5.1 and Gemma 4 prompting guides respectively.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import voice as voice_module
from helpers.utils import get_prompt


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "prompts"

VARIANTS = {
    "mixed": "voice_system_translation_pipeline_en",
    "gpt-5.1": "voice_system_translation_pipeline_gpt5_1_en",
    "gemma4": "voice_system_translation_pipeline_gemma4_en",
}

NEW_VARIANTS = ["gpt-5.1", "gemma4"]

LIVE_TOOLS = [
    "search_documents",
    "search_terms",
    "create_ai_call",
    "create_health_call",
    "get_farmer_milk_collection_details",
    "get_union_scheme_data",
    "signal_conversation_state",
]

ROUTING_INTENTS = [
    "clinical",
    "nutrition",
    "breeding",
    "crop",
    "scheme",
    "market",
    "weather",
    "services",
    "profile",
    "language_switch",
    "out_of_scope",
]


# ---------------------------------------------------------------------------
# Env-var dispatch
# ---------------------------------------------------------------------------

class TestVoicePromptVersionDispatch:
    @pytest.mark.parametrize(
        "env_value, expected_name",
        [
            (None, "voice_system_translation_pipeline_en"),
            ("", "voice_system_translation_pipeline_en"),
            ("mixed", "voice_system_translation_pipeline_en"),
            ("MIXED", "voice_system_translation_pipeline_en"),
            (" mixed ", "voice_system_translation_pipeline_en"),
            ("gpt-5.1", "voice_system_translation_pipeline_gpt5_1_en"),
            ("GPT-5.1", "voice_system_translation_pipeline_gpt5_1_en"),
            ("gemma4", "voice_system_translation_pipeline_gemma4_en"),
            ("Gemma4", "voice_system_translation_pipeline_gemma4_en"),
        ],
    )
    def test_known_values_resolve(self, monkeypatch, env_value, expected_name):
        if env_value is None:
            monkeypatch.delenv("VOICE_AGENT_PROMPT_VERSION", raising=False)
        else:
            monkeypatch.setenv("VOICE_AGENT_PROMPT_VERSION", env_value)
        assert voice_module._resolve_voice_prompt_name() == expected_name

    def test_unknown_value_falls_back_to_mixed(self, monkeypatch, caplog):
        monkeypatch.setenv("VOICE_AGENT_PROMPT_VERSION", "gpt-7-ultra")
        with caplog.at_level("WARNING"):
            resolved = voice_module._resolve_voice_prompt_name()
        assert resolved == "voice_system_translation_pipeline_en"

    def test_default_constant_is_mixed(self):
        assert voice_module.DEFAULT_VOICE_PROMPT_VERSION == "mixed"

    def test_variant_map_matches_expected(self):
        assert voice_module.VOICE_PROMPT_VARIANTS == VARIANTS


# ---------------------------------------------------------------------------
# Variant files all exist and load
# ---------------------------------------------------------------------------

class TestAllVariantsLoad:
    @pytest.mark.parametrize("version, filename_stem", list(VARIANTS.items()))
    def test_file_exists(self, version, filename_stem):
        path = PROMPTS_DIR / f"{filename_stem}.md"
        assert path.is_file(), f"Missing prompt file for variant {version}: {path}"

    @pytest.mark.parametrize("version, filename_stem", list(VARIANTS.items()))
    def test_file_loads_via_get_prompt(self, version, filename_stem):
        text = get_prompt(filename_stem)
        assert text and len(text.strip()) > 200, f"{version} prompt is empty or too short"


# ---------------------------------------------------------------------------
# Per-variant use-case-anchor invariants (applied to the NEW variants only).
# The legacy mixed variant has its own exhaustive locked-substring tests.
# ---------------------------------------------------------------------------

def _load(version: str) -> str:
    return get_prompt(VARIANTS[version])


class TestNewVariantInvariants:
    @pytest.mark.parametrize("version", NEW_VARIANTS)
    @pytest.mark.parametrize("tool", LIVE_TOOLS)
    def test_each_live_tool_is_named(self, version, tool):
        text = _load(version)
        assert tool in text, f"{version} prompt does not mention tool `{tool}`"

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    @pytest.mark.parametrize("intent", ROUTING_INTENTS)
    def test_each_routing_intent_is_named(self, version, intent):
        text = _load(version)
        assert intent in text, f"{version} prompt does not mention routing intent `{intent}`"

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_persona_present(self, version):
        text = _load(version)
        assert "Sarlaben" in text
        assert "Amul AI" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_phone_call_register_present(self, version):
        text = _load(version)
        lower = text.lower()
        assert "spoken" in lower or "phone" in lower

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_english_output_rule_present(self, version):
        text = _load(version)
        lower = text.lower()
        assert "english" in lower

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_hardcoded_pasteurization_fact_present(self, version):
        text = _load(version).lower()
        # Either ISO temperature form or fully-spelled-out voice form must appear.
        assert (
            "85 to 90 degrees celsius" in text
            or "eighty five to ninety degrees celsius" in text
        ), f"{version} prompt is missing the milk pasteurization hardcoded fact"

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_translation_layer_invariants(self, version):
        text = _load(version)
        lower = text.lower()
        assert "machine-translated" in lower or "machine translated" in lower or "pre-translated" in lower or "pretranslated" in lower
        # Kinship-mirror suppression.
        assert "bhai" in lower or "sister" in lower
        # Don't comment on language.
        assert "language" in lower

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_species_default_present(self, version):
        text = _load(version).lower()
        assert "cattle" in text and "buffalo" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_vague_query_single_question_rule(self, version):
        text = _load(version).lower()
        assert "fifteen words" in text or "15 words" in text
        # One short clarification question, not many.
        assert "exactly one" in text or "one short clarification" in text or "ask exactly" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_no_markdown_rule(self, version):
        text = _load(version).lower()
        assert "markdown" in text or "bullets" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_word_cap_present(self, version):
        text = _load(version).lower()
        assert "ninety" in text or "90 words" in text or "90 spoken words" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_closing_line_present(self, version):
        text = _load(version)
        assert "Thank you for using our service" in text
        assert "Wishing you healthy animals" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_signal_conversation_state_values(self, version):
        text = _load(version)
        assert "conversation_closing" in text
        assert "user_frustration" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_milk_collection_iso_date_rule(self, version):
        text = _load(version)
        assert "YYYY-MM-DD" in text
        assert "thirty one" in text.lower() or "31 days" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_ai_booking_no_ordinal_choice(self, version):
        text = _load(version).lower()
        assert "ordinal" in text or "first technician" in text or "position" in text

    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_size_guardrails(self, version):
        text = _load(version)
        words = len(text.split())
        lines = text.count("\n") + 1
        # New variants must stay leaner than the legacy mixed prompt.
        assert words <= 4500, f"{version} prompt grew to {words} words (limit 4500)"
        assert lines <= 320, f"{version} prompt grew to {lines} lines (limit 320)"


# ---------------------------------------------------------------------------
# Voice Examples parseability (each variant must expose at least a few canonical
# spoken examples so smoke tests and humans can read them).
# ---------------------------------------------------------------------------

class TestVoiceExamplesExist:
    @pytest.mark.parametrize("version", NEW_VARIANTS)
    def test_at_least_six_user_assistant_pairs(self, version):
        import re
        text = _load(version)
        pairs = re.findall(r"User:\s*(.+)\nAssistant:\s*(.+)", text)
        assert len(pairs) >= 6, f"{version} only exposes {len(pairs)} voice examples"
