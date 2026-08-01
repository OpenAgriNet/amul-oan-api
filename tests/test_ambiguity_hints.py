"""Disambiguation hints for technical Gujarati terms.

Chat reaches get_ambiguity_hints_for_query on every query: once via the agent's
dynamic system prompt (agents/agrinet.py) and again inside the pretranslation
glossary (app/services/translation.py -> app/services/chat.py). Without it the
translator hallucinates similar-but-wrong conditions — આફરા as "afterbirth
retention", ઇતરડી as "foot rot", ખરવા-મોવાસા as "mastitis".

Recovered from the deleted tests/test_voice_fixes.py: these cases were never
voice-specific, and losing them left the function with zero coverage.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.tools.terms import get_ambiguity_hints_for_query


def test_fat_is_milk_fat_not_stomach():
    result = get_ambiguity_hints_for_query("મારી ભેંસનું ફેટ ઓછું છે")
    assert "milk fat" in result.lower() or "ફેટ" in result
    assert "પેટ" not in result or "NOT પેટ" in result


def test_mati_na_khasvi_is_retained_placenta():
    result = get_ambiguity_hints_for_query("ગાયની માટી ન ખસવી")
    assert "retained placenta" in result.lower() or "મેલી" in result


def test_meli_is_retained_placenta():
    result = get_ambiguity_hints_for_query("મેલી ન પડી")
    assert "retained placenta" in result.lower() or "afterbirth" in result.lower()


def test_karmodi_is_horn_cancer():
    result = get_ambiguity_hints_for_query("ગાયને કરમોડી થયો છે")
    assert "horn cancer" in result.lower() or "કરમોડી" in result


def test_vado_is_shed_not_calf():
    result = get_ambiguity_hints_for_query("વાડો કેવી રીતે બનાવવો")
    assert "shed" in result.lower() or "enclosure" in result.lower()
    assert "પાડો" not in result or "NOT પાડો" in result


def test_samudri_feed_avoids_marine_assumption():
    result = get_ambiguity_hints_for_query("ગાભણ ભેંસને સમુદ્રી દાણ આપવું?")
    assert "repeat" in result.lower() or "clarify" in result.lower() or "સ્પષ્ટ" in result
    assert "seaweed" in result.lower() or "marine feed" in result.lower()


@pytest.mark.parametrize("query", [
    "મારી ભેસ્ટને તાવ છે",
    "ભંચ દૂધ ઓછું આપે છે",
    "ભેંચને ખાવાનું બંધ છે",
])
def test_buffalo_asr_variants_do_not_become_sheep(query):
    """STT mangles ભેંસ in several ways; all must resolve to buffalo."""
    result = get_ambiguity_hints_for_query(query)
    assert "buffalo" in result.lower()
    assert "not sheep" in result.lower() or "NOT sheep" in result
    assert "goat" in result.lower()


def test_uthla_is_repeat_breeder():
    result = get_ambiguity_hints_for_query("મારી ગાય ઉથલા મારે છે")
    assert "repeat breeder" in result.lower()


def test_unmatched_query_returns_a_string():
    assert isinstance(get_ambiguity_hints_for_query("દૂધ કેવી રીતે વધારવું"), str)


def test_include_ask_false_is_the_chat_pretranslation_path():
    """Chat's pretranslation passes include_ask=False; it must not emit ask-the-farmer
    prompts into a glossary that is fed to a translator."""
    with_ask = get_ambiguity_hints_for_query("ગાભણ ભેંસને સમુદ્રી દાણ આપવું?")
    without_ask = get_ambiguity_hints_for_query(
        "ગાભણ ભેંસને સમુદ્રી દાણ આપવું?", include_ask=False,
    )
    assert isinstance(without_ask, str)
    assert len(without_ask) <= len(with_ask)
