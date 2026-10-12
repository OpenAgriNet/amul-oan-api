"""તણખી (Tanakhi) is upward fixation of patella, not તણખા (sparks). AMUL-84.

The term had no rule on either channel, so the pretranslation leg read it as the
one-letter-away everyday word તણખા and the agent could not place the problem.
The fuzzy matcher (partial_ratio >= 80) decides which spellings are safe
triggers: તનખી scores 85 against તનખા (salary) and romanised tankhi scores 83
against tanki (water tank), so neither is a trigger; tanakhi already scores 85
against a spoken/typed tankhi.
"""
import pytest

from agents.voice.tools.terms import get_ambiguity_hints_for_query, get_mini_glossary_for_text

TANAKHI = [
    "મારી ગાયને તણખી થઈ છે, પગ ખેંચાય છે",
    "બળદને તણખીની તકલીફ છે",
    "ભેંસને તણખિ છે",
    "gaay ne tankhi thai che",
]

LOOKALIKES = [
    "ચૂલામાંથી તણખા ઉડે છે",     # sparks, in a sentence
    "તણખા ઉડે છે",               # sparks, short query — scored 85 before the window fix
    "તનખા ક્યારે મળશે",          # salary
    "paani ni tanki saaf karvi",  # water tank, in a sentence
    "tanki",                      # water tank, bare — scored 80 before the window fix
    "tanki cleaning",
    "tank",
    "tank cleaning",
    "thanki",
    "ટાંકી સાફ કરવી",
]


@pytest.mark.parametrize("text", TANAKHI)
def test_rule_fires_on_the_pretranslation_leg(text):
    """include_ask=False is the pretranslation path — the leg that misread it."""
    hints = get_ambiguity_hints_for_query(text, include_ask=False)
    assert "upward fixation of patella" in hints.lower(), text


@pytest.mark.parametrize("text", LOOKALIKES)
def test_rule_does_not_fire_on_lookalike_words(text):
    hints = get_ambiguity_hints_for_query(text, include_ask=False)
    assert "upward fixation of patella" not in hints.lower(), text


def test_answer_translates_back_to_the_farmers_word():
    # Same threshold/max_terms as app/services/translation.py uses on the reply leg.
    mini = get_mini_glossary_for_text(
        "Upward fixation of patella makes the hind leg lock.",
        threshold=0.90, max_terms=40,
    )
    assert "Upward Fixation of Patella -> તણખી" in mini


def test_every_ambiguity_term_still_triggers_its_own_rule():
    """Guard for the whole file, not just તણખી: window matching must never stop a
    term from pulling the rule it belongs to."""
    import json
    from pathlib import Path

    entries = json.loads(
        (Path(__file__).resolve().parents[1] / "assets" / "voice_ambiguity_terms.json").read_text(encoding="utf-8")
    )
    missed = [
        (term, entry["rule"][:40])
        for entry in entries
        for term in entry["gu_terms"]
        if entry["rule"] not in get_ambiguity_hints_for_query(term)
    ]
    assert missed == []
