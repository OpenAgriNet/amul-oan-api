"""બીજદાન (insemination) must not be translated as નિદાન (diagnosis).

ASR drops the જ and returns non-words (બિદાન, બિદદાન, વિદાન…). The nearest real
Gujarati word is નિદાન — one letter away, ન vs બ — so the translator picks
"diagnosis", which is why the agent answered about diagnosis while the farmer
was asking to book an insemination. Edit distance FAVOURS the wrong answer
(બિદાન→નિદાન is 1 edit, બિદાન→બીજદાન is 2), so no similarity-based rewrite can
fix this; the rule has to teach the distinction.

Replay of 88 real utterances from the 265 "AI booking not done" calls, through
the live pretranslation leg: 1/88 correct before, 88/88 after, with all 51 true
નિદાન controls unchanged. See issue #266.
"""
from agents.voice.tools.terms import get_ambiguity_hints_for_query

CORRUPTIONS = [
    "મારે ગાયને બિદાન કરાવવાનું છે",
    "મારે ગાય નું બિદદાન કરાવવાનું છે",
    "ભેંસને વિદાન કરાવવું છે",
    "બિજદાન કરાવવાનું છે",
]


def test_rule_fires_for_every_asr_corruption():
    for text in CORRUPTIONS:
        hints = get_ambiguity_hints_for_query(text, include_ask=False)
        assert "artificial insemination" in hints.lower(), text


def test_rule_also_fires_for_genuine_nidan_and_carries_both_readings():
    """The matcher is fuzzy, so it fires on નિદાન too — by design. The rule must
    therefore state both readings, or firing on a genuine diagnosis question
    would push the translator the wrong way."""
    hints = get_ambiguity_hints_for_query("નિદાન કરવાનું છે", include_ask=False)
    assert "નિદાન" in hints
    assert "genuinely means diagnosis" in hints


def test_rule_is_not_an_ask_entry():
    """`ask` entries are stripped from the pretranslation prompt
    (include_ask=False), so this must be a hardcode entry or it never reaches
    the leg where the damage happens."""
    assert get_ambiguity_hints_for_query(CORRUPTIONS[0], include_ask=False).strip()
