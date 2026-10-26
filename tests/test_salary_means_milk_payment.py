"""A farmer's પગાર (salary) is their milk payment, not a job salary. AMUL-101.

"સોળ સપ્ટેમ્બરથી ત્રીસ નવ સુધીનો પગાર કેટલો થયો" was declined with "I can't give
salary or payment information": pretranslation rendered પગાર as "salary", and
chat moderation listed salary lookups among its hard declines, a rule written in
March, before get_farmer_milk_collection_details existed. The payment is the
Amount on the farmer's milk slips, which that tool returns.

The rule pins the gu->en leg to "milk payment" on both channels, and moderation
and the agent prompts read "salary" as the same question. Its triggers match
verbatim only ("fuzzy": false): પગાર is one letter from પાર (across) and ગાર
(mud plaster), which the fuzzy window scores at 86.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from pathlib import Path

import pytest

from agents.tools.terms import get_ambiguity_hints_for_query
from agents.voice.tools.terms import get_ambiguity_hints_for_query as voice_agent_hints
from app.services.translation import _pretranslation_system_with_glossary
from app.voice import pretranslation as vp

PROMPTS = Path(__file__).resolve().parents[1] / "assets" / "prompts"
MILK_PAYMENT = "render it as 'milk payment'"
TICKET_QUERY = "સોળ સપ્ટેમ્બરથી ત્રીસ નવ સુધીનો પગાર કેટલો થયો"

SALARY = [
    TICKET_QUERY,
    "મારો પગાર ક્યારે આવશે",
    "ગયા મહિનાનો પગાર કેટલો મળ્યો",
    "પગારની વિગત બતાવો",
    "maro pagar ketlo thayo",
    "pagaar ketlo aavyo",
]

LOOKALIKES = [
    "ભેંસ નદી પાર કરી શકે?",     # પાર (across): scores 86 against પગાર
    "છાણ-ગાર થી ગમાણ લીંપવી",     # ગાર (dung plaster): scores 86
    "ભેંસના પગમાં સોજો છે",       # પગ (leg)
    "ગાયના પગાં સુજી ગયા",
    "pagma sojo che",
    "sagar dairy",
    "pagal kutro karde to",
]

# The chat translator, voice's translator (voice's rules, chat's matcher), and
# voice's agent-side hints (its own matcher copy).
MATCHERS = {
    "chat": lambda q: get_ambiguity_hints_for_query(q, include_ask=False),
    "voice_pretranslation": lambda q: vp.get_ambiguity_hints_for_query(q, include_ask=False),
    "voice_agent": voice_agent_hints,
}


@pytest.mark.parametrize("leg", MATCHERS)
@pytest.mark.parametrize("text", SALARY)
def test_pagar_is_translated_as_milk_payment(leg, text):
    assert MILK_PAYMENT in MATCHERS[leg](text), text


@pytest.mark.parametrize("leg", MATCHERS)
def test_salary_in_the_english_the_agent_reads_pulls_the_rule(leg):
    """The agents read the pretranslated English, where તનખા or a Hindi वेतन
    arrives as "salary"."""
    assert MILK_PAYMENT in MATCHERS[leg]("How much was my salary from 16 to 30 September?")


@pytest.mark.parametrize("leg", MATCHERS)
@pytest.mark.parametrize("text", LOOKALIKES)
def test_rule_does_not_fire_on_lookalike_words(leg, text):
    assert MILK_PAYMENT not in MATCHERS[leg](text), text


def test_the_flag_is_what_keeps_the_lookalikes_out():
    rule = {"gu_terms": ["પગાર"], "rule": "test"}
    assert get_ambiguity_hints_for_query("નદી પાર કરી", terms=[rule]) == "- test"
    assert get_ambiguity_hints_for_query("નદી પાર કરી", terms=[{**rule, "fuzzy": False}]) == ""
    assert get_ambiguity_hints_for_query("મારો પગાર", terms=[{**rule, "fuzzy": False}]) == "- test"


def test_both_translators_are_told_to_say_milk_payment():
    assert MILK_PAYMENT in _pretranslation_system_with_glossary(TICKET_QUERY)
    voice_system = vp._build_openai_pretranslation_messages("Gujarati", "gu", TICKET_QUERY, None)[0]["content"]
    assert MILK_PAYMENT in voice_system


def test_chat_moderation_no_longer_declines_salary():
    text = (PROMPTS / "moderation_system.md").read_text(encoding="utf-8")
    assert "passbook/salary" not in text
    assert "which farmers call their salary / પગાર" in text


def test_voice_moderation_passes_pagar_through():
    text = (PROMPTS / "voice_moderation_en.md").read_text(encoding="utf-8")
    assert "**પગાર / pagar / salary**" in text


def test_chat_agent_routes_salary_to_the_milk_tool():
    text = (PROMPTS / "agrinet_system_translation_pipeline.md").read_text(encoding="utf-8")
    assert "milk payment / salary / પગાર for some dates → `get_farmer_milk_collection_details`" in text


@pytest.mark.parametrize("name", [
    "voice_system_translation_pipeline_en.md",
    "voice_system_translation_pipeline_gemma4_en.md",
    "voice_system_translation_pipeline_gpt5_1_en.md",
])
def test_voice_agent_routes_salary_to_the_milk_tool(name):
    text = (PROMPTS / name).read_text(encoding="utf-8")
    assert "salary balance" not in text
    routing = [
        line for line in text.splitlines()
        if "S N F, milk payment" in line and "get_farmer_milk_collection_details" in line
    ]
    assert routing and all("salary" in line for line in routing)
