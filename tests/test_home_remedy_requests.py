"""AMUL-75: a farmer asked Sarlaben for a home remedy (ઘરેલુ ઉપાય) five times
on one call and was refused every time with "no home remedy can help, consult
your vet", even after saying the vet had already given up on the animal.

No prompt said anything about home remedies, so the model fell back on "never
fabricate treatments" and the vet referral and never searched the
ethnoveterinary material. These tests pin the routing rule in every prompt
that answers farmers and the search expansion that adds the corpus term.
"""
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.search import _expand_veterinary_synonyms
from agents.voice.tools.search import _expand_home_remedy_query
from helpers.utils import get_prompt

VOICE_PROMPTS = (
    "voice_system_translation_pipeline_en",
    "voice_system_translation_pipeline_gemma4_en",
    "voice_system_translation_pipeline_gpt5_1_en",
)
VOICE_RENDER_CONTEXT = {
    "creation_date_words": "eleventh February two thousand twenty six",
    "service_channels_words": "chat, voice call, and WhatsApp",
    "helpline_number_words": "zero eight zero three five four five three five four five",
}


def _chat_prompt() -> str:
    return get_prompt("agrinet_system_translation_pipeline.md", context={
        "today_date": "26-10-2026",
        "farmer_context": None, "ambiguity_hints": None,
        "response_max_chars": None, "loan_max_amount": "5,000",
        "loan_interest_rate_pct": "7", "network_tools_enabled": True,
        "vistaar_shc_enabled": False,
    })


def _farmer_prompts() -> list[tuple[str, str]]:
    voice = [(name, get_prompt(name, context=VOICE_RENDER_CONTEXT)) for name in VOICE_PROMPTS]
    return [*voice, ("agrinet_system_translation_pipeline", _chat_prompt())]


@pytest.mark.parametrize("name,text", _farmer_prompts())
def test_home_remedy_requests_search_the_ethnoveterinary_material(name, text):
    assert "`ethnoveterinary home remedy`" in text, name


@pytest.mark.parametrize("name,text", _farmer_prompts())
def test_home_remedies_are_never_refused_on_principle(name, text):
    lowered = text.lower()
    assert "never refuse" in lowered, name
    assert "on principle" in lowered, name


@pytest.mark.parametrize("name,text", _farmer_prompts())
def test_a_missing_remedy_is_never_invented(name, text):
    lowered = text.lower()
    assert "inventing one" in lowered or "invent one" in lowered, name


def test_prod_voice_prompt_keeps_the_vet_has_seen_it_and_drench_safety_rules():
    # `mixed` is the prod variant (amul-infra prod amul-oan-api-pipelines.yaml).
    text = get_prompt("voice_system_translation_pipeline_en", context=VOICE_RENDER_CONTEXT)
    assert "vet has already examined the animal" in text
    assert "cannot be saved" in text
    assert "lying flat or cannot swallow" in text
    assert 'never "ethnoveterinary"' in text


@pytest.mark.parametrize("expand", [_expand_home_remedy_query, _expand_veterinary_synonyms])
@pytest.mark.parametrize("query", [
    "cow constipation home remedy",
    "cow not urinating home remedies",
    "buffalo bloat desi treatment",
    "cow fever traditional medicine",
    "calf diarrhoea gharelu remedy",
])
def test_home_remedy_queries_add_the_corpus_term(expand, query):
    expanded = expand(query)
    assert expanded.startswith(query)
    assert expanded.split().count("ethnoveterinary") == 1


@pytest.mark.parametrize("expand", [_expand_home_remedy_query, _expand_veterinary_synonyms])
def test_an_ethnoveterinary_query_is_not_doubled(expand):
    expanded = expand("cow mastitis ethnoveterinary home remedy")
    assert expanded.lower().split().count("ethnoveterinary") == 1


@pytest.mark.parametrize("expand", [_expand_home_remedy_query, _expand_veterinary_synonyms])
@pytest.mark.parametrize("query", [
    "desi cow milk yield",
    "cow mastitis treatment",
    "home made cattle feed",
    "buffalo heat detection timing",
])
def test_other_queries_are_left_alone(expand, query):
    assert "ethnoveterinary" not in expand(query)
