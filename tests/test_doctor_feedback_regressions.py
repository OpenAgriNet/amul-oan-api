import asyncio
import os
from importlib import import_module

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.search import _expand_veterinary_synonyms
from agents.tools.terms import get_ambiguity_hints_for_query
from app.config import settings
from app.personas import history_session_id_for_persona
from app.services.chat import _sanitize_doctor_stream, sanitize_doctor_answer
from app.services.identity_profile import build_doctor_identity_response
from app.services.translation import _enforce_clinical_pretranslation_terms
from helpers.utils import get_prompt


def test_doctor_identity_never_uses_sarlaben():
    english = build_doctor_identity_response("en", "english", "who are you")
    gujarati = build_doctor_identity_response("gu", "gu", "તમે કોણ છો?")

    assert "Amul Veterinary Assistant" in english
    assert "Sarlaben" not in english
    assert "વેટરનરી આસિસ્ટન્ટ" in gujarati
    assert "સરલાબેન" not in gujarati


def test_doctor_and_farmer_histories_are_isolated():
    assert history_session_id_for_persona("session-1", "farmer") == "session-1"
    assert history_session_id_for_persona("session-1", "doctor") == "session-1:persona:doctor"


def test_doctor_moderation_prompt_allows_clinical_questions_and_keeps_harm_gate():
    prompt = get_prompt("doctor_moderation_system")

    assert "dosage" in prompt
    assert "Do not block" in prompt
    assert "Deliberate animal cruelty" in prompt
    assert "Proceed with the query." in prompt


def test_calf_scours_ambiguity_hint_is_injected():
    hint = get_ambiguity_hints_for_query(
        "વાછરડા ને જાડા થયા હોય તો શું કરવું?", include_ask=False
    )
    assert "calf scours" in hint.lower()
    assert "not fatness" in hint.lower()


@pytest.mark.parametrize(
    "bad_translation",
    [
        "What should be done if the calves have become fat?",
        "How should obese calves be treated?",
        "What to do for overweight calves?",
    ],
)
def test_calf_scours_pretranslation_cannot_become_obesity(bad_translation):
    corrected = _enforce_clinical_pretranslation_terms(
        "વાછરડા ને જાડા થયા હોય તો શું કરવું?", bad_translation
    )
    assert "diarrhea" in corrected.lower() or "scours" in corrected.lower()
    assert "obese" not in corrected.lower()
    assert "overweight" not in corrected.lower()
    assert "become fat" not in corrected.lower()
    assert "calves have diarrhea" in corrected.lower() or "calves with diarrhea" in corrected.lower()


def test_clinical_query_expansion_covers_corpus_synonyms():
    milk_fever = _expand_veterinary_synonyms("milk fever definition cattle buffalo")
    scours = _expand_veterinary_synonyms("calf scours treatment")
    ethnoveterinary_scours = _expand_veterinary_synonyms(
        "calf scours ethnoveterinary ingredients"
    )

    assert "hypocalcemia" in milk_fever
    assert "parturient paresis" in milk_fever
    assert "diarrhea" in scours
    assert "oral rehydration" in scours
    assert "electrolytes" in scours
    assert "dehydration" in scours
    assert "oral rehydration" not in ethnoveterinary_scours


def test_doctor_prompt_prioritizes_standard_calf_scours_care():
    prompt = get_prompt(
        "doctor_system_translation_pipeline.md",
        context={"today_date": "Friday, 21 August 2026"},
    )
    assert "Standard-of-Care Priority" in prompt
    assert "replacement of water and electrolytes" in prompt
    assert "oral versus intravenous fluid criteria" in prompt
    assert "controlled substance" in prompt


def test_doctor_answer_removes_english_and_gujarati_document_references():
    answer = (
        "- Give oral electrolytes.\n"
        "**Sources:** NDDB Handbook; Veterinary App PDF\n"
        "- Reassess dehydration. (source: NDDB)\n"
        "સંદર્ભ: પશુપાલન માર્ગદર્શિકા\n"
    )
    cleaned = sanitize_doctor_answer(answer)

    assert "Give oral electrolytes" in cleaned
    assert "Reassess dehydration" in cleaned
    assert "NDDB" not in cleaned
    assert "સંદર્ભ" not in cleaned


def test_doctor_stream_removes_reference_label_split_across_chunks():
    async def source():
        for chunk in [
            "- Give oral electrolytes.\n**Sou",
            "rces:** NDDB Handbook\n- Reassess dehydration.\n",
        ]:
            yield chunk

    async def collect():
        return "".join([chunk async for chunk in _sanitize_doctor_stream(source())])

    output = asyncio.run(collect())
    assert "Give oral electrolytes" in output
    assert "Reassess dehydration" in output
    assert "Sources" not in output
    assert "NDDB" not in output


def test_doctor_identity_short_circuit_bypasses_both_moderation_and_rag(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    chat = import_module("app.services.chat")
    recorded = {"history_key": None}

    monkeypatch.setattr(settings, "chat_persona_override_enabled", True)
    monkeypatch.setattr(chat, "propagate_attributes", None)
    monkeypatch.setattr(chat, "get_langfuse_client", None)

    async def update_history(key, _messages):
        recorded["history_key"] = key

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("identity short circuit must bypass model calls")

    monkeypatch.setattr(chat, "update_message_history", update_history)
    monkeypatch.setattr(chat.moderation_agent, "run", unexpected)
    monkeypatch.setattr(chat.doctor_moderation_agent, "run", unexpected)
    monkeypatch.setattr(chat, "translate_to_english_pretranslation", unexpected)
    monkeypatch.setattr(chat.doctor_agent, "iter", unexpected)

    async def collect():
        return "".join(
            [
                chunk
                async for chunk in chat.stream_chat_messages(
                    query="who are you",
                    session_id="doctor-identity",
                    source_lang="en",
                    target_lang="english",
                    channel="web",
                    user_id="anonymous",
                    history=[],
                    user_info={"user_type": "farmer"},
                    background_tasks=fastapi.BackgroundTasks(),
                    requested_persona="doctor",
                    history_session_id="doctor-identity:persona:doctor",
                )
            ]
        )

    response = asyncio.run(collect())
    assert "Amul Veterinary Assistant" in response
    assert "Sarlaben" not in response
    assert recorded["history_key"] == "doctor-identity:persona:doctor"
