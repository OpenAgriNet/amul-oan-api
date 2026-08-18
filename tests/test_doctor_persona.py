from agents.doctor import DOCTOR_TOOLS
from app.config import settings
from app.personas import resolve_chat_persona
from helpers.utils import get_prompt


def test_signed_doctor_claim_selects_doctor_persona(monkeypatch):
    monkeypatch.setattr(settings, "chat_persona_override_enabled", False)

    assert resolve_chat_persona({"user_type": "doctor"}) == "doctor"
    assert resolve_chat_persona({"user_type": "farmer"}) == "farmer"
    assert resolve_chat_persona({}) == "farmer"


def test_request_override_is_guarded_by_backend_flag(monkeypatch):
    monkeypatch.setattr(settings, "chat_persona_override_enabled", False)
    assert resolve_chat_persona({"user_type": "farmer"}, "doctor") == "farmer"

    monkeypatch.setattr(settings, "chat_persona_override_enabled", True)
    assert resolve_chat_persona({"user_type": "farmer"}, "doctor") == "doctor"
    assert resolve_chat_persona({"user_type": "doctor"}, "farmer") == "farmer"


def test_doctor_agent_exposes_only_document_search():
    assert len(DOCTOR_TOOLS) == 1
    assert DOCTOR_TOOLS[0].name == "search_documents"


def test_doctor_prompt_requires_evidence_and_explicit_corpus_gaps():
    prompt = get_prompt(
        "doctor_system_translation_pipeline.md",
        context={"today_date": "Tuesday, 18 August 2026"},
    )

    assert "credentialed veterinary doctor" in prompt
    assert "Use `search_documents(query, top_k)` before answering" in prompt
    assert "Not covered in the current document corpus." in prompt
    assert "Never invent, infer, extrapolate" in prompt
    assert "Do not offer or initiate a health call" in prompt
