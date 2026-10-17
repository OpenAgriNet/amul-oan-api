"""With an empty AI-technician list the agent offered "Ramesh Patel or Suresh
Patel" — the example names from its own system prompt — as real options: 199
turns / 68 callers in 7d on voice-production (to 2026-09-08), 199/199 with no
technician in context. Callers repeated the names back and the agent booked with
invented identifiers (29 partner calls, 29 HTTP 500). Three defences, one class each."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.tools.beckn import network as beckn_network
from agents.voice.models.ai_call import AISpecies
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from agents.voice.tools import ai_call as ai_mod
from app.voice.farmer import _build_ai_technician_summary

PROMPTS = Path(__file__).resolve().parents[1] / "assets" / "prompts"
PROMPT_FILES = [
    "voice_system_translation_pipeline_en.md",
    "voice_system_translation_pipeline_gpt5_1_en.md",
    "voice_system_translation_pipeline_gemma4_en.md",
]
LEAKED_NAMES = ["Ramesh Patel", "Suresh Patel", "Mahesh Parmar", "રાકેશ પટેલ", "સુરેશ પટેલ"]
VALID_TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="  # 24 base64 chars — the real prod shape


@pytest.mark.parametrize("filename", PROMPT_FILES)
def test_prompt_has_no_speakable_technician_name(filename):
    text = (PROMPTS / filename).read_text(encoding="utf-8")
    for name in LEAKED_NAMES:
        assert name not in text, f"{filename} still offers {name!r} for the model to speak"
    assert "never a name from these instructions or an example" in text


def _envelope(technicians):
    return FarmerDataEnvelope(
        farmers=[FarmerRecord(farmerName="Rameshbhai", farmerCode="F123",
                              societyCode="1066", unionCode="2021")],
        aiTechnicians=technicians,
    )


def test_empty_technician_context_forbids_naming():
    summary = _build_ai_technician_summary(_envelope([]))
    assert "Do NOT name any technician" in summary


def test_populated_technician_context_is_unchanged():
    summary = _build_ai_technician_summary(_envelope([{
        "farmerName": "Rameshbhai", "societyName": "Anand Dairy Society",
        "societyCode": "1066", "unionCode": "2021",
        "technicians": [{"userId": VALID_TECH_ID, "fullName": "Real Name",
                         "mobileNumber": "9876543210"}],
    }]))
    assert f"id={VALID_TECH_ID} full_name=Real Name" in summary
    assert "Do NOT name any technician" not in summary


# (union, society, farmer, technician id) seen hitting the partner API in prod
@pytest.mark.parametrize("identifiers", [
    ("U11223", "S67890", "F12345", "T55667"),
    ("UNION_CODE_FROM_CONTEXT", "SOCIETY_CODE_FROM_CONTEXT", "647", "SURESH_PATEL_ID_FROM_CONTEXT"),
    ("MISSING", "NOT_AVAILABLE", "None", "T001"),
    ("159", "00002", "Rathod Sanjay Shri Jagats", "/cT4TzbfxFOo+L+ZN9x1ZQ=="),
    ("2017", "00699", "0739", "Hiteshbhai Patel"),
    ("", "", "", ""),
])
def test_invented_identifiers_are_rejected(identifiers):
    assert ai_mod._invalid_booking_identifier(*identifiers) is not None


# real triples from successful prod bookings — codes are NOT always numeric
@pytest.mark.parametrize("codes", [
    ("159", "00002", "5058"), ("M001", "2169", "0092"),
    ("2021", "NA4192", "NA0001"), ("2004", "55", "NA01"),
])
def test_real_prod_identifiers_pass(codes):
    assert ai_mod._invalid_booking_identifier(*codes, VALID_TECH_ID) is None


def test_invented_booking_never_reaches_the_partner_api(monkeypatch):
    calls = {"n": 0}

    async def fake_confirm(*args, **kwargs):  # pragma: no cover - must not run
        calls["n"] += 1

    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", fake_confirm)

    async def in_scope():
        return True

    ctx = SimpleNamespace(deps=SimpleNamespace(session_id="s1", ensure_in_scope=in_scope,
                                               farmer_unions=[]))
    out = asyncio.run(ai_mod.create_ai_call(ctx, "U11223", "S67890", "F12345", "T55667",
                                            next(iter(AISpecies))))
    assert calls["n"] == 0
    assert out == ai_mod.INVALID_IDENTIFIERS_MESSAGE
