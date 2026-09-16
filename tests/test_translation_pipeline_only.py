"""The Gemma translation pipeline is the only supported chat path.

`use_translation_pipeline` used to be a public query parameter, but it already
defaulted to true and translation-required profiles forced it on; its only
remaining effect was picking the legacy `assets/prompts/agrinet_system.md`
prompt for farmer conversations. Both the flag and that prompt are gone, so
these tests pin the properties that removal is supposed to guarantee.
"""
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.agrinet import get_agrinet_instructions
from agents.deps import FarmerContext
from app.models.requests import ChatRequest
from app.services.chat import stream_chat_messages

TRANSLATION_PROMPT_MARKERS = (
    "## Critical Language Rule",
    "Always answer in **English only**.",
    "The system translates your answer to the user's language downstream.",
)


def _instructions(**deps_kwargs):
    deps = FarmerContext(query="My cow is not eating", **deps_kwargs)
    return get_agrinet_instructions(SimpleNamespace(deps=deps))


# ── the prompt ────────────────────────────────────────────────────────────────

def test_the_legacy_prompt_file_is_gone():
    assert not (ROOT / "assets" / "prompts" / "agrinet_system.md").exists()


def test_the_farmer_agent_always_renders_the_translation_pipeline_prompt():
    rendered = _instructions()
    for marker in TRANSLATION_PROMPT_MARKERS:
        assert marker in rendered


def test_a_supplied_flag_cannot_select_a_different_prompt():
    # FarmerContext ignores unknown fields, so a stale caller passing the old
    # flag gets the translation prompt anyway rather than a second code path.
    assert "use_translation_pipeline" not in FarmerContext.model_fields
    off = FarmerContext(query="q", use_translation_pipeline=False)
    assert not hasattr(off, "use_translation_pipeline")
    for marker in TRANSLATION_PROMPT_MARKERS:
        assert marker in _instructions(use_translation_pipeline=False)


def test_the_agent_has_no_prompt_branch_left():
    source = (ROOT / "agents" / "agrinet.py").read_text(encoding="utf-8")
    assert "agrinet_system.md" not in source
    assert source.count("get_prompt(") == 1


# ── the public contract ───────────────────────────────────────────────────────

def test_the_chat_request_model_no_longer_exposes_the_flag():
    assert "use_translation_pipeline" not in ChatRequest.model_fields


def test_supplying_the_flag_on_a_request_is_inert():
    request = ChatRequest(query="hello", use_translation_pipeline=False)
    assert not hasattr(request, "use_translation_pipeline")


def test_the_service_signature_no_longer_accepts_the_flag():
    assert "use_translation_pipeline" not in inspect.signature(stream_chat_messages).parameters


def test_the_generated_openapi_contract_has_no_such_parameter():
    import main

    schema = main.app.openapi()
    chat_get = schema["paths"]["/api/chat/"]["get"]
    names = {p["name"] for p in chat_get.get("parameters", [])}
    assert "query" in names, "sanity: chat query params are declared on the operation"
    assert "use_translation_pipeline" not in names


# ── the doctor persona is untouched ───────────────────────────────────────────

def test_the_doctor_prompt_selection_is_unchanged():
    source = (ROOT / "agents" / "doctor.py").read_text(encoding="utf-8")
    assert "doctor_system_translation_pipeline.md" in source
    assert source.count("get_prompt(") == 1
    assert "use_translation_pipeline" not in source
