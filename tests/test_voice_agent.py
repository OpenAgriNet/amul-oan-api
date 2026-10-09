"""Voice's agent in this repo: its tools as voice has them, its model from
llm_core, and the caller's identity read off this repo's FarmerContext."""
import asyncio
import json
import os
import typing
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.deps import FarmerAccount, FarmerContext
from agents.voice import agent as voice_agent_module
from agents.voice.services import farmer_identity as fi
from agents.voice.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS
from app.voice.farmer import _collect_farmer_accounts
from agents.tools.models.farmer_transport import FarmerDataEnvelope

_ACCOUNT = FarmerAccount(union_code="1", society_code="22", farmer_code="333")
_PROFILE_STATUSES = typing.get_args(FarmerContext.model_fields["farmer_profile_status"].annotation)


def _deps(status, accounts=()):
    return FarmerContext(query="q", farmer_profile_status=status, farmer_accounts=list(accounts))


@pytest.mark.parametrize("status, state", [
    ("found", fi.FOUND),
    ("not_found", fi.NOT_FOUND),
    ("unavailable", fi.UNRESOLVED),
    ("anonymous", fi.ANONYMOUS),
])
def test_the_identity_state_is_read_off_the_profile_status(status, state):
    assert fi.identity_state_for_deps(_deps(status)) == state


def test_every_profile_status_has_a_state():
    assert set(_PROFILE_STATUSES) == {"found", "not_found", "unavailable", "anonymous"}
    for status in _PROFILE_STATUSES:
        assert fi.identity_state_for_deps(_deps(status)) in (fi.FOUND, fi.NOT_FOUND, fi.UNRESOLVED, fi.ANONYMOUS)


def test_no_deps_or_no_status_fails_closed():
    assert fi.identity_state_for_deps(None) == fi.UNRESOLVED
    assert fi.identity_state_for_deps(SimpleNamespace()) == fi.UNRESOLVED
    assert fi.identity_state_for_deps(SimpleNamespace(farmer_profile_status="unknown")) == fi.UNRESOLVED
    # Voice's own field name means nothing here.
    assert fi.identity_state_for_deps(SimpleNamespace(farmer_identity="found")) == fi.UNRESOLVED


@pytest.mark.parametrize("status, accounts, offered", [
    ("found", [_ACCOUNT], True),
    ("found", [], False),
    ("not_found", [_ACCOUNT], False),
    ("unavailable", [_ACCOUNT], False),
    ("anonymous", [_ACCOUNT], False),
])
def test_identity_taking_tools_are_offered_only_for_a_usable_identity(status, accounts, offered):
    tool_def = SimpleNamespace(name="create_ai_call")
    ctx = SimpleNamespace(deps=_deps(status, accounts))

    result = asyncio.run(fi.prepare_requires_farmer_identity(ctx, tool_def))

    assert (result is tool_def) is offered
    assert fi.has_usable_farmer_identity(ctx.deps) is offered
    assert ("booking" in fi.identity_tool_groups(ctx.deps)) is offered


def test_a_context_built_without_a_status_offers_no_identity_tools():
    assert fi.has_usable_farmer_identity(FarmerContext(query="q", farmer_accounts=[_ACCOUNT])) is False


def test_the_agents_have_voices_tools():
    base = [
        "search_terms", "search_documents", "create_ai_call", "get_farmer_milk_collection_details",
        "signal_conversation_state", "find_nearby_vet_offices", "check_loan_eligibility",
    ]
    assert [tool.name for tool in BASE_TOOLS] == base
    assert [tool.name for tool in SIGNED_IN_FARMER_TOOLS] == ["get_union_scheme_data", "get_farmer_bonus_amount"]
    assert sorted(voice_agent_module.voice_agent._function_toolset.tools) == sorted(base)
    assert sorted(voice_agent_module.voice_agent_signed_in._function_toolset.tools) == sorted(
        [*base, "get_union_scheme_data", "get_farmer_bonus_amount"]
    )


def test_the_agents_take_their_model_from_llm_core():
    for agent in (voice_agent_module.voice_agent, voice_agent_module.voice_agent_signed_in):
        assert agent.model is None
        assert agent._deps_type is FarmerContext


def test_assets_are_found_from_outside_the_repo(monkeypatch, tmp_path):
    from pathlib import Path

    from agents.voice.tools import terms, vet_offices

    monkeypatch.chdir(tmp_path)

    assert terms._load_gu_term_policy()
    assert terms._load_ambiguity_terms() == json.loads(
        (Path(__file__).resolve().parents[1] / "assets" / "voice_ambiguity_terms.json").read_text(encoding="utf-8")
    )
    assert vet_offices._asset_path() == Path(__file__).resolve().parents[1] / "assets" / "vet_offices.json"


def test_accounts_are_collected_once_per_complete_triple():
    envelope = FarmerDataEnvelope.model_validate({
        "farmers": [
            {"unionCode": "1", "societyCode": "22", "farmerCode": "333", "farmerName": "A"},
            {"unionCode": "1", "societyCode": "22", "farmerCode": "333"},
            {"unionCode": "1", "societyCode": "22"},
        ],
    })

    assert [(a.union_code, a.society_code, a.farmer_code, a.farmer_name) for a in _collect_farmer_accounts(envelope)] == [
        ("1", "22", "333", "A"),
    ]
    assert _collect_farmer_accounts(None) == []
