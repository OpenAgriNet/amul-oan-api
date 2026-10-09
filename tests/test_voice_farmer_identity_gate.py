"""The invariant from issue #282: if farmer identity is unresolved for a turn,
no tool that requires farmer identity is offered on that turn.

Voice rendered "no record exists" and "we have not resolved this caller yet"
identically — as an empty farmer block — while still exposing every
identity-taking tool. The model filled the required code slots with `MISSING`,
`UNKNOWN`, `F12345`, `UNION_CODE_FROM_CONTEXT` and farmer names. Measured
2026-09-01..09-14 on voice-production: create_ai_call 10.4% and
get_farmer_milk_collection_details 10.2%.

The prompt already told the model to stop (see the create_ai_call docstring and
voice_system_translation_pipeline_en.md), and that instruction is exactly what
those rates ignore. These tests pin the structural fix instead: the tool is
absent from the schema, and the context says what is unavailable and why.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
from types import SimpleNamespace

import pytest

import app.voice.farmer as voice
from app.voice.agent_input import _runtime_context_message
from agents.deps import FarmerAccount, FarmerContext
from agents.tools.models.farmer_transport import FarmerDataEnvelope
from agents.voice.services import farmer_identity as fi
from agents.voice.tools import BASE_TOOLS


def _found_envelope():
    return FarmerDataEnvelope.from_records(
        [{"farmerName": "Rameshbhai", "farmerCode": "5058",
          "societyCode": "00002", "unionCode": "159"}],
        source="api",
        lookup_status="found",
    )


def _deps(state, accounts):
    return FarmerContext(
        query="q", signed_in=True, mobile="9876543210",
        farmer_profile_status=fi.profile_status_for_state(state), farmer_accounts=accounts,
    )


_REAL_ACCOUNT = FarmerAccount(union_code="159", society_code="00002", farmer_code="5058")


# ── State derivation ────────────────────────────────────────────────────────
# Causes A and D in the issue (cold-fetch timeout, refresh-lock race) both
# surface as a bare None from the cache layer. Reading that as "not_found" would
# tell a registered farmer their number is unregistered.

def test_none_envelope_is_unresolved_not_not_found():
    assert fi.identity_state_for_envelope(None) == fi.UNRESOLVED


def test_explicit_not_found_is_not_found():
    env = FarmerDataEnvelope.not_found(source="api")
    assert fi.identity_state_for_envelope(env) == fi.NOT_FOUND


def test_empty_envelope_without_verdict_is_unresolved():
    """We never claim upstream said "no such farmer" when it said nothing."""
    assert fi.identity_state_for_envelope(FarmerDataEnvelope()) == fi.UNRESOLVED


def test_records_present_is_found():
    assert fi.identity_state_for_envelope(_found_envelope()) == fi.FOUND


# ── The gate ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state", [fi.UNRESOLVED, fi.NOT_FOUND])
def test_tool_is_withheld_when_identity_is_not_found(state):
    tool_def = SimpleNamespace(name="create_ai_call")
    ctx = SimpleNamespace(deps=_deps(state, [_REAL_ACCOUNT]))
    assert asyncio.run(fi.prepare_requires_farmer_identity(ctx, tool_def)) is None


def test_tool_is_withheld_when_found_but_no_complete_account():
    """_collect_farmer_accounts drops records missing any of the three codes, so
    a "found" caller can still have nothing to book with."""
    tool_def = SimpleNamespace(name="get_farmer_milk_collection_details")
    ctx = SimpleNamespace(deps=_deps(fi.FOUND, []))
    assert asyncio.run(fi.prepare_requires_farmer_identity(ctx, tool_def)) is None


def test_tool_is_offered_when_identity_is_usable():
    tool_def = SimpleNamespace(name="create_ai_call")
    ctx = SimpleNamespace(deps=_deps(fi.FOUND, [_REAL_ACCOUNT]))
    assert asyncio.run(fi.prepare_requires_farmer_identity(ctx, tool_def)) is tool_def


def test_deps_default_fails_closed():
    """A FarmerContext built without the field must not unlock the tools."""
    assert fi.has_usable_farmer_identity(FarmerContext(query="q")) is False


def _tools_offered_to_the_model(deps: FarmerContext) -> list[str]:
    """Tool names in the schema an actual agent run hands the model.

    Asserting on the prepare callback alone would only prove the callback
    returns None; it would not prove pydantic-ai drops the tool from what the
    model can see. This drives a real Agent over a FunctionModel and reads the
    tool definitions it was actually given.
    """
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    seen: dict[str, list[str]] = {}

    def capture(messages, info: AgentInfo) -> ModelResponse:
        seen["tools"] = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent = Agent(
        model=FunctionModel(capture),
        deps_type=FarmerContext,
        tools=BASE_TOOLS,
        output_type=str,
    )
    asyncio.run(agent.run("hi", deps=deps))
    return seen["tools"]


@pytest.mark.parametrize("state,accounts", [
    (fi.UNRESOLVED, [_REAL_ACCOUNT]),   # cold-fetch timeout / refresh race (causes A, D)
    (fi.NOT_FOUND, [_REAL_ACCOUNT]),    # caller genuinely unregistered (cause C)
    (fi.FOUND, []),                     # records returned, no complete code triple
])
def test_model_is_not_offered_identity_tools_without_identity(state, accounts):
    """The #282 invariant, end to end: the model cannot call what it cannot see."""
    offered = _tools_offered_to_the_model(_deps(state, accounts))
    for name in fi.IDENTITY_GATED_TOOLS:
        assert name not in offered
    # Retrieval is unaffected — the caller can still ask ordinary questions.
    assert "search_documents" in offered


def test_model_is_offered_identity_tools_when_identity_is_usable():
    offered = _tools_offered_to_the_model(_deps(fi.FOUND, [_REAL_ACCOUNT]))
    for name in fi.IDENTITY_GATED_TOOLS:
        assert name in offered


def test_every_identity_taking_tool_carries_the_gate():
    """A new identity-taking tool must not be able to ship unguarded."""
    gated = {
        t.name for t in BASE_TOOLS
        if getattr(t, "prepare", None) is fi.prepare_requires_farmer_identity
    }
    assert gated == set(fi.IDENTITY_GATED_TOOLS)


# ── What the model is told ──────────────────────────────────────────────────

def test_found_state_adds_no_capability_lines():
    assert fi.unavailable_capability_lines(fi.FOUND) == []


def test_not_found_tells_the_caller_to_register():
    text = "\n".join(fi.unavailable_capability_lines(fi.NOT_FOUND))
    assert "not registered" in text
    assert "milk society" in text
    assert "try again" not in text


def test_unresolved_tells_the_caller_to_retry_and_not_that_it_is_missing():
    text = "\n".join(fi.unavailable_capability_lines(fi.UNRESOLVED))
    assert "try again shortly" in text
    assert "not registered" not in text
    # The failure mode a hidden tool invites: the model telling the caller the
    # service does not exist, when it exists and simply could not run.
    assert "Do not say the service does not exist." in text


@pytest.mark.parametrize("state", [fi.UNRESOLVED, fi.NOT_FOUND])
def test_capability_lines_name_every_gated_tool(state):
    text = "\n".join(fi.unavailable_capability_lines(state))
    for capability in fi._GATED_CAPABILITY_NAMES:
        assert capability in text


# ── The blank block that started it ─────────────────────────────────────────

@pytest.mark.parametrize("envelope,expected", [
    (None, "could not be loaded"),
    (FarmerDataEnvelope.not_found(source="api"), "no farmer record is registered"),
])
def test_summary_is_no_longer_blank(envelope, expected):
    summary = voice._build_compact_farmer_summary(envelope)
    assert summary != ""
    assert expected in summary


def test_found_summary_is_unchanged():
    summary = voice._build_compact_farmer_summary(_found_envelope())
    assert "- Farmer name: Rameshbhai" in summary
    assert "Unavailable for this turn" not in summary


# ── The tool-groups line ────────────────────────────────────────────────────
# It hardcoded "booking", advertising it on exactly the turns where booking was
# impossible.

@pytest.mark.parametrize("state", [fi.UNRESOLVED, fi.NOT_FOUND])
def test_booking_group_not_advertised_without_identity(state):
    assert "booking" not in fi.identity_tool_groups(_deps(state, [_REAL_ACCOUNT]))


def test_booking_group_advertised_with_identity():
    groups = fi.identity_tool_groups(_deps(fi.FOUND, [_REAL_ACCOUNT]))
    assert "booking" in groups
    assert "signed-in-farmer-data" in groups


def test_runtime_context_does_not_promise_farmer_tools_when_unresolved():
    message = _runtime_context_message(_deps(fi.UNRESOLVED, [_REAL_ACCOUNT]))
    assert "- Farmer-data tools are not available for this turn." in message


# ── Backstops ───────────────────────────────────────────────────────────────
# These run below the gate. They exist so a tool reached by another path still
# cannot forward an invented code to the partner API.

@pytest.mark.parametrize("code", [
    "MISSING", "UNKNOWN", "unknown", "not_provided", "not_available",
    "NA", "None", "PLACEHOLDER", "UNION_CODE_FROM_CONTEXT", "",
])
def test_digit_rule_rejects_invented_codes(code):
    """Every real code carries a digit — zero exceptions across 28,089 bookings.
    Shape alone accepted MISSING, UNKNOWN, NA, None and PLACEHOLDER."""
    assert fi.invalid_identity_code_field(code, "00002", "5058") == "union_code"


@pytest.mark.parametrize("codes", [
    ("159", "00002", "5058"), ("M001", "2169", "0092"),
    ("2021", "NA4192", "NA0001"), ("2004", "55", "NA01"),
])
def test_real_prod_codes_pass(codes):
    assert fi.invalid_identity_code_field(*codes) is None


def test_digit_bearing_invention_caught_by_context_crosscheck():
    """F12345 / U11223 pass shape and digit; only the caller's own accounts
    distinguish them."""
    assert fi.invalid_identity_code_field(
        "U11223", "S67890", "F12345", [_REAL_ACCOUNT]
    ) == "codes_not_in_context"


def test_crosscheck_accepts_the_callers_own_account():
    assert fi.invalid_identity_code_field(
        "159", "00002", "5058", [_REAL_ACCOUNT]
    ) is None


def test_milk_lookup_refuses_instead_of_using_model_codes():
    """The fallback returned "fetched successfully - no milk collection records"
    for 96 farmers who did have records. Refusing is the correct failure."""
    from agents.voice.tools.milk_collection import get_farmer_milk_collection_details

    ctx = SimpleNamespace(deps=_deps(fi.UNRESOLVED, []))
    out = asyncio.run(get_farmer_milk_collection_details(
        ctx, "U11223", "S67890", "F12345", "2026-09-01", "2026-09-14",
    ))
    assert "not available" in out
    assert "fetched successfully" not in out
