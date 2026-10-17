"""create_ai_call is only offered, and only runs, when a farmer record was resolved.

Without a profile the agent has no real booking identifiers or technician
options, so the tool is hidden, the technician/account lookups never run, and
the prompt tells the agent what it cannot do.
"""
import asyncio
from types import SimpleNamespace

import pytest

import agents.tools.ai_call as ai_mod
import agents.tools.beckn.amul as amul
from agents.agrinet import get_agrinet_instructions
from agents.deps import FarmerContext
from agents.tools.models.ai_call import AISpecies

VALID_TECH_ID = "A" * 22 + "=="


async def _in_scope():
    return True


def _deps(status, mobile="9000000000"):
    return SimpleNamespace(
        session_id="s1",
        mobile=mobile,
        farmer_unions=[],
        farmer_profile_status=status,
        ensure_in_scope=_in_scope,
    )


@pytest.mark.parametrize("status", ["anonymous", "not_found", "unavailable"])
def test_tool_hidden_without_a_resolved_farmer(status):
    sentinel = object()
    ctx = SimpleNamespace(deps=_deps(status))
    assert asyncio.run(ai_mod.prepare_create_ai_call(ctx, sentinel)) is None


def test_tool_hidden_when_found_but_no_mobile():
    sentinel = object()
    ctx = SimpleNamespace(deps=_deps("found", mobile=None))
    assert asyncio.run(ai_mod.prepare_create_ai_call(ctx, sentinel)) is None


def test_tool_shown_for_a_resolved_farmer():
    sentinel = object()
    ctx = SimpleNamespace(deps=_deps("found"))
    assert asyncio.run(ai_mod.prepare_create_ai_call(ctx, sentinel)) is sentinel


@pytest.mark.parametrize("status", ["anonymous", "not_found", "unavailable"])
def test_call_without_profile_never_reaches_account_or_technician_lookup(monkeypatch, status):
    async def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("no upstream lookup without a farmer profile")

    monkeypatch.setattr(amul, "resolve_authenticated_account", boom)
    monkeypatch.setattr(amul, "search_ai_technicians", boom)
    monkeypatch.setattr(ai_mod, "_book_via_network", boom)

    ctx = SimpleNamespace(deps=_deps(status))
    out = asyncio.run(
        ai_mod.create_ai_call(ctx, "101", "202", "303", VALID_TECH_ID, AISpecies.COW)
    )
    assert out == ai_mod.NO_FARMER_PROFILE_MESSAGE


def _instructions(**deps):
    return get_agrinet_instructions(
        SimpleNamespace(deps=FarmerContext(query="book an AI visit", **deps))
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("anonymous", "not signed in"),
        ("not_found", "No farmer record was found"),
        ("unavailable", "could not be fetched right now"),
    ],
)
def test_prompt_states_what_is_unavailable_without_a_profile(status, reason):
    rendered = _instructions(
        farmer_profile_status=status,
        farmer_info="# Farmer Context\n\nNo farmer information found for mobile number `1`.",
        mobile="9000000000" if status != "anonymous" else None,
    )
    assert "## Farmer Profile: NOT available" in rendered
    assert reason in rendered
    assert ai_mod.NO_FARMER_PROFILE_MESSAGE in rendered
    # The raw not-found markdown is not presented as a profile.
    assert "Farmer Profile (from authenticated session)" not in rendered
    assert "create_ai_call(union_code" not in rendered


def test_prompt_for_a_resolved_farmer_keeps_profile_and_tool():
    rendered = _instructions(
        farmer_profile_status="found",
        farmer_info="# Farmer Context\n\n- **Matched farmer records:** 1",
        mobile="9000000000",
    )
    assert "Farmer Profile (from authenticated session)" in rendered
    assert "Matched farmer records" in rendered
    assert "create_ai_call(union_code" in rendered
    assert "Farmer Profile: NOT available" not in rendered
    assert ai_mod.NO_FARMER_PROFILE_MESSAGE not in rendered
