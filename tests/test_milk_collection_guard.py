"""Tests for the get_farmer_milk_collection_details availability guard + code
validation.

Root cause (systematic-debugging): the tool was exposed to the LLM with required,
LLM-supplied union/society/farmer codes but no availability guard and no code
validation. With no farmer context the LLM hallucinated placeholder codes (0/0/0)
that reached the live backend (PashuGPT 500). These tests pin both fixes:
  - prepare-callback hides the tool unless a farmer is resolved (farmer_unions).
  - the tool refuses missing/placeholder codes without calling the backend.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

import agents.tools.milk_collection as mc


def _ctx(unions):
    return SimpleNamespace(deps=SimpleNamespace(farmer_unions=unions))


# ── prepare-callback: hide the tool when no farmer is resolved ─────────────────

def test_prepare_hides_tool_when_no_farmer_unions():
    sentinel = object()
    assert (
        asyncio.run(mc.prepare_get_farmer_milk_collection_details(_ctx([]), sentinel))
        is None
    )
    assert (
        asyncio.run(mc.prepare_get_farmer_milk_collection_details(_ctx(None), sentinel))
        is None
    )
    # whitespace-only entries don't count as a resolved farmer
    assert (
        asyncio.run(mc.prepare_get_farmer_milk_collection_details(_ctx(["  "]), sentinel))
        is None
    )


def test_prepare_shows_tool_when_farmer_resolved():
    sentinel = object()
    out = asyncio.run(
        mc.prepare_get_farmer_milk_collection_details(_ctx(["banas"]), sentinel)
    )
    assert out is sentinel


# ── tool refuses placeholder / missing codes WITHOUT hitting the backend ──────

@pytest.mark.parametrize(
    "union,society,farmer",
    [
        ("0", "0", "0"),      # the observed hallucination
        ("", "1066", "123"),  # empty union
        ("2021", "0", "123"), # placeholder society
        ("2021", "1066", ""), # empty farmer
    ],
)
def test_tool_refuses_placeholder_codes(monkeypatch, union, society, farmer):
    called = False

    async def fake_api(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(mc, "get_farmer_milk_collection_details_api", fake_api)
    monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

    out = asyncio.run(
        mc.get_farmer_milk_collection_details(
            union, society, farmer, "2026-01-01", "2026-01-10"
        )
    )
    assert "failed" in out.lower()
    assert called is False  # the live backend must never be reached with junk codes


def test_tool_calls_backend_with_valid_codes(monkeypatch):
    captured = {}

    async def fake_api(request, token):
        captured["request"] = request
        captured["token"] = token
        return SimpleNamespace(milk=[], deduction=[])

    monkeypatch.setattr(mc, "get_farmer_milk_collection_details_api", fake_api)
    monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

    out = asyncio.run(
        mc.get_farmer_milk_collection_details(
            "2021", "1066", "123", "2026-01-01", "2026-01-10"
        )
    )
    assert "successfully" in out.lower()
    assert captured["request"].union_code == "2021"
    assert captured["request"].society_code == "1066"
    assert captured["request"].farmer_code == "123"


# ── lenient model (#12) + None-safe markdown formatter coupling ────────────────

def test_tool_formats_partial_rows_without_crashing(monkeypatch):
    """The lenient FarmerMilkCollection model (#12) allows None fields (partial
    PashuGPT rows). Chat's markdown formatter must render them as '-' rather than
    crash on f-string/`.replace` of None."""
    from app.models.milk_collection import (
        FarmerMilkCollectionResponseModel,
        MilkCollectionRecordModel,
        DeductionRecordModel,
    )

    async def fake_api(request, token):
        return FarmerMilkCollectionResponseModel(
            milk=[MilkCollectionRecordModel(
                date="2026-01-01", shift="M", qty=12.0, fat=None, snf=None, amount=None,
            )],
            deduction=[DeductionRecordModel(date=None, account_name=None, amount=None)],
        )

    monkeypatch.setattr(mc, "get_farmer_milk_collection_details_api", fake_api)
    monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

    out = asyncio.run(
        mc.get_farmer_milk_collection_details(
            "2021", "1066", "123", "2026-01-01", "2026-01-10"
        )
    )
    assert "successfully" in out.lower()
    assert "-" in out  # None fields rendered as '-', no crash
