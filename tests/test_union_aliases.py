"""Tests for canonical union-name normalization and its use in the union scheme tool.

A farmer-source API returns a union by its dairy brand or a spelling variant
(e.g. "sarhad" for Kutch's Sarhad Dairy). The scheme tool must resolve those to
the canonical union so scheme lookup works. The AI-call ban list is keyed on
those same canonical names.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

from app.models.union import (
    AI_CALL_BANNED_UNIONS,
    UNION_BANNED_MESSAGE,
    UNION_BANNED_MESSAGE_GU,
    UNION_BANNED_MESSAGE_HI,
    UnionName,
    any_union_banned_from_ai_calls,
    canonical_union_name,
    is_ai_call_banned_union,
    resolve_supported_unions,
    union_banned_message,
)
import agents.tools.union_schemes as us


# ── canonical_union_name ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("sarhad", "kutch"),
    ("Sarhad", "kutch"),
    ("  KACHCHH  ", "kutch"),
    ("kutchh", "kutch"),
    ("kutch", "kutch"),
    ("banaskantha", "banas"),
    ("banas", "banas"),
    ("dudhsagar", "mehsana"),
    ("mehsana", "mehsana"),
    ("sursagar", "surendranagar"),
    ("Sursagar", "surendranagar"),
    ("sumul", "sumul"),
    ("kaira", "kaira"),   # no alias -> unchanged
    ("", ""),
    (None, ""),
])
def test_canonical_union_name(raw, expected):
    assert canonical_union_name(raw) == expected


def test_alias_targets_are_valid_unions():
    from app.models.union import UNION_NAME_ALIASES
    valid = {u.value for u in UnionName}
    for canonical in UNION_NAME_ALIASES.values():
        assert canonical in valid


def test_resolve_supported_unions_canonicalizes_and_deduplicates():
    supported = {UnionName.BANAS.value, UnionName.KUTCH.value}
    resolved = resolve_supported_unions(
        ["banaskantha", "kutch", "sarhad", "banas", "dudhsagar"],
        supported,
    )
    assert resolved == [UnionName.BANAS.value, UnionName.KUTCH.value]


# ── AI-call union ban list ────────────────────────────────────────────────────

def test_ai_call_banned_unions_contains_only_kutch():
    assert AI_CALL_BANNED_UNIONS == frozenset({UnionName.KUTCH.value})


@pytest.mark.parametrize("lang,expected", [
    (None, UNION_BANNED_MESSAGE),
    ("en", UNION_BANNED_MESSAGE),
    ("english", UNION_BANNED_MESSAGE),
    ("gu", UNION_BANNED_MESSAGE_GU),
    ("gujarati", UNION_BANNED_MESSAGE_GU),
    ("hi", UNION_BANNED_MESSAGE_HI),
    ("hindi", UNION_BANNED_MESSAGE_HI),
    ("unknown", UNION_BANNED_MESSAGE),
])
def test_union_banned_message_by_lang(lang, expected):
    assert union_banned_message(lang) == expected


@pytest.mark.parametrize("raw", [
    "kutch",
    "Kutch",
    "sarhad",
    "Sarhad",
    "  KACHCHH  ",
    "kutchh",
])
def test_kutch_aliases_are_banned_from_ai_calls(raw):
    assert is_ai_call_banned_union(raw) is True


@pytest.mark.parametrize("raw", [
    "banas",
    "banaskantha",
    "kaira",
    "mehsana",
    "dudhsagar",
    "",
    None,
])
def test_non_kutch_unions_are_not_banned_from_ai_calls(raw):
    assert is_ai_call_banned_union(raw) is False


@pytest.mark.parametrize("names,expected", [
    (["sarhad"], True),
    (["kutch"], True),
    (["kaira", "sarhad"], True),
    (["banas", "kaira"], False),
    ([], False),
    (None, False),
])
def test_any_union_banned_from_ai_calls(names, expected):
    assert any_union_banned_from_ai_calls(names) is expected


# ── tool resolves aliased unions to data ──────────────────────────────────────

def _ctx(unions):
    return SimpleNamespace(deps=SimpleNamespace(farmer_unions=unions))


def test_tool_resolves_sarhad_to_kutch(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", False)

    async def fake_records(union_name):
        assert union_name == "kutch"  # canonicalized before lookup
        return [{"scheme_title": "Group Personal Accident Insurance Scheme (GPAIS)"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sarhad"]), None))
    assert "GPAIS" in out
    assert "could not be determined" not in out


def test_tool_unsupported_union_still_fails(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    out = asyncio.run(us.get_union_scheme_data(_ctx(["dudhsagar"]), None))
    # dudhsagar canonicalizes to mehsana, which has no scheme source -> unsupported
    assert "could not be determined" in out


def test_network_failure_degrades_like_the_direct_path(monkeypatch):
    """A raising seeker must not fail the tool.

    The direct Redis path answers "temporarily unavailable"; before this the
    `enable_network` branch returned *before* that error handling, so a seeker
    timeout / HTTP error propagated out of the tool instead.
    """
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", True)

    import agents.tools.beckn_network as bn

    async def boom(scheme_name, union=None):
        raise RuntimeError("seeker connection reset")

    monkeypatch.setattr(bn, "network_union_schemes", boom)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["banas"]), None))
    assert "temporarily unavailable" in out


def test_network_success_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", True)

    import agents.tools.beckn_network as bn

    async def ok(scheme_name, union=None):
        assert union == "banas"
        return "Banas Network Scheme"

    monkeypatch.setattr(bn, "network_union_schemes", ok)

    assert asyncio.run(us.get_union_scheme_data(_ctx(["banas"]), None)) == "Banas Network Scheme"


def test_prepare_and_runtime_agree_for_banaskantha(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", False)
    sentinel = object()

    async def fake_records(union_name):
        assert union_name == UnionName.BANAS.value
        return [{"scheme_title": "Banas Test Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["banaskantha"]), sentinel))
    assert prepared is sentinel

    out = asyncio.run(us.get_union_scheme_data(_ctx(["banaskantha"]), None))
    assert "Banas Test Scheme" in out


def test_prepare_and_runtime_agree_for_sursagar(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", False)
    sentinel = object()

    async def fake_records(union_name):
        assert union_name == UnionName.SURENDRANAGAR.value
        return [{"scheme_title": "Sursagar Test Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["sursagar"]), sentinel))
    assert prepared is sentinel

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sursagar"]), None))
    assert "Sursagar Test Scheme" in out


def test_prepare_and_runtime_agree_for_sumul(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    monkeypatch.setattr(us.settings, "enable_network", False)
    sentinel = object()

    async def fake_records(union_name):
        assert union_name == UnionName.SUMUL.value
        return [{"scheme_title": "Sumul Test Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["sumul"]), sentinel))
    assert prepared is sentinel

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sumul"]), None))
    assert "Sumul Test Scheme" in out
