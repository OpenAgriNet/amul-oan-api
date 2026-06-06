"""Tests for canonical union-name normalization and its use in the union scheme tool.

A farmer-source API returns a union by its dairy brand or a spelling variant
(e.g. "sarhad" for Kutch's Sarhad Dairy). The scheme tool must resolve those to
the canonical union so scheme lookup works.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

from app.models.union import UnionName, canonical_union_name
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
    ("panchamrut", "panchmahal"),
    ("PANCHAMRUT", "panchmahal"),
    ("panchmahal", "panchmahal"),
    ("dudhsagar", "mehsana"),
    ("mehsana", "mehsana"),
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


# ── tool resolves aliased unions to data ──────────────────────────────────────

def _ctx(unions):
    return SimpleNamespace(deps=SimpleNamespace(farmer_unions=unions))


def test_tool_resolves_sarhad_to_kutch(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)

    async def fake_records(union_name):
        assert union_name == "kutch"  # canonicalized before lookup
        return [{"scheme_title": "Group Personal Accident Insurance Scheme (GPAIS)"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sarhad"]), None))
    assert "GPAIS" in out
    assert "could not be determined" not in out


def test_tool_resolves_panchamrut_to_panchmahal(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)

    async def fake_records(union_name):
        assert union_name == "panchmahal"  # canonicalized before lookup
        return [{"scheme_title": "Mini Village Water Supply Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["panchamrut"]), None))
    assert "Mini Village Water Supply Scheme" in out
    assert "could not be determined" not in out


def test_tool_unsupported_union_still_fails(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    out = asyncio.run(us.get_union_scheme_data(_ctx(["dudhsagar"]), None))
    # dudhsagar canonicalizes to mehsana, which has no scheme source -> unsupported
    assert "could not be determined" in out


def test_prepare_exposes_tool_when_union_auth_disabled(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", False)
    tool_def = SimpleNamespace(name="get_union_scheme_data")
    out = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["dudhsagar"]), tool_def))
    assert out is tool_def


def test_prepare_hides_tool_for_unsupported_union_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    tool_def = SimpleNamespace(name="get_union_scheme_data")
    out = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["dudhsagar"]), tool_def))
    assert out is None


def test_prepare_exposes_tool_for_aliased_supported_union(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", True)
    tool_def = SimpleNamespace(name="get_union_scheme_data")
    out = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["sarhad"]), tool_def))
    assert out is tool_def


def test_tool_auth_disabled_queries_all_supported_unions(monkeypatch):
    monkeypatch.setattr(us.settings, "scheme_require_union_auth", False)
    queried: list[str] = []

    async def fake_records(union_name):
        queried.append(union_name)
        return [{"scheme_title": f"{union_name}-scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["dudhsagar"]), None))
    assert queried == sorted(us.SUPPORTED_SCHEME_UNIONS)
    assert "banas-scheme" in out
