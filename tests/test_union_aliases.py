from types import SimpleNamespace

import pytest

from agents.tools import union_schemes as schemes
from agents.tools.models.union import (
    AI_CALL_BANNED_UNIONS,
    UNION_BANNED_MESSAGE,
    UNION_BANNED_MESSAGE_BN,
    UNION_BANNED_MESSAGE_GU,
    UNION_BANNED_MESSAGE_PA,
    UNION_BANNED_MESSAGE_MR,
    UNION_BANNED_MESSAGE_HI,
    UNION_NAME_ALIASES,
    UnionName,
    any_union_banned_from_ai_calls,
    canonical_union_name,
    is_ai_call_banned_union,
    resolve_supported_unions,
    union_banned_message,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
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
        ("Dudhdhara", "bharuch"),
        ("  DUDH DHARA ", "bharuch"),
        ("bharuch", "bharuch"),
        ("vasudhara", "vasudhara"),
        ("sumul", "sumul"),
        ("kaira", "kaira"),
        ("", ""),
        (None, ""),
    ],
)
def test_union_aliases(raw, expected):
    assert canonical_union_name(raw) == expected


def test_alias_targets_are_valid_unions():
    valid = {union.value for union in UnionName}
    assert set(UNION_NAME_ALIASES.values()) <= valid


def test_supported_unions_are_canonicalized_and_deduplicated():
    assert resolve_supported_unions(
        ["banaskantha", "kutch", "sarhad", "banas", "dudhsagar"],
        {UnionName.BANAS.value, UnionName.KUTCH.value},
    ) == [UnionName.BANAS.value, UnionName.KUTCH.value]


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
    ("bn", UNION_BANNED_MESSAGE_BN),
    ("bengali", UNION_BANNED_MESSAGE_BN),
    ("pa", UNION_BANNED_MESSAGE_PA),
    ("punjabi", UNION_BANNED_MESSAGE_PA),
    ("mr", UNION_BANNED_MESSAGE_MR),
    ("marathi", UNION_BANNED_MESSAGE_MR),
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


@pytest.mark.asyncio
async def test_scheme_tool_resolves_alias_and_uses_beckn(monkeypatch):
    monkeypatch.setattr(schemes.settings, "scheme_require_union_auth", True)

    async def lookup(scheme_name, union=None):
        assert scheme_name == "insurance"
        assert union == UnionName.KUTCH.value
        return "Kutch Network Scheme"

    monkeypatch.setattr(schemes, "network_union_schemes", lookup)
    result = await schemes.get_union_scheme_data(_ctx(["sarhad"]), "insurance")
    assert result == "Kutch Network Scheme"


@pytest.mark.asyncio
async def test_scheme_network_failure_degrades(monkeypatch):
    monkeypatch.setattr(schemes.settings, "scheme_require_union_auth", True)

    async def lookup(*args, **kwargs):
        raise RuntimeError("seeker unavailable")

    monkeypatch.setattr(schemes, "network_union_schemes", lookup)
    result = await schemes.get_union_scheme_data(_ctx(["banas"]))
    assert "temporarily unavailable" in result


@pytest.mark.asyncio
async def test_scheme_tool_rejects_unsupported_union(monkeypatch):
    monkeypatch.setattr(schemes.settings, "scheme_require_union_auth", True)
    result = await schemes.get_union_scheme_data(_ctx(["dudhsagar"]))
    assert "could not be determined" in result


@pytest.mark.asyncio
async def test_prepare_matches_supported_union_set():
    sentinel = object()
    assert await schemes.prepare_get_union_scheme_data(_ctx(["banaskantha"]), sentinel) is sentinel
    assert await schemes.prepare_get_union_scheme_data(_ctx(["dudhsagar"]), sentinel) is None


def test_ingestion_sources_cover_every_supported_union():
    from app.services.scheme_ingestion import SUPPORTED_UNION_SOURCE_MAP

    assert schemes.SUPPORTED_SCHEME_UNIONS == set(SUPPORTED_UNION_SOURCE_MAP)
