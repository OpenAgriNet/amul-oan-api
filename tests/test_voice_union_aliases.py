"""Tests for canonical union-name normalization and its use in the voice union scheme tool.

A farmer-source API returns a union by its dairy brand or a spelling variant
(e.g. "sarhad" for Kutch's Sarhad Dairy). The scheme tool must resolve those to
the canonical union so scheme lookup works. The AI-call ban list is keyed on
those same canonical names.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import asyncio
from types import SimpleNamespace

import pytest

from app.voice.models.union import (
    AI_CALL_BANNED_UNIONS,
    UNION_BANNED_MESSAGE,
    UNION_BANNED_MESSAGES,
    UNION_NAME_ALIASES,
    UnionName,
    any_union_banned_from_ai_calls,
    canonical_union_name,
    is_ai_call_banned_union,
    resolve_supported_unions,
    union_banned_message_for_lang,
)
import agents.voice.tools.union_schemes as us


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
    ("dudhdhara", "bharuch"),
    ("Dudh Dhara", "bharuch"),
    ("  Dudh Dhara  ", "bharuch"),
    ("  DUDHDHARA  ", "bharuch"),
    ("bharuch", "bharuch"),
    ("vasudhara", "vasudhara"),
    ("sumul", "sumul"),
    ("kaira", "kaira"),
    ("", ""),
    (None, ""),
])
def test_canonical_union_name(raw, expected):
    assert canonical_union_name(raw) == expected


def test_alias_targets_are_valid_unions():
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


def test_union_banned_message_is_the_agreed_farmer_facing_string():
    assert UNION_BANNED_MESSAGE == "Kindly contact your Milk Society to book the service."
    assert UNION_BANNED_MESSAGES["en"] == UNION_BANNED_MESSAGE
    assert UNION_BANNED_MESSAGES["gu"] == "કૃપા કરીને આપની દૂધ મંડળીનો સંપર્ક કરશો."
    assert UNION_BANNED_MESSAGES["hi"] == "कृपया सेवा बुक करने के लिए अपनी दूध मंडली से संपर्क करें।"


@pytest.mark.parametrize("lang,expected_key", [
    ("en", "en"),
    ("english", "en"),
    ("gu", "gu"),
    ("gujarati", "gu"),
    ("hi", "hi"),
    ("hindi", "hi"),
    ("mr", "en"),
    (None, "en"),
    ("", "en"),
])
def test_union_banned_message_for_lang(lang, expected_key):
    assert union_banned_message_for_lang(lang) == UNION_BANNED_MESSAGES[expected_key]


def _ctx(unions):
    return SimpleNamespace(deps=SimpleNamespace(farmer_unions=unions))


def test_tool_resolves_sarhad_to_kutch(monkeypatch):
    async def fake_records(union_name):
        assert union_name == "kutch"  # canonicalized before lookup
        return [{"scheme_title": "Group Personal Accident Insurance Scheme (GPAIS)"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sarhad"]), None))
    assert "GPAIS" in out
    assert "could not be determined" not in out


def test_tool_unsupported_union_still_fails():
    out = asyncio.run(us.get_union_scheme_data(_ctx(["dudhsagar"]), None))
    assert "could not be determined" in out


def test_prepare_and_runtime_agree_for_banaskantha(monkeypatch):
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
    sentinel = object()

    async def fake_records(union_name):
        assert union_name == UnionName.SUMUL.value
        return [{"scheme_title": "Sumul Test Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["sumul"]), sentinel))
    assert prepared is sentinel

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sumul"]), None))
    assert "Sumul Test Scheme" in out


def test_prepare_and_runtime_agree_for_sabar(monkeypatch):
    sentinel = object()

    async def fake_records(union_name):
        assert union_name == UnionName.SABAR.value
        return [{"scheme_title": "Sabar Test Scheme"}]

    monkeypatch.setattr(us, "get_cached_scheme_records_for_union", fake_records)

    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["sabar"]), sentinel))
    assert prepared is sentinel

    out = asyncio.run(us.get_union_scheme_data(_ctx(["sabar"]), None))
    assert "Sabar Test Scheme" in out


def test_scheme_cache_keys_match_chat_ingestion_sources():
    """Voice reads the same Redis keys that amul-oan-api scheme ingestion writes."""
    from app.voice.scheme_ingestion import (
        SUPPORTED_SCHEME_UNIONS,
        SUPPORTED_UNION_SOURCE_KEYS,
        get_source_keys_for_union,
    )

    assert get_source_keys_for_union(UnionName.BANAS.value) == (
        "banasdairy.coop/home/inputactivities#milkproducers",
    )
    assert get_source_keys_for_union(UnionName.KUTCH.value) == (
        "sarhaddairy.coop/for-our-milk-producers",
    )
    assert get_source_keys_for_union(UnionName.SUMUL.value) == ("sumul.com/farmer-section",)
    assert get_source_keys_for_union("sursagar") == ("sursagardairy.com/farmer/milkproducers",)
    assert get_source_keys_for_union(UnionName.SURENDRANAGAR.value) == (
        "sursagardairy.com/farmer/milkproducers",
    )
    assert get_source_keys_for_union(UnionName.BHARUCH.value) == (
        "dudhdharadairy.in/for_our_milk_producers",
    )
    assert get_source_keys_for_union("dudhdhara") == (
        "dudhdharadairy.in/for_our_milk_producers",
    )
    assert get_source_keys_for_union("dudh dhara") == (
        "dudhdharadairy.in/for_our_milk_producers",
    )
    assert get_source_keys_for_union("vasudhara") == ()
    assert get_source_keys_for_union(UnionName.SABAR.value) == (
        "sabardairy.org/for-our-milk-producers",
    )
    assert get_source_keys_for_union(UnionName.GANDHINAGAR.value) == (
        "madhurdairy.org/forourmilkproducers",
    )
    assert get_source_keys_for_union("madhur dairy") == (
        "madhurdairy.org/forourmilkproducers",
    )
    assert us.SUPPORTED_SCHEME_UNIONS == frozenset(SUPPORTED_UNION_SOURCE_KEYS)
    assert SUPPORTED_SCHEME_UNIONS == frozenset(
        {
            UnionName.BANAS.value,
            UnionName.KUTCH.value,
            UnionName.SUMUL.value,
            UnionName.SURENDRANAGAR.value,
            UnionName.BHARUCH.value,
            UnionName.SABAR.value,
            UnionName.GANDHINAGAR.value,
        }
    )


def test_voice_reads_every_union_chat_ingests():
    """A union chat's scheduler writes but voice does not read is a union whose
    callers hear no schemes (Gandhinagar, until 4f2b7f7 was ported)."""
    from app.services import scheme_ingestion as chat_ingestion
    from app.voice.scheme_ingestion import SUPPORTED_UNION_SOURCE_KEYS

    written = {
        union: tuple(source.cache_key for source in sources)
        for union, sources in chat_ingestion.SUPPORTED_UNION_SOURCE_MAP.items()
    }
    assert SUPPORTED_UNION_SOURCE_KEYS == written


class _Redis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)


def test_a_madhur_farmer_hears_the_schemes_chat_ingested(monkeypatch):
    from app.services import scheme_ingestion as chat_ingestion
    from app.voice import farmer as voice_farmer
    from app.voice import scheme_ingestion as voice_ingestion

    redis = _Redis()
    record = {
        "union_name": UnionName.GANDHINAGAR.value,
        "scheme_title": "Madhur Test Scheme",
        "scheme_url": "https://example.org/madhur-test-scheme.pdf",
    }
    asyncio.run(chat_ingestion.cache_source_records(
        chat_ingestion.MADHUR_SOURCE.cache_key, [record], redis_client=redis,
    ))
    monkeypatch.setattr(voice_ingestion, "_redis_client", redis)
    sentinel = object()

    summary = asyncio.run(voice_farmer._build_union_scheme_summary(["madhur dairy"]))
    prepared = asyncio.run(us.prepare_get_union_scheme_data(_ctx(["madhur dairy"]), sentinel))
    out = asyncio.run(us.get_union_scheme_data(_ctx(["madhur dairy"]), None))

    assert "Madhur Test Scheme: https://example.org/madhur-test-scheme.pdf" in summary
    assert prepared is sentinel
    assert "Madhur Test Scheme" in out
