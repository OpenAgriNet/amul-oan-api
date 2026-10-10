"""Chunk-boundary parity for Gujarati terminology normalization."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import (
    GU_TERM_POLICY,
    _StreamingGujaratiTermNormalizer,
    _apply_gu_term_replacements,
    translation_channel,
)


def _stream_terms(chunks: list[str], *, channel: str = "chat", lang: str = "gu") -> str:
    with translation_channel(channel):
        normalizer = _StreamingGujaratiTermNormalizer(lang)
    return "".join([*(normalizer.feed(chunk) for chunk in chunks), normalizer.flush()])


def _unary_terms(text: str, *, channel: str = "chat") -> str:
    with translation_channel(channel):
        return _apply_gu_term_replacements(text)


def test_every_policy_term_matches_unary_at_every_chunk_split() -> None:
    forbidden = GU_TERM_POLICY["forbidden"]
    assert len(forbidden) == 85

    for source, replacement in forbidden.items():
        expected = _unary_terms(source)
        assert expected == replacement
        for split in range(len(source) + 1):
            assert _stream_terms([source[:split], source[split:]]) == expected, (
                source,
                split,
            )


def test_longer_policy_terms_win_over_prefix_rules() -> None:
    forbidden = GU_TERM_POLICY["forbidden"]
    nested = [
        (longer, shorter)
        for longer in forbidden
        for shorter in forbidden
        if longer != shorter and longer.startswith(shorter)
    ]
    assert nested, "policy fixture must retain at least one prefix-overlap case"

    for longer, _ in nested:
        assert _stream_terms([longer]) == forbidden[longer]


@pytest.mark.parametrize("source", ["paho", "PAHO", "ગર્ભવતી"])
def test_shared_fixed_terms_match_at_every_chunk_split(source: str) -> None:
    expected = _unary_terms(source)
    for split in range(len(source) + 1):
        assert _stream_terms([source[:split], source[split:]]) == expected


def test_ascii_terms_require_whole_word_boundaries() -> None:
    assert _stream_terms(["xpaho", "y"]) == "xpahoy"
    assert _stream_terms(["in", "organic"]) == "inorganic"
    assert _stream_terms(["paho", " organic"]) == "બાવલું જૈવિક"


@pytest.mark.parametrize("source", ["organic", "ORGANIC", "ઓર્ગેનિક"])
def test_chat_organic_terms_are_stream_safe(source: str) -> None:
    expected = _unary_terms(source, channel="chat")
    assert expected == "જૈવિક"
    for split in range(len(source) + 1):
        assert _stream_terms(
            [source[:split], source[split:]], channel="chat"
        ) == expected


def test_voice_does_not_apply_chat_organic_rules() -> None:
    assert _stream_terms(["ઓર્ગે", "નિક"], channel="voice") == "ઓર્ગેનિક"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("બૈડા પર", "પીઠ પર"),
        ("બરડામાં", "પીઠમાં"),
        ("બૈડું ની બાજુ", "પીઠની બાજુ"),
        ("બરડુ ના ભાગપર", "પીઠના ભાગપર"),
    ],
)
def test_voice_body_context_matches_at_every_chunk_split(
    source: str, expected: str
) -> None:
    assert _unary_terms(source, channel="voice") == expected
    for split in range(len(source) + 1):
        assert _stream_terms(
            [source[:split], source[split:]], channel="voice"
        ) == expected


def test_generic_body_context_still_uses_policy_default() -> None:
    assert _stream_terms(["બૈ", "ડા"], channel="voice") == "શરીર"
    assert _stream_terms(["બર", "ડું"], channel="chat") == "શરીર"


def test_flush_releases_an_incomplete_term_prefix() -> None:
    normalizer = _StreamingGujaratiTermNormalizer("gu")
    assert normalizer.feed("વોડ") == ""
    assert normalizer.flush() == "વોડ"


def test_non_gujarati_stream_is_unbuffered() -> None:
    normalizer = _StreamingGujaratiTermNormalizer("english")
    assert normalizer.feed("ગર્ભ") == "ગર્ભ"
    assert normalizer.feed("વતી") == "વતી"
    assert normalizer.flush() == ""
