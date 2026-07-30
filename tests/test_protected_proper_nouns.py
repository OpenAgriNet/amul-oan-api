"""Pinned Gujarati renderings for protected proper nouns.

Covers the Ramesh pin. TranslateGemma lengthens the first vowel — રામેશ
("Raamesh") for રમેશ — and the two translation paths disagree on the rest of the
name: the unary path keeps રામેશભાઈ while the streaming path drops ભાઈ and emits
a bare રામેશ. The pin therefore normalises the VOWEL rather than the whole name.
All raw strings below are verbatim live output from the 27b-base LB (the streaming
ones captured inside a prod pod), so these tests fail if the pin's regex stops
matching what the model actually emits on either path.
"""

import os

# Import-time guard used across this tests/ tree: translation.py builds clients at
# module import, which needs a key present even though these tests never call out.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import (
    _apply_protected_output,
    _buffered_protected_stream,
    _protected_output_triggers,
)

# Verbatim TranslateGemma output for an English sentence naming the farmer.
TG_RAW = "નમસ્તે રામેશભાઈ કાનુભાઈ પરમાર, છેલ્લાં 7 દિવસમાં તમારું દૂધ 17.83 લિટર છે."
TG_PINNED = "નમસ્તે રમેશભાઈ કાનુભાઈ પરમાર, છેલ્લાં 7 દિવસમાં તમારું દૂધ 17.83 લિટર છે."

# Verbatim STREAMING-path output from a prod pod: ભાઈ dropped off the first name.
# A ભાઈ-anchored pattern silently missed this, which is the whole point of the pin
# being on the vowel.
TG_STREAM_RAW = "નમસ્તે રામેશ કાનુભાઈ પરમાર છેલ્લાં સાત દિવસમાં આપનું દૂધ રહ્યું છે."
TG_STREAM_PINNED = "નમસ્તે રમેશ કાનુભાઈ પરમાર છેલ્લાં સાત દિવસમાં આપનું દૂધ રહ્યું છે."

EN_SRC = "Hello RAMESHBHAI KANUBHAI PARMAR2, your milk for the last 7 days is 17.83 litres."


def test_long_vowel_rendering_is_pinned():
    out = _apply_protected_output(TG_RAW, _protected_output_triggers(EN_SRC, "gu"))
    assert out == TG_PINNED
    assert "રામેશભાઈ" not in out


def test_already_correct_rendering_is_left_alone():
    """A tier that already spells it correctly (gemma-4 / managed overflow) must
    round-trip unchanged — the pin is a self-replace, not a double-edit."""
    good = "રમેશભાઈ પાસે ૩ ભેંસ છે."
    assert _apply_protected_output(good, _protected_output_triggers(EN_SRC, "gu")) == good


@pytest.mark.parametrize("src", [
    "Hello RAMESHBHAI, your milk is ready.",
    "Hello Rameshbhai, your milk is ready.",
    "hello rameshbhai, your milk is ready.",
])
def test_trigger_gate_is_case_insensitive(src):
    """The name arrives in whatever case the record or the agent's sentence used."""
    assert _protected_output_triggers(src, "gu")


def test_no_trigger_when_name_absent():
    """Ordinary traffic must not be rewritten (or, on the streaming path, buffered)."""
    assert _protected_output_triggers("Your milk fat is 8.91 percent.", "gu") == []


def test_no_trigger_for_non_gujarati_target():
    assert _protected_output_triggers(EN_SRC, "hi") == []


@pytest.mark.asyncio
async def test_pin_survives_a_chunk_boundary_mid_name():
    """Streaming splits tokens arbitrarily; the holdback buffer must let a name
    broken across two chunks still match before it is flushed."""
    chunks = ["નમસ્તે રા", "મેશભાઈ કાનુભાઈ પરમાર, તમારું દૂધ તૈયાર છે."]

    async def gen():
        for c in chunks:
            yield c

    triggers = _protected_output_triggers(EN_SRC, "gu")
    out = "".join([c async for c in _buffered_protected_stream(gen(), triggers)])
    assert "રમેશભાઈ" in out
    assert "રામેશભાઈ" not in out


def test_streaming_rendering_without_bhai_is_pinned():
    """The streaming tier drops ભાઈ off the first name; the pin must still fire.
    This is the regression that a ભાઈ-anchored pattern let through to prod."""
    out = _apply_protected_output(TG_STREAM_RAW, _protected_output_triggers(EN_SRC, "gu"))
    assert out == TG_STREAM_PINNED
    assert "રામેશ" not in out


def test_technician_short_form_is_untouched():
    """The booking matcher's technician fixture is already correctly spelled, so the
    pin must not perturb it — it fires only on the long-vowel misspelling."""
    tech = "રમેશ પટેલ સાથે બુક કરો"
    assert _apply_protected_output(tech, _protected_output_triggers(EN_SRC, "gu")) == tech
