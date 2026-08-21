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


# ── The long 'aa' is not always wrong ──────────────────────────────────────────
# રામેશ્વર (Rameshwar) = રામ + ઈશ્વર and is CORRECTLY long-aa. "RAMESH" is a
# substring of "Rameshwar", so the trigger arms on it; the output pattern is what
# has to know better. These pin the boundary.

@pytest.mark.parametrize("gu", [
    "રામેશ્વર મંદિર ગામની નજીક છે.",            # conjunct શ્વ
    "રામેશ્વરમ ગયા વર્ષે ગયા હતા.",              # conjunct, longer word
    "ટેકનિશિયન રામેશવર પટેલ કાલે આવશે.",        # same name written without the conjunct
])
def test_legitimate_long_aa_is_not_shortened(gu):
    """Regression: this exact corruption was live in prod (રામેશ્વર -> રમેશ્વર)."""
    triggers = _protected_output_triggers("Rameshwar temple visit", "gu")
    assert _apply_protected_output(gu, triggers) == gu


def test_mixed_sentence_fixes_the_name_and_spares_the_place():
    """Both in one sentence: the farmer's name is corrected, Rameshwaram is not."""
    src = "Rameshbhai went to Rameshwaram last year."
    raw = "રામેશભાઈ ગયા વર્ષે રામેશ્વરમ ગયા હતા."
    out = _apply_protected_output(raw, _protected_output_triggers(src, "gu"))
    assert out == "રમેશભાઈ ગયા વર્ષે રામેશ્વરમ ગયા હતા."


def test_name_followed_by_an_unrelated_va_word_still_fixed():
    """The lookahead is one character wide — a separate word starting with વ must
    not stop the pin firing on the name itself."""
    raw = "રામેશ વગેરે લોકો આવ્યા."
    out = _apply_protected_output(raw, _protected_output_triggers("Ramesh and others", "gu"))
    assert out == "રમેશ વગેરે લોકો આવ્યા."


# ── The KDCC bank name (AMUL-51) ──────────────────────────────────────────────
# The pin must repair the name whether or not the model keeps the "- Nadiad"
# branch, and must not disturb the લિમિટેડ / case-ending the model appended.

BANK_EN = (
    "You are eligible for a micro loan of Rs 5,000 from Kheda District Central "
    "Co-Operative Bank Limited - Nadiad."
)
# Verbatim from the AMUL-51 report: the model dropped "Nadiad", so the old
# pattern (which required a trailing નડિયાદ) never fired and this reached prod.
BANK_RAW_NO_BRANCH = "ખેડા જિલ્લા કેન્દ્રીય સહકારી બેંક લિમિટેડમાંથી લોન મંજૂર થઈ છે."
BANK_FIXED_NO_BRANCH = "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક લિમિટેડમાંથી લોન મંજૂર થઈ છે."

BANK_RAW_WITH_BRANCH = "ખેડા જિલ્લા કેન્દ્રીય સહકારી બેંક લિમિટેડ - નડિયાદમાંથી લોન મંજૂર થઈ છે."
BANK_FIXED_WITH_BRANCH = "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક લિમિટેડ - નડિયાદમાંથી લોન મંજૂર થઈ છે."


def test_bank_name_pinned_when_model_drops_the_branch():
    """The AMUL-51 regression. The case ending (માંથી) attaches to લિમિટેડ with no
    space, so the pattern has to stop at બેંક and leave the tail alone."""
    out = _apply_protected_output(BANK_RAW_NO_BRANCH, _protected_output_triggers(BANK_EN, "gu"))
    assert out == BANK_FIXED_NO_BRANCH
    assert "જિલ્લા કેન્દ્રીય સહકારી" not in out


def test_bank_name_pinned_when_model_keeps_the_branch():
    """The case the old pattern already handled — must not regress."""
    out = _apply_protected_output(BANK_RAW_WITH_BRANCH, _protected_output_triggers(BANK_EN, "gu"))
    assert out == BANK_FIXED_WITH_BRANCH


@pytest.mark.parametrize("raw,fixed", [
    # 'Central' dropped by the model
    ("ખેડા જિલ્લા સહકારી બેંક લિમિટેડ", "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક લિમિટેડ"),
    # 'મધ્યસ્થ' instead of 'કેન્દ્રીય'
    ("ખેડા જિલ્લા મધ્યસ્થ સહકારી બેંક", "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક"),
    # half-transliterated, and the ક્ટ conjunct written open (ડિસ્ટ્રિકટ)
    ("ખેડા ડિસ્ટ્રિકટ કેન્દ્રીય સહકારી બેંક", "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક"),
    # 'કો ઓપરેટિવ' written with a space instead of the hyphen
    ("ખેડા જિલ્લા સેન્ટ્રલ કો ઓપરેટિવ બેંક", "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક"),
])
def test_bank_name_variants_are_normalised(raw, fixed):
    assert _apply_protected_output(raw, _protected_output_triggers(BANK_EN, "gu")) == fixed


def test_bank_pin_is_idempotent():
    """Already-correct text must round-trip: the pin re-matches its own output, so a
    self-replace is the only safe shape for a rule applied to a streaming buffer."""
    triggers = _protected_output_triggers(BANK_EN, "gu")
    once = _apply_protected_output(BANK_RAW_NO_BRANCH, triggers)
    assert _apply_protected_output(once, triggers) == once


@pytest.mark.parametrize("src", [
    # The agent composes its own English and shortens the name it was given. A
    # substring gate on the full "…Limited - Nadiad" missed every one of these.
    "You qualify for a micro loan from Kheda District Central Co-Operative Bank.",
    "Visit your nearest Kheda District Central Co-operative Bank branch.",
    "a loan from the kheda district central co-operative bank limited",
])
def test_bank_gate_arms_on_shortened_english_names(src):
    assert _protected_output_triggers(src, "gu")


@pytest.mark.parametrize("src", [
    "Rainfall in Kheda district is expected tomorrow.",
    "Kheda district mandi prices are higher this week.",
])
def test_bank_gate_does_not_arm_on_plain_district_mentions(src):
    """Arming costs an 80-char stream holdback on every chunk, so ordinary Kheda
    weather/market answers must not trip it."""
    assert _protected_output_triggers(src, "gu") == []


def test_bank_pin_does_not_fire_on_another_district_bank():
    """The pattern is anchored on ખેડા — a different district's bank is left alone."""
    other = "મહેસાણા જિલ્લા કેન્દ્રીય સહકારી બેંક લિમિટેડમાંથી"
    assert _apply_protected_output(other, _protected_output_triggers(BANK_EN, "gu")) == other


@pytest.mark.asyncio
async def test_bank_pin_survives_a_chunk_boundary_mid_name():
    chunks = ["ખેડા જિલ્લા કેન્દ્રીય સહ", "કારી બેંક લિમિટેડમાંથી લોન મંજૂર થઈ છે."]

    async def gen():
        for c in chunks:
            yield c

    triggers = _protected_output_triggers(BANK_EN, "gu")
    out = "".join([c async for c in _buffered_protected_stream(gen(), triggers)])
    assert out == BANK_FIXED_NO_BRANCH


@pytest.mark.asyncio
async def test_bank_pin_does_not_duplicate_words_arriving_after_the_match():
    """The buffer is rewritten on every chunk, so a rule that APPENDS words the model
    has not emitted yet duplicates them when they arrive. Pinning only through બેંક
    keeps લિમિટેડ / નડિયાદ the model's to supply — this is the guard on that."""
    chunks = ["ખેડા જિલ્લા કેન્દ્રીય સહકારી બેંક", " લિમિટેડ", " - નડિયાદમાંથી લોન મંજૂર થઈ છે."]

    async def gen():
        for c in chunks:
            yield c

    triggers = _protected_output_triggers(BANK_EN, "gu")
    out = "".join([c async for c in _buffered_protected_stream(gen(), triggers)])
    assert out == BANK_FIXED_WITH_BRANCH
    assert out.count("લિમિટેડ") == 1
    assert out.count("બેંક") == 1
