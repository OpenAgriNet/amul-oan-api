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
    translation_channel,
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
    "Co-Operative Bank Limited."
)
# The branch suffix was dropped from the name, but the model can still emit નડિયાદ
# from a farmer's own question or a KB passage, so the pin must keep coping with it.
BANK_EN_WITH_BRANCH = BANK_EN[:-1] + " - Nadiad."
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
    out = _apply_protected_output(BANK_RAW_WITH_BRANCH,
                                  _protected_output_triggers(BANK_EN_WITH_BRANCH, "gu"))
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


# ── The assistant's own name (AMUL-112) and the brand ─────────────────────────
# Raw strings are verbatim prod stream_translation output for the agent's English
# self-introduction, 2026-10-19.

@pytest.mark.parametrize("lang,raw,fixed", [
    ("gu", "હું સર્લાબેન છું,", "હું સરલાબેન છું,"),
    ("hi", "मैं सारलाबेन हूँ,", "मैं सरलाबेन हूँ,"),
    ("bn", "আমি সারলাবেন, আমিউল-এর", "আমি সরলাবেন, আমুল-এর"),
    # Already right: self-replace.
    ("gu", "હું સરલાબેન છું.", "હું સરલાબેન છું."),
    ("mr", "मी सरलाबेन आहे.", "मी सरलाबेन आहे."),
    ("pa", "ਮੈਂ ਸਰਲਾਬੇਨ ਹਾਂ।", "ਮੈਂ ਸਰਲਾਬੇਨ ਹਾਂ।"),
])
def test_sarlaben_is_spelt_the_way_the_product_spells_it(lang, raw, fixed):
    src = "I am Sarlaben, Amul's AI digital assistant."
    assert _apply_protected_output(raw, _protected_output_triggers(src, lang)) == fixed


def test_gujarati_brand_pinned_in_chat_but_voice_is_left_alone():
    src = "Give **Amul Power Dan**, which is balanced feed."
    raw = "અમુલ પાવર ડેન** આપો"
    assert _apply_protected_output(raw, _protected_output_triggers(src, "gu")) == "અમૂલ પાવર ડેન** આપો"
    with translation_channel("voice"):
        assert _protected_output_triggers(src, "gu") == []


def test_bengali_words_that_start_like_the_brand_are_kept():
    src = "Amul gives priceless support."
    text = "অমূল্য সহায়তা, অমূলক ভয়"
    assert _apply_protected_output(text, _protected_output_triggers(src, "bn")) == text


# ── A silage supplier's name (AMUL-111) ───────────────────────────────────────
# AMUL-111 reported the supplier Natural Silage said as કુદરતી સાઇલેજ, and a 17 Oct
# 2026 voice call said કુદરતી સિલેજ; the product says it નેચરલ સાઇલેજ. The other raw
# strings are the spellings voice's glossary row (Silage -> સાયલેજ) and the
# translators can give.

SILAGE_EN = "You can buy silage from Natural Silage."


@pytest.mark.parametrize("raw,fixed", [
    # The ticket: "natural" translated instead of said as the name.
    ("કુદરતી સાઇલેજ પાસેથી સાયલેજ લઈ શકો.", "નેચરલ સાઇલેજ પાસેથી સાયલેજ લઈ શકો."),
    # The 17 Oct call: only the supplier named Natural changes, not the next one.
    ("કુદરતી સિલેજ ધનલક્ષ્મી એગ્રો સિલેજ", "નેચરલ સાઇલેજ ધનલક્ષ્મી એગ્રો સિલેજ"),
    # Voice's glossary row writes silage સાયલેજ.
    ("કુદરતી સાયલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
    ("પ્રાકૃતિક સાયલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
    ("નેચરલ સાયલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
    ("નેચરલ સાઈલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
    ("નેચ્યુરલ સાયલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
    # A case ending glued to the name is the model's and stays.
    ("કુદરતી સાયલેજમાંથી મંગાવો.", "નેચરલ સાઇલેજમાંથી મંગાવો."),
    ("કુદરતી સાઇલેજનો નંબર", "નેચરલ સાઇલેજનો નંબર"),
    # Already right: self-replace.
    ("નેચરલ સાઇલેજ પાસેથી", "નેચરલ સાઇલેજ પાસેથી"),
], ids=["ticket", "call-17-oct", "glossary-silage", "prakrutik", "nechral-sayalej", "nechral-long-ii",
        "nechyural", "case-ending-manthi", "case-ending-no", "already-right"])
def test_natural_silage_is_said_the_way_the_product_says_it(raw, fixed):
    assert _apply_protected_output(raw, _protected_output_triggers(SILAGE_EN, "gu")) == fixed


@pytest.mark.parametrize("src", [
    SILAGE_EN,
    "For silage bales, call NATURAL SILAGE, Anand.",
    "natural silage supplies maize silage.",
])
def test_natural_silage_gate_arms_in_chat_and_voice_in_any_case(src):
    assert _protected_output_triggers(src, "gu")
    with translation_channel("voice"):
        assert _protected_output_triggers(src, "gu")


@pytest.mark.parametrize("src", [
    "Feed silage daily.",
    "Natural mating is better than artificial insemination.",
    "Silage is a natural feed.",
])
def test_natural_silage_gate_does_not_arm_without_the_name(src):
    assert _protected_output_triggers(src, "gu") == []


@pytest.mark.parametrize("text", [
    "કુદરતી ફળવણી કરતાં કૃત્રિમ બીજદાન સારું છે.",  # natural service, the glossary's
    "રોજ સાયલેજ આપો.",  # silage on its own
    "કુદરતી રીતે બનેલું સાયલેજ",  # the two words apart
    "અકુદરતી સાયલેજ",  # unnatural
    "- લીલો ચારો: તાજો અને કુદરતી\n- સાયલેજ: સૂકી ઋતુમાં સારું",  # two chat list items
    "આ ચારો તાજો અને કુદરતી\nસાયલેજ પણ આપી શકાય",  # the words on two lines
], ids=["natural-service", "silage-alone", "words-apart", "unnatural", "list-items", "two-lines"])
def test_other_natural_and_silage_words_are_left_alone(text):
    src = "Natural Silage sells silage. Natural mating is better than artificial insemination."
    assert _apply_protected_output(text, _protected_output_triggers(src, "gu")) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks,said", [
    # Split inside the name.
    (["કુદરતી સા", "યલેજમાંથી સાયલેજ મંગાવો."], "નેચરલ સાઇલેજમાંથી સાયલેજ મંગાવો."),
    # The case ending comes in the next chunk: kept, not doubled.
    (["કુદરતી સાયલેજ", "માંથી સાયલેજ મંગાવો."], "નેચરલ સાઇલેજમાંથી સાયલેજ મંગાવો."),
    # Past the holdback: the start of the answer is flushed while the name forms.
    (
        ["આ નંબર પર ફોન કરો. " * 5 + "કુદરતી", " સાયલેજમાંથી મંગાવો."],
        "આ નંબર પર ફોન કરો. " * 5 + "નેચરલ સાઇલેજમાંથી મંગાવો.",
    ),
], ids=["split-in-name", "ending-in-next-chunk", "past-the-holdback"])
async def test_natural_silage_pin_survives_a_chunk_boundary(chunks, said):
    async def gen():
        for c in chunks:
            yield c

    triggers = _protected_output_triggers(SILAGE_EN, "gu")
    out = "".join([c async for c in _buffered_protected_stream(gen(), triggers)])
    assert out == said
