"""A farmer who asks in મણ reads મણ back (AMUL-124).

Pretranslation turns મણ into "maund", the agent answers "per maund", and the
reply leg must bring back the farmer's own word, not a transliteration
("માઉન્ડ") or the English word left in Latin letters.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.tools.terms import get_mini_glossary_for_text
from app.services.translation import _post_normalize_gu_translation, translation_channel


@pytest.mark.parametrize("text", [
    "Groundnut (Bold, Local): ₹1,450 per maund (20 kg)",
    "1 quintal is 5 maunds.",
])
def test_the_reply_leg_is_told_maund_is_man(text):
    assert "maund -> મણ" in get_mini_glossary_for_text(text, threshold=0.90, max_terms=40)


def test_voice_reply_leg_is_told_maund_is_man():
    from agents.voice.tools.terms import EN_INDEX as VOICE_EN_INDEX

    glossary = get_mini_glossary_for_text(
        "Groundnut is 1450 rupees per maund at Junagadh market yard.",
        threshold=0.90, max_terms=40, gu_en_index=VOICE_EN_INDEX,
    )
    assert "maund -> મણ" in glossary


@pytest.mark.parametrize("lang,word", [("hi", "मन"), ("mr", "मण"), ("pa", "ਮਣ"), ("bn", "মণ")])
def test_other_languages_get_their_own_maund(lang, word):
    assert f"maund -> {word}" in get_mini_glossary_for_text(
        "₹1,450 per maund", threshold=0.90, max_terms=40, target_lang=lang
    )


@pytest.mark.parametrize("channel", ["chat", "voice"])
@pytest.mark.parametrize("raw", [
    "મગફળી: ₹૧,૪૫૦ પ્રતિ maund",
    "મગફળી: ₹૧,૪૫૦ પ્રતિ Maunds",
    "મગફળી: ₹૧,૪૫૦ પ્રતિ માઉન્ડ",
    "મગફળી: ₹૧,૪૫૦ પ્રતિ મૌન્ડ",
])
def test_a_maund_that_slips_through_becomes_man(channel, raw):
    with translation_channel(channel):
        out = _post_normalize_gu_translation(raw, target_lang="gu", strip_outer=True)
    assert "પ્રતિ મણ" in out
    assert "maund" not in out.lower() and "માઉન્ડ" not in out and "મૌન્ડ" not in out


@pytest.mark.parametrize("channel", ["chat", "voice"])
# No digits here: voice speaks numbers as words, which is not what this checks.
@pytest.mark.parametrize("text", ["મગફળીનો ભાવ પ્રતિ મણ છે.", "હમણાં ભાવ શું છે?", "જીવાતના આક્રમણનો ઉપાય"])
def test_man_and_words_containing_it_are_left_alone(channel, text):
    with translation_channel(channel):
        assert _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True) == text
