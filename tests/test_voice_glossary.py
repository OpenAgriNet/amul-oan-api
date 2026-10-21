"""Voice's glossary: chat's shared rows with voice's own on top.

AMUL-90 (voice-oan-api fec6370, amul-dev only): voice reads abortion as
ગર્ભપાત where the shared glossary says તરવાઈ જવું, and has a Silage row. Voice
hears them in pretranslation, says them in output translation and finds them
in its term lookup. Chat keeps its own rows.
"""
import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.voice.tools.terms import search_terms
from app.services import translation
from app.voice.pretranslation import _get_glossary_hints_for_gu_query

_ANSWER = "The cow had an abortion. Feed silage daily."


def _instruction(channel, text=_ANSWER):
    with translation.translation_channel(channel):
        instruction, _ = translation._prepare_translation_inputs(text, "english", "gujarati", None)
    return instruction


def test_voice_hears_and_says_abortion_and_silage_as_amul_dev_does():
    hints = _get_glossary_hints_for_gu_query("ગાયને ગર્ભપાત થઈ ગયો, સાયલેજ આપું?")
    instruction = _instruction("voice")

    assert "ગર્ભપાત = abortion" in hints and "સાયલેજ = Silage" in hints
    assert "'abortion' must be translated as 'ગર્ભપાત'" in instruction
    assert "'Silage' must be translated as 'સાયલેજ'" in instruction
    assert "તરવાઈ" not in instruction
    assert "abortion -> ગર્ભપાત" in asyncio.run(search_terms("abortion", max_results=1))


def test_chat_keeps_its_abortion_row():
    instruction = _instruction("chat")

    assert "'ABORTION' must be translated as 'પશુ તરવાઈ જવું'" in instruction
    assert "ગર્ભપાત" not in instruction and "સાયલેજ" not in instruction


@pytest.mark.parametrize("answer, term, voice_says, chat_says", [
    ("Feed the heifer a balanced ration.", "HEIFER", "વોડકી, ઝોટુ, પાડી", "પાડી"),
    (
        "Check the TDN — Total Digestible Nutrients of the feed.",
        "TDN — Total Digestible Nutrients", "ટીડીએન", "કુલ પાચ્ય પોષક તત્વ (ટીડીએન)",
    ),
], ids=["heifer", "tdn"])
def test_voice_says_heifer_and_tdn_as_voice_oan_api_does(answer, term, voice_says, chat_says):
    """voice-oan-api amul-prod's rows, not chat's preferred terms: પાડી alone is
    a female buffalo calf."""
    assert f"'{term}' must be translated as '{voice_says}'" in _instruction("voice", answer)
    assert f"'{term}' must be translated as '{chat_says}'" in _instruction("chat", answer)
