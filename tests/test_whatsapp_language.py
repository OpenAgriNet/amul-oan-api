"""WhatsApp: a message is answered in the language it is written in.

The WhatsApp integration tags Gujarati-script messages "gu", Devanagari "hi" and
everything else "en". Bengali and Punjabi messages were answered in English and
Marathi ones in Hindi (prod Langfuse, 15-24 Nov 2026). The messages below are
farmers' WhatsApp messages from those traces, with the tag they arrived with.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from app.channels.whatsapp_language import whatsapp_languages
from app.models.requests import ChatRequest
from app.routers import chat as chat_router

BENGALI = [
    "দুধ উৎপাদন কীভাবে বাড়ানো যায়?",
    "৭ দিন দুধ",
    "বাংল বলুন",
    "মোসকে কি খবর দিলে ফাট বাড়বে",
    "দয়া করে আমার গতকালের দুধের হিসাবটি দিন।",
    "৭",
]

MARATHI = [
    "महेश दूध देत नाही उपाय सांगा",
    "बफेलो च्या सडाला जखम झाली आहे तरी काय उपाय करावा",
    "10/112026ते 23/11/2026दूध तपशील",
    "10 ते 20 तारखेपर्यंत. दुध संकलन",
    "दुध उत्पादन वाढवण्याचे उपाय",
    "गेल्या ७ दिवसांत जमा केलेल्या दुधाचा तपाशील",
    "11-11-2026ते 20-11-2026 चे दुध आणि पेमेंट किती होईल सांग",
    "दुधाचे दर वाढले की नाही म्हशीचे.",
    "माझा किती रुपये चे झाले",
    "एक नोव्हेंबर ते 10 नोव्हेंबर पगार स्लिप",
    "पगाराची रक्कम",
    "महेश बफेलो शरीराचे तापमान किती असते",
    "उवांचे नियंत्रण कसे करावे",
    "1-11-2026 ते 10-11-2026 किती रक्कम आहे",
]

HINDI = [
    "16 नवंबर से 23 नवंबर तक का कितना हुआ दूध",
    "मेरे गाय की दूध में snf कम बता रहा है",
    "पिछले ७ का जमा किया हुआ दूध का विवरण",
    "Buffelo का मिल्क कैसे बढ़ाएं",
    "बोनस कब तक प्राप्त होगा",
    "उसने 10 दिन पहले बच्चा दिया",
    "1नवंबर से 20नवंबर तक का हिसाब",
    "दूध जांच के लिए कितना एमएल दूध सचिव ले सकता है",
    "6.1 फैट 8.9 SNF हैं तो 5.9 फैट का रेट मिलता हैं",
    "थानेला के लिए क्या करे",
    "दूध उत्पादन कैसे बढ़ाया जा सकता है?",
    "भैंस का गोबर पतला हो रहा है 15 दिन से",
    "किलनी नियंत्रण",
    "दुध कैसे। बढ़ाया जा सकता है",
    # Gujarati written in Devanagari: not Marathi, so it stays as tagged.
    "गर्मी आह दिल नी तकलीफ होय तो",
    # One word says nothing about Hindi vs Marathi.
    "उपाय",
    "ही",
]


@pytest.mark.parametrize("query", BENGALI)
@pytest.mark.parametrize("sent", [("en", "en"), ("hi", "hi"), ("gu", "gu")])
def test_bengali_script_runs_in_bengali(query, sent):
    assert whatsapp_languages(query, *sent) == ("bn", "bn")


@pytest.mark.parametrize("sent", [("en", "en"), ("hi", "hi")])
def test_gurmukhi_runs_in_punjabi(sent):
    assert whatsapp_languages("ਦੁੱਧ ਉਤਪਾਦਨ ਕਿਵੇਂ ਵਧਾਈਏ", *sent) == ("pa", "pa")


@pytest.mark.parametrize("sent", [("en", "en"), ("gu", "gu")])
def test_gujarati_script_runs_in_gujarati(sent):
    assert whatsapp_languages("મારી ગાય દૂધ ઓછું આપે છે", *sent) == ("gu", "gu")


@pytest.mark.parametrize("query", MARATHI)
def test_marathi_wording_runs_in_marathi(query):
    assert whatsapp_languages(query, "hi", "hi") == ("mr", "mr")


@pytest.mark.parametrize("query", HINDI)
def test_hindi_tagged_devanagari_stays_hindi(query):
    assert whatsapp_languages(query, "hi", "hi") == ("hi", "hi")


def test_untagged_devanagari_runs_in_hindi():
    assert whatsapp_languages("दूध उत्पादन कैसे बढ़ाया जा सकता है?", "en", "en") == ("hi", "hi")


def test_a_marathi_tag_is_kept_for_devanagari_without_marathi_words():
    assert whatsapp_languages("किलनी नियंत्रण", "mr", "mr") == ("mr", "mr")


@pytest.mark.parametrize(("query", "sent"), [
    ("Hello", ("en", "en")),
    ("7 days milk", ("en", "en")),
    ("Amul cattle feed price for দুধ cows please", ("en", "en")),
    # Mostly-Latin Gujarati the integration tagged gu stays Gujarati.
    ("Cow ni kharidi કરવી છે", ("gu", "gu")),
    ("This is conword in ગુજરાતી પ્લીઝ", ("gu", "gu")),
    # Telugu is not a language the app offers.
    ("ఆవు రూములో చిన్న చిన్న గడ్డలు ఉన్నాయి", ("en", "en")),
    ("", ("gu", "gu")),
    ("123 ?", ("en", "en")),
])
def test_other_messages_keep_the_callers_languages(query, sent):
    assert whatsapp_languages(query, *sent) == sent


async def _langs_seen(monkeypatch, *, channel, query, source_lang="en", target_lang="en"):
    seen = {}

    async def _stream(**kwargs):
        seen.update(kwargs)
        yield "ok"

    monkeypatch.setattr(chat_router, "stream_chat_messages", _stream)
    monkeypatch.setattr(chat_router, "_get_message_history", AsyncMock(return_value=[]))
    await chat_router.chat_endpoint(
        BackgroundTasks(),
        request=ChatRequest(
            query=query, session_id="s1", stream=False, channel=channel,
            source_lang=source_lang, target_lang=target_lang,
        ),
        user_info={"phone": "9876543210", "auth_type": "api_key"},
    )
    return seen["source_lang"], seen["target_lang"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "sent", "runs_in"), [
    ("দুধ উৎপাদন কীভাবে বাড়ানো যায়?", ("en", "en"), ("bn", "bn")),
    ("ਦੁੱਧ ਉਤਪਾਦਨ ਕਿਵੇਂ ਵਧਾਈਏ", ("en", "en"), ("pa", "pa")),
    ("उवांचे नियंत्रण कसे करावे", ("hi", "hi"), ("mr", "mr")),
    ("थानेला के लिए क्या करे", ("hi", "hi"), ("hi", "hi")),
    ("Ho", ("en", "en"), ("en", "en")),
])
async def test_whatsapp_turn_runs_in_the_message_language(monkeypatch, query, sent, runs_in):
    langs = await _langs_seen(
        monkeypatch, channel="whatsapp", query=query, source_lang=sent[0], target_lang=sent[1]
    )
    assert langs == runs_in


@pytest.mark.asyncio
async def test_web_keeps_the_language_the_app_sent(monkeypatch):
    """The web app sends the language the farmer picked; it is never overridden."""
    langs = await _langs_seen(monkeypatch, channel="web", query="দুধ উৎপাদন কীভাবে বাড়ানো যায়?")
    assert langs == ("en", "en")
