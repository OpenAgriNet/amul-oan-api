import os
import asyncio
from importlib import import_module

import pytest

from app.services.identity_profile import (
    build_identity_profile_table,
    is_identity_query,
)

os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("who are you", True),
        ("who is sarlaben", True),
        ("introduce yourself", True),
        ("તમે કોણ છો?", True),
        ("તું કોણ છે?", True),
        ("તમારું પરિચય આપો", True),
        ("સરલાબેન કોણ છે", True),
        ("my cow has fever", False),
        # Lightly-padded identity phrases still short-circuit.
        ("hey, who are you?", True),
        ("introduce yourself please", True),
        ("તમે કોણ છો?", True),
        # Compound queries carrying a real agricultural question must NOT be
        # hijacked by the identity short-circuit — the second question would be
        # silently dropped otherwise.
        ("who are you and my cow has fever", False),
        ("tell me who are you and what medicine for mastitis", False),
        ("તમે કોણ છો અને મારી ગાય ને તાવ છે", False),
        # Bengali.
        ("আপনি কে?", True),
        ("তুমি কে", True),
        ("সরলাবেন কে?", True),
        ("আপনার পরিচয় দিন", True),
        # কে starts কেমন ("how") — "how are you" is not an identity query.
        ("আপনি কেমন আছেন?", False),
        ("আপনি কে এবং আমার গরুর জ্বর হয়েছে", False),
        # Punjabi (Gurmukhi).
        ("ਤੁਸੀਂ ਕੌਣ ਹੋ?", True),
        ("ਤੂੰ ਕੌਣ ਹੈਂ", True),
        ("ਸਰਲਾਬੇਨ ਕੌਣ ਹੈ?", True),
        ("ਇਹ ਕਿਹੜੀ ਸੇਵਾ ਹੈ", True),
        ("ਤੁਸੀਂ ਕੌਣ ਹੋ ਅਤੇ ਮੇਰੀ ਗਾਂ ਨੂੰ ਬੁਖ਼ਾਰ ਹੈ", False),
        # Marathi.
        ("तुम्ही कोण आहात?", True),
        ("तू कोण आहेस", True),
        ("सरलाबेन कोण आहे?", True),
        ("तुमची ओळख सांगा", True),
        ("ही कोणती सेवा आहे", True),
        # Hindi uses कौन, not Marathi's कोण — it must not match.
        ("आप कौन हैं", False),
        ("तुम्ही कोण आहात आणि माझ्या गाईला ताप आहे", False),
    ],
)
def test_identity_query_detection(query: str, expected: bool):
    assert is_identity_query(query) is expected


def test_identity_table_english_format():
    table = build_identity_profile_table("en", "english", "who are you")
    assert table.startswith("| Field | Details |\n|---|---|")
    assert "| Name | Sarlaben |" in table
    assert "| Organization | Amul |" in table
    assert "Your trusted digital dairy companion" in table


def test_identity_table_gujarati_format():
    table = build_identity_profile_table("gu", "gujarati", "તમારું પરિચય આપો")
    assert table.startswith("| ક્ષેત્ર | વિગતો |\n|---|---|")
    assert "| નામ | સરલાબેન |" in table
    assert "| સંસ્થા | અમૂલ |" in table
    assert "| ઉપલબ્ધતા | ૨૪x૭ ૦૮૦-૩૫૪૫૩૫૪૫ પર ચેટ, વોઇસ કૉલ અને વોટ્સએપ |" in table
    assert "| મારા મૂલ્યો | ખેડૂત પ્રથમ; વિશ્વસનીય અને ભરોસાપાત્ર માર્ગદર્શન; સહકારી ભાવના; સૌ માટે સુલભતા; સતત શિક્ષણ અને નવીનતા |" in table
    assert "તમારી વિશ્વસનીય ડિજિટલ ડેરી સાથી" in table


def test_identity_table_bengali_format():
    table = build_identity_profile_table("bn", "bn", "আপনি কে?")
    assert table.startswith("| ক্ষেত্র | বিবরণ |\n|---|---|")
    assert "| নাম | সরলাবেন |" in table
    assert "| প্রতিষ্ঠান | আমুল |" in table
    assert "আপনার বিশ্বস্ত ডিজিটাল" in table


def test_identity_table_bengali_selected_from_script():
    # No explicit language: a Bengali-script query still gets the Bengali table.
    table = build_identity_profile_table("", "", "আপনি কে?")
    assert table.startswith("| ক্ষেত্র | বিবরণ |")


def test_identity_table_punjabi_format():
    table = build_identity_profile_table("pa", "pa", "ਤੁਸੀਂ ਕੌਣ ਹੋ?")
    assert table.startswith("| ਖੇਤਰ | ਵੇਰਵਾ |\n|---|---|")
    assert "| ਨਾਮ | ਸਰਲਾਬੇਨ |" in table
    assert "| ਸੰਸਥਾ | ਅਮੂਲ |" in table
    assert "ਤੁਹਾਡੀ ਭਰੋਸੇਯੋਗ ਡਿਜੀਟਲ" in table


def test_identity_table_punjabi_selected_from_script():
    # No explicit language: a Gurmukhi query still gets the Punjabi table.
    table = build_identity_profile_table("", "", "ਤੁਸੀਂ ਕੌਣ ਹੋ?")
    assert table.startswith("| ਖੇਤਰ | ਵੇਰਵਾ |")


def test_identity_table_gujarati_not_stolen_by_adjacent_gurmukhi_range():
    # Gurmukhi (U+0A00-U+0A7F) and Gujarati (U+0A80-U+0AFF) are adjacent blocks;
    # a Gujarati query must still get the Gujarati table.
    table = build_identity_profile_table("", "", "તમે કોણ છો")
    assert table.startswith("| ક્ષેત્ર | વિગતો |")
def test_identity_table_marathi_format():
    table = build_identity_profile_table("mr", "mr", "तुम्ही कोण आहात?")
    assert table.startswith("| क्षेत्र | तपशील |\n|---|---|")
    assert "| नाव | सरलाबेन |" in table
    assert "| संस्था | अमूल |" in table
    assert "तुमची विश्वासार्ह डिजिटल" in table


def test_identity_table_marathi_selected_from_wording_not_script():
    # Devanagari is shared with Hindi, so the Marathi table is reached by
    # Marathi-specific wording, not by the script alone.
    table = build_identity_profile_table("", "", "तुम्ही कोण आहात?")
    assert table.startswith("| क्षेत्र | तपशील |")


def test_identity_table_hindi_devanagari_still_falls_back_to_english():
    # Regression: adding Marathi must not hijack Hindi, which has no table.
    # The query must be Devanagari — an English one returns on an earlier branch
    # and never exercises the script/wording check this is meant to pin.
    table = build_identity_profile_table("hi", "hi", "आप कौन हैं?")
    assert table.startswith("| Field | Details |")
    # Undeclared language, Hindi wording: still English, not the Marathi table.
    assert build_identity_profile_table("", "", "आप कौन हैं?").startswith("| Field | Details |")


def test_chat_identity_short_circuit_bypasses_moderation_and_translation(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    chat_module = import_module("app.services.chat")
    background_tasks = fastapi.BackgroundTasks()

    recorded: dict = {"messages": None}

    async def _fake_update_history(_session_id, messages):
        recorded["messages"] = messages

    async def _unexpected_moderation(*args, **kwargs):  # pragma: no cover - safety assertion
        raise AssertionError("moderation must not run on identity query")

    async def _unexpected_pretranslation(*args, **kwargs):  # pragma: no cover - safety assertion
        raise AssertionError("pretranslation must not run on identity query")

    monkeypatch.setattr(chat_module, "update_message_history", _fake_update_history)
    monkeypatch.setattr(chat_module.moderation_agent, "run", _unexpected_moderation)
    monkeypatch.setattr(chat_module, "pretranslate_with_tier", _unexpected_pretranslation)

    async def _collect():
        return [
            chunk
            async for chunk in chat_module.stream_chat_messages(
                query="who are you",
                session_id="identity-session",
                source_lang="en",
                target_lang="english",
                channel="web",
                user_id="anonymous",
                history=[],
                user_info={},
                background_tasks=background_tasks,
            )
        ]

    chunks = asyncio.run(_collect())

    assert len(chunks) == 1
    assert chunks[0].startswith("| Field | Details |\n|---|---|")
    assert "| Name | Sarlaben |" in chunks[0]
    assert recorded["messages"] is not None
    assert len(recorded["messages"]) == 2
