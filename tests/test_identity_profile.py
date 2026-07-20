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
    monkeypatch.setattr(chat_module, "translate_to_english_pretranslation", _unexpected_pretranslation)

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
                use_translation_pipeline=True,
                pipeline_variant="legacy",
            )
        ]

    chunks = asyncio.run(_collect())

    assert len(chunks) == 1
    assert chunks[0].startswith("| Field | Details |\n|---|---|")
    assert "| Name | Sarlaben |" in chunks[0]
    assert recorded["messages"] is not None
    assert len(recorded["messages"]) == 2
