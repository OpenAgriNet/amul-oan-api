"""Memory is default-deny: a farmer not on MEMORY_ALLOWED_USERS gets no memory at all."""
import asyncio
from types import MappingProxyType

import pytest

import app.services.memory as memory
import app.voice.background as background
from app.channels.chat import WEB
from app.services import chat as chat_service
from agents.farmer_context import FarmerContextBundle
from app.turn.types import TelephonyCall, Turn

LISTED, OTHER = "9876543210", "9123456780"


def _open(monkeypatch, allowed, *, enabled=True, voice=True):
    monkeypatch.setattr(memory, "MEMORY_ENABLED", enabled)
    monkeypatch.setattr(memory, "MEMORY_ALLOWED_USERS", memory._parse_allowed_users(allowed))
    monkeypatch.setattr(background, "MEMORY_VOICE_ENABLED", voice)


def _chat_allowed(phone, persona="farmer", auth_type="jwt"):
    return chat_service._build_deps(
        query="q", session_id="s", lang_code="en", persona=persona, channel=WEB,
        user_info={"phone": phone, "auth_type": auth_type},
        farmer_context=(FarmerContextBundle(markdown=""), "found"),
    ).memory_read_allowed


def _voice_allowed(phone, monkeypatch):
    async def check(**kwargs):
        return None

    monkeypatch.setattr(background, "check_moderation", check)
    monkeypatch.setattr(background, "check_non_meaningful_streak", check)

    async def farmer(mobile):
        return None

    monkeypatch.setattr(background, "get_or_fetch_farmer_data", farmer)

    async def run():
        turn = Turn(
            query="q", session_id="call", source_lang="en", target_lang="en", user_id=phone,
            authenticated_user=MappingProxyType({"phone": phone}), history=(), history_session_id="call",
            channel=WEB, persona="farmer", call=TelephonyCall(process_id="p"),
        )
        bg = background.VoiceBackground(turn, execution=object(), render=None)
        return bg.memory_read_allowed

    return asyncio.run(run())


@pytest.mark.parametrize("value, parsed", [
    ("", set()),
    (" , ", set()),
    ("*", {"*"}),
    ("+91-98765-43210, 9123456780", {LISTED, OTHER}),
])
def test_list_parsing(value, parsed):
    assert memory._parse_allowed_users(value) == parsed


@pytest.mark.parametrize("allowed, enabled, phone, expected", [
    ("", True, LISTED, False),         # the flag alone opens nothing
    (OTHER, True, LISTED, False),
    (LISTED, True, LISTED, True),
    (LISTED, False, LISTED, False),    # the list alone opens nothing
    ("*", True, LISTED, True),
    ("*", True, None, False),
])
def test_open_for(monkeypatch, allowed, enabled, phone, expected):
    _open(monkeypatch, allowed, enabled=enabled)
    assert memory.memory_open_for(phone) is expected


def test_chat_gate(monkeypatch):
    _open(monkeypatch, LISTED)
    assert _chat_allowed(LISTED) is True
    assert _chat_allowed(OTHER) is False
    assert _chat_allowed(LISTED, auth_type="api_key") is False
    assert _chat_allowed(LISTED, persona="doctor") is False


def test_voice_gate(monkeypatch):
    _open(monkeypatch, LISTED)
    assert _voice_allowed(LISTED, monkeypatch) is True
    assert _voice_allowed(OTHER, monkeypatch) is False


def test_voice_stays_off_for_a_listed_farmer_without_its_switch(monkeypatch):
    _open(monkeypatch, LISTED, voice=False)
    assert _voice_allowed(LISTED, monkeypatch) is False
