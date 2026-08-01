"""The channel seam: what differs between delivery channels, as data."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.channels.base import Channel, ModerationMode, Surface
from app.channels.chat import WHATSAPP_RESPONSE_MAX_CHARS, WEB, WHATSAPP, profile_for


def test_whatsapp_caps_response_length_and_web_does_not():
    assert profile_for("whatsapp").response_max_chars == WHATSAPP_RESPONSE_MAX_CHARS
    assert profile_for("web").response_max_chars is None


def test_channel_resolution_is_case_insensitive_and_trims():
    assert profile_for("WhatsApp") is WHATSAPP
    assert profile_for("  whatsapp ") is WHATSAPP


def test_unknown_and_missing_channels_fall_back_to_web():
    """Matches the previous behaviour: anything but literal "whatsapp" got no cap."""
    for value in (None, "", "telephony", "sms", "nonsense"):
        assert profile_for(value) is WEB, value


def test_every_chat_profile_is_the_chat_surface_and_blocks_on_moderation():
    for p in (WEB, WHATSAPP):
        assert p.surface is Surface.CHAT
        assert p.moderation is ModerationMode.BLOCKING
        assert p.translation_channel == "chat"


def test_profiles_are_immutable():
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        WEB.response_max_chars = 10


def test_channel_and_surface_are_separate_axes():
    """`channel` already meant web|whatsapp; the surface axis must not reuse it."""
    assert {c.value for c in Channel} == {"web", "whatsapp"}
    assert Surface.CHAT.value == "chat"
