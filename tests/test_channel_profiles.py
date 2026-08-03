"""The channel seam: what differs between delivery channels, as data."""
import dataclasses
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.base import Channel
from app.channels.chat import WHATSAPP_RESPONSE_MAX_CHARS, WEB, WHATSAPP, profile_for


def test_whatsapp_caps_response_length_and_web_does_not():
    assert profile_for("whatsapp").response_max_chars == WHATSAPP_RESPONSE_MAX_CHARS
    assert profile_for("web").response_max_chars is None


def test_channel_resolution_is_case_insensitive():
    assert profile_for("WhatsApp") is WHATSAPP
    assert profile_for("WHATSAPP") is WHATSAPP


@pytest.mark.parametrize("value", [None, "", " ", "telephony", "sms", "voice", "nonsense"])
def test_unknown_and_missing_channels_fall_back_to_web(value):
    """Matches the previous behaviour: anything but literal "whatsapp" got no cap."""
    assert profile_for(value) is WEB


def test_every_channel_has_a_profile():
    for c in Channel:
        assert profile_for(c.value).channel is c


def test_profiles_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        WEB.response_max_chars = 10
