"""The chat surface's profiles, one per delivery channel."""
from __future__ import annotations

from app.channels.base import Channel, ChannelProfile

#: WhatsApp renders long messages poorly and truncates server-side, so responses
#: are capped before rendering rather than after.
WHATSAPP_RESPONSE_MAX_CHARS = 1600

WEB = ChannelProfile(channel=Channel.WEB, response_max_chars=None)
WHATSAPP = ChannelProfile(channel=Channel.WHATSAPP, response_max_chars=WHATSAPP_RESPONSE_MAX_CHARS)

_BY_CHANNEL = {WEB.channel: WEB, WHATSAPP.channel: WHATSAPP}

# A Channel member without a profile would raise KeyError at request time.
assert set(_BY_CHANNEL) == set(Channel), f"no profile for {set(Channel) - set(_BY_CHANNEL)}"


def profile_for(channel: str | None) -> ChannelProfile:
    """Resolve a request's channel string to a profile.

    Unknown and missing values fall back to WEB, matching the previous behaviour
    where anything other than the literal "whatsapp" got no character cap.
    """
    try:
        resolved = Channel((channel or "web").lower())
    except ValueError:
        resolved = Channel.WEB
    return _BY_CHANNEL[resolved]
