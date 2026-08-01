"""The chat surface's profiles, one per delivery channel."""
from __future__ import annotations

from app.channels.base import Channel, ChannelProfile, ModerationMode, Surface

#: WhatsApp renders long messages poorly and truncates server-side, so responses
#: are capped before rendering rather than after.
WHATSAPP_RESPONSE_MAX_CHARS = 1600

WEB = ChannelProfile(
    surface=Surface.CHAT,
    channel=Channel.WEB,
    moderation=ModerationMode.BLOCKING,
    translation_channel="chat",
    response_max_chars=None,
)

WHATSAPP = ChannelProfile(
    surface=Surface.CHAT,
    channel=Channel.WHATSAPP,
    moderation=ModerationMode.BLOCKING,
    translation_channel="chat",
    response_max_chars=WHATSAPP_RESPONSE_MAX_CHARS,
)

_BY_CHANNEL = {Channel.WEB: WEB, Channel.WHATSAPP: WHATSAPP}


def profile_for(channel: str | None) -> ChannelProfile:
    """Resolve a request's channel string to a profile.

    Unknown and missing values fall back to WEB, matching the previous behaviour
    where anything other than the literal "whatsapp" got no character cap.
    """
    try:
        resolved = Channel((channel or "web").strip().lower())
    except ValueError:
        resolved = Channel.WEB
    return _BY_CHANNEL[resolved]
