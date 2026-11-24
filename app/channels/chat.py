"""The chat surface's profiles, one per delivery channel."""
from __future__ import annotations

from app.channels.base import Channel, ChannelProfile

#: WhatsApp renders long messages poorly and truncates server-side, so responses
#: are capped before rendering rather than after.
WHATSAPP_RESPONSE_MAX_CHARS = 1600

WEB = ChannelProfile(channel=Channel.WEB, response_max_chars=None, supports_rich_artifacts=True)
WHATSAPP = ChannelProfile(
    channel=Channel.WHATSAPP,
    response_max_chars=WHATSAPP_RESPONSE_MAX_CHARS,
    supports_rich_artifacts=False,
)

_BY_CHANNEL = {WEB.channel: WEB, WHATSAPP.channel: WHATSAPP}

# A Channel member without a profile would raise KeyError at request time.
assert set(_BY_CHANNEL) == set(Channel), f"no profile for {set(Channel) - set(_BY_CHANNEL)}"


#: The Bengali-Assamese Unicode block. Of the languages we serve only Bengali is
#: written in it, so a message in this script names its own language.
_BENGALI_BLOCK = range(0x0980, 0x0A00)


def whatsapp_languages(query: str, source_lang: str, target_lang: str) -> tuple[str, str]:
    """The (source, target) languages a WhatsApp turn runs in.

    The WhatsApp integration only sends gu/hi/en, so a farmer writing Bengali
    arrived tagged "en" and was answered in English (prod, Nov 2026). A message
    written mostly in Bengali script is answered in Bengali whatever the caller
    sent; any other message keeps the caller's languages.
    """
    script_chars = [c for c in query or "" if c.isalpha() or ord(c) in _BENGALI_BLOCK]
    bengali = sum(1 for c in script_chars if ord(c) in _BENGALI_BLOCK)
    if script_chars and bengali * 2 >= len(script_chars):
        return "bn", "bn"
    return source_lang, target_lang


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
