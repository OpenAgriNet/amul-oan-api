"""What differs between surfaces, expressed as data.

Two axes, deliberately named apart because the codebase already overloads one of
them:

  surface  chat | voice   — which pipeline shape a turn runs
  channel  web | whatsapp | telephony — the delivery medium beneath a surface

``channel`` already meant web|whatsapp throughout ``stream_chat_messages`` before
the seam existed, so the new axis takes the new word rather than redefining the
old one.

A profile is a record, not a base class to inherit from. Voice is served by
voice-oan-api today; when it folds in it should arrive as another profile plus
its stage implementations, not as a second orchestrator — that duplication is
what #171 documented and what removing the voice fork undid.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Surface(str, Enum):
    CHAT = "chat"
    # VOICE = "voice" — added when voice-oan-api folds in.


class Channel(str, Enum):
    WEB = "web"
    WHATSAPP = "whatsapp"


class ModerationMode(str, Enum):
    """Whether the turn waits on the moderation verdict before answering.

    Chat blocks. Voice runs it concurrently with pretranslation and resolves the
    verdict at the points that can still emit, because a caller cannot be left
    in silence — hence the distinction is a profile field rather than a branch.
    """

    BLOCKING = "blocking"
    CONCURRENT = "concurrent"


@dataclass(frozen=True)
class ChannelProfile:
    surface: Surface
    channel: Channel
    moderation: ModerationMode
    #: Name passed to ``app.services.translation.translation_channel``; selects
    #: per-channel post-normalization and translation rules.
    translation_channel: str
    #: Hard cap on rendered response length, or None for no cap.
    response_max_chars: Optional[int] = None

    @property
    def is_whatsapp(self) -> bool:
        return self.channel is Channel.WHATSAPP
