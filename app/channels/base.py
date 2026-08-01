"""What differs between delivery channels, expressed as data.

Naming, because the codebase already overloads one of these words: ``channel``
is the delivery medium (web | whatsapp | telephony). The orthogonal axis —
which pipeline shape a turn runs — is a *surface*, and it gets introduced when
there is a second one to model, not before.

The record carries only fields something reads. Voice's differences (a
non_meaningful stage chat lacks, a TTFT deadline, telephony liveness, TTS
chunking) are not guessed at here: they are structural, and modelling them as
flags would put a channel conditional inside the orchestrator, which is the
duplication this seam exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Channel(str, Enum):
    WEB = "web"
    WHATSAPP = "whatsapp"


@dataclass(frozen=True)
class ChannelProfile:
    channel: Channel
    #: Hard cap on rendered response length, or None for no cap.
    response_max_chars: Optional[int] = None
