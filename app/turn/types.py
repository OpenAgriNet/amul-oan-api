"""The channel seam: what differs between SURFACES, expressed as structure.

``app/channels`` models the *delivery medium* (web | whatsapp) — how rendered
text is delivered. This package models the orthogonal axis ``app/channels/base``
reserved: which *pipeline shape* a turn runs. Chat is one surface; voice is the
second, and folding it in must populate these structures rather than introduce a
second orchestrator.

Design constraints carried over from the research (docs/channel-seam-design.md):

* Voice's differences are **structural, not scalar**. An earlier design of
  per-surface scalar flags was rejected because nothing read them and they
  modelled the difference wrongly.
* A flat ordered stage list was also rejected: moderation has no list position
  (it is a future with several consumers at different times, awaited inside the
  agent run to gate irreversible writes), and the agent stage is split by a gate
  between producing the first chunk and emitting it.

So this file grows one structure at a time, and — following the rule stated in
``app/channels/base`` — **a field appears only when something reads it**. Empty
placeholders for the not-yet-built structures would be the rejected flag design
wearing a different hat.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Sequence


class Surface(str, Enum):
    #: The text pipeline served from this repo.
    CHAT = "chat"
    # VOICE lands when voice is ported off voice-oan-api, not before.


@dataclass(frozen=True)
class ClassifierResult:
    """A pre-turn classifier's decision to answer the turn without the agent.

    Returned by a classifier that MATCHED. A classifier that does not apply
    returns ``None`` and the chain moves on.
    """

    #: The text to emit to the caller.
    canned_text: str

    #: Messages to append to the session history, or None to persist nothing.
    #: Chat's identity path persists a (user, assistant) pair; several of voice's
    #: classifiers (hold-message, STT signal) deliberately persist nothing, which
    #: is why this is Optional rather than always-a-pair.
    history_pair: Optional[Sequence[Any]] = None

    #: Bypass the surface's output normalizer.
    #:
    #: Load-bearing, not cosmetic: voice's hangup line ``"Goodbye."`` must skip
    #: the Gujarati allow-list normalizer, which would otherwise strip the Latin
    #: characters and leave ``"."``. Chat has no normalizer today, so every chat
    #: classifier leaves this False — but the bit is modelled here because the
    #: merge is where it gets silently dropped.
    raw: bool = False


#: A pre-turn classifier: decides, before any background task is spawned or any
#: model is called, whether the turn can be answered outright.
#:
#: Runs BEFORE the background set is spawned, which is why classifiers never have
#: to cancel anything — a property worth preserving when voice's six land.
Classifier = Callable[["Turn"], Awaitable[Optional[ClassifierResult]]]


@dataclass(frozen=True)
class Turn:
    """The request-invariant inputs of one turn.

    Not a new concept — this is the argument list the orchestrator already
    receives, given a name so the structures below can be handed one value
    instead of eight positional parameters.
    """

    query: str
    session_id: str
    source_lang: str
    target_lang: str
    user_id: str
    history: Sequence[Any]
    user_info: dict


@dataclass(frozen=True)
class SurfaceProfile:
    """What a surface populates. One field per first-class structure.

    Currently: the pre-turn classifier chain. The background set, the liveness
    channel and the sink land as they are built — see the module docstring for
    why they are not stubbed in advance.
    """

    surface: Surface
    #: Ordered. First match wins and ends the turn.
    classifiers: tuple[Classifier, ...] = ()
