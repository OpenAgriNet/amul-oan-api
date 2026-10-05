"""The seam contract: what a turn receives, what it emits, and the surface it runs.

``app/channels`` models the *delivery medium* (web | whatsapp) — how rendered
text is delivered. This module models the orthogonal axis ``app/channels/base``
reserved: which *pipeline shape* a turn runs (chat today, voice when it is
ported). The two compose rather than nest; see docs/channel-seam-design.md.

Transport-free by construction: nothing here imports FastAPI, Redis, or a
telemetry client, and nothing here may. The transport adapter (the chat router's
``stream_chat_messages`` today) builds a ``Turn``, consumes ``Emission`` values,
and decides what each one means on its wire.

Following the rule in ``app/channels/base``, a field appears only when something
reads it. The sink and the telemetry are on ``SurfaceProfile`` because ``run_turn``
reads them and chat populates them with its real ones. The background set and the
liveness channel are not stubbed: they land with the voice surface that
populates them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Union,
)

from app.channels.base import ChannelProfile

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from pydantic_ai.messages import ModelMessage

    from app.personas import ChatPersona


class Surface(str, Enum):
    #: The text pipeline served from this repo.
    CHAT = "chat"
    # VOICE lands when voice is ported off voice-oan-api, not before.


@dataclass(frozen=True)
class Turn:
    """Request-invariant input: what the transport established before orchestration.

    It replaces the positional parameter list ``stream_chat_messages`` passes; it
    is not a new product concept.

    Deliberately absent:

    * the request's ``stream`` flag — SSE versus accumulated JSON is a transport
      decision over the same emission stream;
    * a model or resolved pipeline profile — ``llm_core`` resolves the sticky
      pipeline from ``session_id`` inside the turn;
    * a Redis client, a telemetry span, ``FastAPI.BackgroundTasks``, or a mutable
      artifact output list — runtime services are wired where the turn is
      composed, and output leaves only as ``Emission`` values.
    """

    query: str
    session_id: str
    source_lang: str
    target_lang: str
    user_id: str
    #: Verified identity claims (the decoded JWT). Read-only: the adapter hands
    #: over a private copy, so nothing downstream can alter the caller's claims.
    authenticated_user: Mapping[str, Any]
    history: tuple[ModelMessage, ...]
    #: Where this turn's history is persisted. Resolved by the transport; differs
    #: from ``session_id`` when a persona keeps its own conversation.
    history_session_id: str
    channel: ChannelProfile
    persona: ChatPersona


# ── what a turn emits ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextEmission:
    """Ordinary caller-visible text, from the model or a deterministic path."""

    text: str
    #: Bypass the surface's output normalizer. Chat has none, so its adapter
    #: ignores this; voice's hangup line ``"Goodbye."`` must skip the Gujarati
    #: allow-list normalizer or it is spoken as ``"."``.
    raw: bool = False


@dataclass(frozen=True)
class AgentActivityEmission:
    """The agent has begun work that fallback must not replay (tool calls).

    On chat this signal is consumed inside ``llm_core``'s first-token walker,
    which is where the commit decision is made, so ``run_turn`` never yields it
    and the chat adapter drops it. It is part of the union so a surface that
    needs the commit point outside ``llm_core`` has a typed place for it rather
    than a sentinel crossing stage boundaries.
    """


@dataclass(frozen=True)
class SideChannelEmission:
    """Caller-visible output delivered OUTSIDE the response stream.

    Voice's telephony nudge is an HTTP POST to a separate endpoint; typed apart
    so it can neither be dropped by a text-only signature nor spoken in-band by
    the TTS batcher. Chat has none.
    """

    text: str


@dataclass(frozen=True)
class ArtifactEmission:
    """The turn's validated private documents (e.g. a Soil Health Card).

    Kept outside model text, translation, TTS, history and trace bodies. One
    emission carries the whole batch because the web client's contract is a
    single terminal frame; see ``app.chat_artifacts``.
    """

    artifacts: tuple[Mapping[str, Any], ...]


Emission = Union[TextEmission, AgentActivityEmission, SideChannelEmission, ArtifactEmission]


# ── runtime services wired where the turn is composed ───────────────────────


class DeferredScheduler(Protocol):
    """Runs work AFTER the turn's response has been delivered.

    "After" is load-bearing: suggestion generation reads the history this turn
    writes, so running it concurrently with the turn would build suggestions
    from the previous turn. The chat adapter backs this with
    ``FastAPI.BackgroundTasks``, which runs once the response has finished.
    """

    def schedule(self, fn: Callable[..., Any], /, *args: Any) -> None: ...


# ── the pre-turn classifier chain ───────────────────────────────────────────


@dataclass(frozen=True)
class ClassifierResult:
    """A pre-turn classifier's decision to answer the turn without the agent.

    Returned by a classifier that MATCHED. A classifier that does not apply
    returns ``None`` and the chain moves on.
    """

    #: The text to emit to the caller.
    canned_text: str

    #: Names the path in logs and telemetry (e.g. ``"identity"``).
    label: str

    #: Messages to append to the session history, or None to persist nothing.
    #: Chat's identity path persists a (user, assistant) pair; voice's
    #: hold-message path deliberately persists nothing.
    history_pair: Optional[tuple[ModelMessage, ...]] = None

    #: Carried onto the ``TextEmission``; see ``TextEmission.raw``.
    raw: bool = False


#: Decides, before any background task is spawned or any model is called,
#: whether the turn can be answered outright. Running before the background set
#: is why classifiers never have to cancel anything.
Classifier = Callable[[Turn], Awaitable[Optional[ClassifierResult]]]


# ── the sink ────────────────────────────────────────────────────────────────


class TurnSink(Protocol):
    """One turn's sink: the agent's English token stream in, the caller's text out.

    Chat and voice need different objects here, not two settings of one. Chat
    passes English through or stream-translates it in sentence batches; voice is
    a streaming batcher with its own normalizer and cross-chunk state.
    """

    def stream(self, english: AsyncIterator[str]) -> AsyncIterator[str]:
        """What the caller receives, chunk by chunk."""
        ...

    def final_text(self) -> Optional[str]:
        """The turn's complete output for the trace, or None if nothing was produced."""
        ...


class SinkFactory(Protocol):
    """Builds a turn's sink from what ``run_turn`` knows at that point."""

    def __call__(
        self,
        turn: Turn,
        *,
        execution: Any,
        deps: Any,
        translate_to: Optional[str],
    ) -> TurnSink: ...


# ── telemetry ───────────────────────────────────────────────────────────────


class TurnTelemetry(Protocol):
    """One turn's telemetry: the single root span and how the turn went.

    ``run_turn`` owns the lifecycle (one root, opened and closed once, the
    outcome recorded on every exit); the surface owns the contract written
    inside it. Chat writes ``chat.turn.v1``; voice's root, stamps and outcome
    vocabulary are its own.
    """

    def root(self) -> AbstractContextManager[None]:
        """Open the turn's root span for the whole turn, and close it after."""
        ...

    def record_output(self, text: str, label: str) -> None:
        """Record what the caller received. ``label`` names the path for logs."""
        ...

    def record_outcome(self, outcome: str) -> None:
        """Record how the turn ended: ``success``, ``cancelled`` or ``error``."""
        ...


class TelemetryFactory(Protocol):
    """Builds a turn's telemetry from what ``run_turn`` knows before the root opens."""

    def __call__(
        self,
        turn: Turn,
        *,
        pipeline_profile: str,
        pipeline_trace: Any,
    ) -> TurnTelemetry: ...


@dataclass(frozen=True)
class SurfaceProfile:
    """What a surface populates. One field per structure that is built."""

    surface: Surface
    #: Ordered. First match wins and ends the turn. Chat has one; voice has five.
    classifiers: tuple[Classifier, ...] = ()
    #: Turns the agent's English stream into caller text. A surface that always
    #: answers from its classifiers never reaches it, so it may be left unset.
    sink: Optional[SinkFactory] = None
    #: The turn's root span and what is recorded in it. Left unset (as in tests
    #: that build a bare surface), the turn runs without writing a trace.
    telemetry: Optional[TelemetryFactory] = None
