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
reads them and chat populates them with its real ones; so is pretranslation.
``Turn.call`` is there because voice's classifiers read it, and the background set
and liveness because ``run_turn`` runs voice's.
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
class TelephonyCall:
    """The phone call a voice turn belongs to, as the voice adapter found it.

    Established before the turn starts, like the rest of ``Turn``.
    """

    process_id: Optional[str]
    #: This turn is the farmer's reply to the telephony provider's opening
    #: question on an outbound call. The outbound consent gate owns that turn,
    #: so the greeting, identity and fragment short-circuits leave it alone.
    #: The adapter works it out from the outbound stage in Redis.
    outbound_consent_turn: bool = False


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
    #: The call a voice turn belongs to. None on chat.
    call: Optional[TelephonyCall] = None


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

    It is due while the turn is waiting inside an ``await`` (on the model), where
    a generator cannot hand anything out, so it goes through the
    ``SideChannelSender`` ``run_turn`` is given rather than being yielded.
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


class SideChannelSender(Protocol):
    """Delivers a ``SideChannelEmission`` now, outside the response stream.

    Voice's sends the telephony nudge. Chat has none.
    """

    async def send(self, emission: SideChannelEmission) -> None: ...


class StalenessCheck(Protocol):
    """Whether this request should still speak.

    A call turn goes stale when the caller hangs up or a newer request takes
    over the session; one that keeps talking can speak over the newer one.
    Returns None while the turn should carry on, otherwise the outcome to
    record (voice's are ``client_disconnected`` and ``stale_request``), and the
    turn stops without emitting or writing anything more. ``reason`` names the
    point that asked, for the logs. Chat has none.
    """

    async def __call__(self, reason: str) -> Optional[str]: ...


# ── the pre-turn classifier chain ───────────────────────────────────────────


@dataclass(frozen=True)
class ClassifierResult:
    """A decision to answer the turn with fixed text instead of the agent's.

    Returned by a pre-turn classifier that MATCHED (one that does not apply
    returns ``None`` and the chain moves on), and by the gate before the first
    emission when a background check says the agent's answer must not be sent.
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


# ── the background set and the gate ─────────────────────────────────────────


class TurnBackground(Protocol):
    """One turn's background work, and the gate that consults it.

    Built by ``run_turn`` right after the classifier chain, and building it
    starts the work, so it runs alongside everything up to the agent's first
    chunk. ``run_turn`` pulls that chunk, then asks ``gate``; nothing reaches
    the caller before the answer. ``close`` runs on every exit.
    """

    async def gate(self) -> Optional[ClassifierResult]:
        """None to let the agent's answer through, else what to answer instead."""
        ...

    async def close(self) -> None:
        """Cancel and reap whatever is still running. Never raises."""
        ...


class BackgroundFactory(Protocol):
    """Builds, and so starts, a turn's background work."""

    def __call__(self, turn: Turn, *, execution: Any) -> TurnBackground: ...


# ── liveness ────────────────────────────────────────────────────────────────


class TurnLiveness(Protocol):
    """Keeps a caller who is waiting on the model from hearing only silence.

    Started by ``run_turn`` right after the classifier chain, and stopped just
    before the first thing the caller hears, or when the turn ends without that.
    """

    async def stop(self) -> None:
        """Stop, and reap anything still running. Idempotent; never raises."""
        ...


class LivenessFactory(Protocol):
    """Builds, and so starts, a turn's liveness."""

    def __call__(
        self,
        turn: Turn,
        *,
        started_at: float,
        send: SideChannelSender,
        is_stale: Optional[StalenessCheck],
    ) -> TurnLiveness: ...


# ── pretranslation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pretranslated:
    """The caller's query as the agent will read it."""

    query: str
    #: The language the agent answers in, before any output translation.
    lang: str


class Pretranslation(Protocol):
    """Puts the caller's query into the language the agent reads.

    Returns what the agent is asked, or an answer that ends the turn instead
    (voice asks the caller to repeat when nothing usable came through). It is
    given the turn's background set: voice's must decline a rejected query
    rather than ask for a repeat, and hands the background the English words
    history keeps for the turn. Chat's ignores it.
    """

    async def __call__(
        self,
        turn: Turn,
        *,
        execution: Any,
        background: Optional[TurnBackground],
    ) -> Union[Pretranslated, ClassifierResult]: ...


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
    """Builds a turn's sink from what ``run_turn`` knows at that point.

    ``is_stale`` is the turn's staleness check, for a sink that must stop
    speaking mid-answer when the request goes stale (voice's). Chat has none.
    """

    def __call__(
        self,
        turn: Turn,
        *,
        execution: Any,
        deps: Any,
        translate_to: Optional[str],
        is_stale: Optional[StalenessCheck],
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
        """Record how the turn ended: ``success``, ``cancelled``, ``error``, or
        the outcome a ``StalenessCheck`` stopped it with."""
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
    #: Checks that run alongside the turn and gate its first emission. Chat has
    #: none: its moderation decides before the agent starts.
    background: Optional[BackgroundFactory] = None
    #: What the caller hears while the model works. It needs a side channel, so a
    #: turn run without a ``SideChannelSender`` has none. Chat has none.
    liveness: Optional[LivenessFactory] = None
    #: Puts the query into the language the agent reads. Like the sink, a surface
    #: that always answers from its classifiers may leave it unset.
    pretranslation: Optional[Pretranslation] = None
