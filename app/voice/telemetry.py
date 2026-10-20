"""Voice's turn telemetry for ``SurfaceProfile.telemetry``: one turn.v1 root per call turn.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``): the trace
``stream_voice_message`` opens, the metadata it sets before the root, and how it
finishes. It sends turn.v1 (amul-telemetry ``telemetry/turn.v1.yaml``): the root is
named ``turn``, its metadata is flat ``amul.*``, the outcome is the ``turn_outcome``
score. ``run_turn`` drives it, and voice's modules record their steps through
``current_trace``.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from agents.tools.farmer import normalize_phone_to_mobile
from app.llm_core import Step
from app.llm_core import trace as _pipeline_trace
from app.services.telemetry_stamps import turn_v1_metadata
from app.services.translation import INDIAN_LANGUAGES
from app.turn.types import Turn
from app.voice.farmer import _is_signed_in_session
from app.voice.outbound import normalize_call_type
from app.voice.trace import _current, create_voice_trace
from helpers.utils import get_logger

try:
    from langfuse import get_client as _get_langfuse_client
except ImportError:  # pragma: no cover
    _get_langfuse_client = None

logger = get_logger(__name__)

# turn.v1 names the delivery medium a pipeline; a call is telephony whichever
# provider carried it.
PIPELINE = "telephony"


@dataclass(frozen=True)
class RouteTimings:
    """What the route did before the turn, timed as voice's router timed it."""

    #: ``time.perf_counter()`` when the request came in.
    started_at: float
    ownership_claim_ms: float
    history_load_ms: float
    history_messages: int


def _translation_mode(source_lang: str, target_lang: str) -> str:
    """turn.v1's translation_mode: ``translated`` when the turn has a translation
    step, as voice decides them (the question pre-translated unless it is English,
    the answer translated back for an Indian target language), else ``direct``."""
    translated = source_lang not in {"en", "english"} or target_lang in INDIAN_LANGUAGES
    return "translated" if translated else "direct"


class VoiceTelemetry:
    """One call turn's ``turn`` trace."""

    def __init__(
        self,
        turn: Turn,
        *,
        pipeline_profile: str,
        pipeline_trace,
        route: Optional[RouteTimings] = None,
    ) -> None:
        call = turn.call
        self._session_id = turn.session_id
        self._pipeline_trace = pipeline_trace
        self._source_lang = (turn.source_lang or "gu").strip().lower()
        self._target_lang = (turn.target_lang or "gu").strip().lower()
        trace = create_voice_trace(
            session_id=turn.session_id,
            user_id=turn.user_id,
            query=turn.query,
            source_lang=turn.source_lang,
            target_lang=turn.target_lang,
            provider=call.provider if call is not None else None,
            process_id=call.process_id if call is not None else None,
        )
        # The turn's record, for the summary log: as voice kept it.
        agent = pipeline_trace.steps.get("agent") if pipeline_trace is not None else None
        trace.metadata["pipeline_profile"] = pipeline_profile
        trace.metadata["request_model"] = agent.model if agent is not None else None
        trace.metadata["request_provider"] = agent.provider if agent is not None else None
        trace.metadata["call_type"] = normalize_call_type(call.call_type if call is not None else None)
        _pipeline_trace.add_compact_metadata(pipeline_trace, trace.metadata)
        if route is not None:
            # From when the request came in, as voice's trace was, so total_ms
            # includes claiming the session and loading its history.
            trace.started_at = route.started_at
            trace.attach_stage_timing("ownership_claim", route.ownership_claim_ms)
            trace.attach_stage_timing(
                "history_load", route.history_load_ms, history_messages=route.history_messages
            )
        # The root's trace-level keys, as turn.v1 names them. The metadata and field
        # names are built here on purpose: the contract test reads them from these calls.
        signed_in = _is_signed_in_session(turn.authenticated_user, turn.user_id)
        trace.root_metadata = turn_v1_metadata(
            pipeline=PIPELINE,
            direction=trace.metadata["call_type"],
            # As the provider names itself, lowercased (RAYA -> raya).
            provider=(call.provider or "").lower() if call is not None else None,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            translation_mode=_translation_mode(self._source_lang, self._target_lang),
            pipeline_profile=pipeline_profile,
            pipeline_flags=_pipeline_trace.compact_metadata(pipeline_trace).get("pipeline_flags"),
            persona=turn.persona,
            process_id=call.process_id if call is not None else None,
            request_epoch=call.request_epoch if call is not None else None,
            signed_in="true" if signed_in else "false",
            # The caller's number is a farmer id when the call signed in, and goes
            # out only where the deployment sends phones (VOICE_TRACE_USER_ID_MODE).
            farmer_id=normalize_phone_to_mobile(turn.user_id) if signed_in and trace.sends_phone else None,
        )
        self._trace = trace

    @contextmanager
    def root(self):
        trace = self._trace
        token = _current.set(trace)
        try:
            # The root stays open for the whole turn, so moderation, translation,
            # tools and pydantic-ai's spans all land in one trace. It is finished
            # from inside its own context, with the root the current span, so what
            # the turn knows only at its end lands on the trace.
            with trace.request_context():
                try:
                    trace.set_language(self._source_lang, self._target_lang)
                    yield
                except Exception as exc:
                    self._finish(trace.outcome or "error", error=exc)
                    raise
                finally:
                    self._finish(trace.outcome or "success")
        finally:
            try:
                _current.reset(token)
            except ValueError:
                # Closed from another context, as when the event loop finalises
                # an abandoned turn; there is nothing to restore there.
                pass

    def _finish(self, outcome: str, *, error: BaseException | None = None) -> None:
        self._trace.finish(outcome, error=error, served_tier=self._agent_served_tier())

    def _agent_served_tier(self) -> Optional[str]:
        """The tier that produced the answer, as ``route[index]``; None until the agent ran."""
        try:
            return _pipeline_trace.step_metadata(self._pipeline_trace, Step.AGENT).get("amul.served_tier")
        except Exception as exc:  # tracing never breaks the turn
            logger.debug("Voice served tier unavailable: %s", exc)
            return None

    def record_output(self, text: str, label: str) -> None:
        # The sink records the agent's answer chunk by chunk as the caller hears it.
        if label == "final":
            return
        # Any other answer names the path that gave it, which voice's record keeps
        # as the route; on the trace it is the outcome. The closing line follows
        # the agent's answer and leaves it as it is.
        if label != "closing":
            self._trace.set_route(label)
        self._trace.record_emit(text)

    def record_outcome(self, outcome: str) -> None:
        self._trace.set_outcome(outcome)
        self._score_turn_outcome(outcome)

    def _score_turn_outcome(self, outcome: str) -> None:
        # From inside the root: outside it Langfuse skips the score. Synchronous
        # and never raising: run_turn calls this in a finally that can run during
        # GeneratorExit.
        if _get_langfuse_client is None:
            return
        try:
            _get_langfuse_client().score_current_trace(
                name="turn_outcome",
                value=outcome,
                data_type="CATEGORICAL",
                comment="How the turn ended, as amul.outcome says it",
            )
        except Exception as e:  # pragma: no cover
            logger.debug("Langfuse: voice turn_outcome score failed: %s", e)
