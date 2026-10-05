"""Voice's turn telemetry for ``SurfaceProfile.telemetry``: one ``voice.turn.v1`` root.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``): the trace
``stream_voice_message`` opens, the metadata it sets before the root, the
``pipeline_profile`` score inside it, and how it finishes. ``run_turn`` drives it,
and voice's modules record their stages through ``current_trace``.
"""
from __future__ import annotations

from contextlib import contextmanager

from app.llm_core import trace as _pipeline_trace
from app.turn.types import Turn
from app.voice.outbound import normalize_call_type
from app.voice.trace import _current, create_voice_trace
from helpers.utils import get_logger

try:
    from langfuse import get_client as _get_langfuse_client
except ImportError:  # pragma: no cover
    _get_langfuse_client = None

logger = get_logger(__name__)

# How run_turn's outcomes read in voice's vocabulary. A turn cut off because the
# caller went away is one voice records as a disconnect.
_OUTCOMES = {"cancelled": "client_disconnected"}


class VoiceTelemetry:
    """One call turn's ``agent_journey`` trace."""

    def __init__(self, turn: Turn, *, pipeline_profile: str, pipeline_trace) -> None:
        call = turn.call
        self._session_id = turn.session_id
        self._pipeline_profile = pipeline_profile
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
        agent = pipeline_trace.steps.get("agent") if pipeline_trace is not None else None
        trace.metadata["pipeline_profile"] = pipeline_profile
        trace.metadata["request_model"] = agent.model if agent is not None else None
        trace.metadata["request_provider"] = agent.provider if agent is not None else None
        trace.metadata["call_type"] = normalize_call_type(call.call_type if call is not None else None)
        # pipeline_profile, pipeline_flags and one pc_<step> per step, on the root.
        _pipeline_trace.add_compact_metadata(pipeline_trace, trace.metadata)
        self._trace = trace

    @contextmanager
    def root(self):
        trace = self._trace
        token = _current.set(trace)
        try:
            # The root stays open for the whole turn, so moderation, translation,
            # tools and pydantic-ai's spans all land in one agent_journey trace.
            with trace.request_context():
                self._score_pipeline_profile()
                trace.set_language(self._source_lang, self._target_lang)
                yield
        except Exception as exc:
            trace.finish(trace.outcome or "error", error=exc)
            raise
        finally:
            trace.finish(trace.outcome or "success")
            try:
                _current.reset(token)
            except ValueError:
                # Closed from another context, as when the event loop finalises
                # an abandoned turn; there is nothing to restore there.
                pass

    def _score_pipeline_profile(self) -> None:
        # From inside the root: outside it Langfuse skips the score. score_id is
        # per session, so later turns of the call upsert the same score.
        if _get_langfuse_client is None:
            return
        try:
            _get_langfuse_client().score_current_trace(
                name="pipeline_profile",
                value=self._pipeline_profile,
                data_type="CATEGORICAL",
                score_id=f"voice-variant-{(self._session_id or '')[:180]}",
                comment="Sticky pipeline variant for this voice session",
            )
        except Exception as e:  # pragma: no cover
            logger.debug("Langfuse: voice pipeline_profile score failed: %s", e)

    def record_output(self, text: str, label: str) -> None:
        # The sink records the agent's answer chunk by chunk as the caller hears it.
        if label == "final":
            return
        # Any other answer names the path that gave it, which voice records as the
        # route. The closing line follows the agent's answer and leaves it as it is.
        if label != "closing":
            self._trace.set_route(label)
        self._trace.record_emit(text)

    def record_outcome(self, outcome: str) -> None:
        self._trace.set_outcome(_OUTCOMES.get(outcome, outcome))
