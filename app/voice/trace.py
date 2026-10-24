"""Voice's trace: the ``turn`` root of a call turn and what it records.

From voice-oan-api (``app/services/voice_trace.py`` at amul-dev ``3b19835``). It
sends turn.v1 (amul-telemetry ``telemetry/turn.v1.yaml``): one root named ``turn``
with flat ``amul.*`` metadata, the blocks voice recorded on the root written as the
output of the step each describes (moderation, pre_translation, agent, farmer_context,
nudge, ...), and the caller's words and number as they are: turn.v1 keeps them raw and
nothing here hashes them (amul-telemetry, decided 2026-10-13 and 2026-10-20).

What voice recorded stays recorded: ``metadata``, ``stages``, ``timings_ms`` and
``counters`` are the turn's in-memory record, and the ``VOICE_TRACE_SUMMARY`` log line
holds them. The record keeps only the length of the caller's words and not the caller's
number, so the log holds neither the words, the number, nor a hash of them.

``current_trace`` is how voice's modules reach the turn's trace. Voice passed its
trace around by hand; on the seam, ``VoiceTelemetry`` (``app/voice/telemetry.py``)
sets it for the length of the turn's root.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from app.config import settings
from app.llm_core import trace as pipeline_trace
from app.services.telemetry_stamps import TURN_ROOT, forward_chat_telemetry_metadata, turn_v1_fields

logger = logging.getLogger(__name__)

# Voice's milk readout routes, as turn.v1's milk_readout step names its source.
_MILK_READOUT_SOURCES = {"outbound_milk_readout": "cache", "outbound_milk_readout_cold": "cold"}
# What the nudge block says, as the nudge step's output.
_NUDGE_OUTPUT_KEYS = ("armed", "sent", "trigger", "cancel_reason")


def _now() -> float:
    return time.perf_counter()


def _duration_ms(started_at: float) -> float:
    return round((_now() - started_at) * 1000.0, 2)


def trace_text(text: Optional[str]) -> Optional[str]:
    """``text`` as the trace carries it: the caller's words as they are (turn.v1 keeps
    the raw text, decided 2026-10-13); nothing for no words."""
    return text or None


def _safe_update(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse observation update failed: %s", exc)


def text_length(text: Optional[str]) -> dict[str, Any]:
    """What the record (and so the summary log) keeps of some words: their length,
    never the words and never a hash of them. The trace carries the words."""
    return {"chars": len(text or "")}


def step_tiers(execution: Any, step: Any) -> dict[str, str]:
    """turn.v1's tier metadata for one step (``amul.configured_tier``,
    ``amul.served_tier``, ``amul.attempts``), from what the walker recorded on this
    turn's pipeline trace; ``{}`` when nothing was. Tracing never breaks the turn."""
    try:
        return pipeline_trace.step_metadata(execution.begin_trace(), step)
    except Exception as exc:
        logger.debug("Step tiers unavailable for %s: %s", step, exc)
        return {}


def check_mode(*, skipped: bool = False, failed_closed: bool = False, failed_open: bool = False) -> str:
    """turn.v1's ``amul.check_mode``: how a check concluded. ``model`` when its
    model decided; ``failed_closed`` when a failure stopped the turn, ``failed_open``
    when one let it through; ``skipped`` when the check did not run."""
    if skipped:
        return "skipped"
    if failed_closed:
        return "failed_closed"
    if failed_open:
        return "failed_open"
    return "model"


def _status(exc_type: Any, exc: BaseException | None) -> tuple[str, str, str | None]:
    """(status, level, status_message) for how a step ended."""
    if exc_type is None:
        return "ok", "DEFAULT", None
    status = "cancelled" if exc_type.__name__ == "CancelledError" else "error"
    return status, "ERROR", str(exc)[:300] if exc else exc_type.__name__


class _Observation:
    """One step's Langfuse observation, opened as the current span so what the
    step calls nests under it; or, ``under_root``, as a direct child of the root
    that is not the current span, for a step that runs while another step is
    current and whose calls are handed it as their parent (the sink's
    post_translation span, open while the agent step is). Without a root it is
    opened under the current span.

    What a step knows only once it ran (its output, its ``amul.*`` metadata, an
    error it recorded without raising, an input known only at its end) is given to
    ``record`` and written when the step closes, with level ERROR and the reason
    when the step raised. ``observe=False`` keeps the step without an observation.
    """

    def __init__(
        self,
        trace: "VoiceTrace",
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        model: str | None = None,
        observe: bool = True,
        under_root: bool = False,
    ) -> None:
        self.trace = trace
        self.name = name
        self.as_type = as_type
        self.input = input
        self.model = model
        self.observe = observe
        self.under_root = under_root
        self.output: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.level: str | None = None
        self.status_message: str | None = None
        self.observation: Any | None = None
        self._cm: Any | None = None
        self._late_input: Any | None = None

    def __enter__(self) -> "_Observation":
        if self.observe and self.trace.enabled and self.trace.langfuse_client is not None:
            try:
                if self.under_root and self.trace.root_observation is not None:
                    opened_with = {"input": self.input, "model": self.model}
                    self.observation = self.trace.root_observation.start_observation(
                        name=self.name,
                        as_type=self.as_type,
                        **{key: value for key, value in opened_with.items() if value is not None},
                    )
                else:
                    self._cm = self.trace.langfuse_client.start_as_current_observation(
                        name=self.name,
                        as_type=self.as_type,
                        input=self.input,
                        model=self.model,
                    )
                    self.observation = self._cm.__enter__()
            except Exception as exc:
                logger.debug("Langfuse observation start failed for %s: %s", self.name, exc)
                self._cm = None
                self.observation = None
        return self

    def record(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if input is not None:
            self._late_input = input
        if output is not None:
            self.output = output
        if metadata:
            self.metadata.update(metadata)
        if level is not None:
            self.level = level
        if status_message is not None:
            self.status_message = status_message

    def __exit__(self, exc_type, exc, tb) -> bool:
        _, level, status_message = _status(exc_type, exc)
        _safe_update(
            self.observation,
            input=self._late_input,
            output=self.output,
            metadata=self.metadata or None,
            level=self.level or level,
            status_message=self.status_message or status_message,
        )
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc, tb)
            except Exception as close_exc:
                logger.debug("Langfuse observation close failed for %s: %s", self.name, close_exc)
        elif self.observation is not None:
            # Opened off the root: ended here, as the SDK leaves that to the opener.
            try:
                self.observation.end()
            except Exception as close_exc:
                logger.debug("Langfuse observation close failed for %s: %s", self.name, close_exc)
        return False


class _StageTimer(_Observation):
    """A step voice times: its observation, and its entry in the turn's ``stages``
    record (name, duration, status and the ``metadata`` given here, which is the
    record's, not the observation's). ``observe=False`` times without an
    observation, for a step whose observation is opened elsewhere (one span for
    the turn's post_translation, timed batch by batch)."""

    def __init__(
        self,
        trace: "VoiceTrace",
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        model: str | None = None,
        observe: bool = True,
    ) -> None:
        super().__init__(trace, name, as_type=as_type, input=input, model=model, observe=observe)
        self.recorded = dict(metadata or {})
        self.started_at = _now()

    def __enter__(self) -> "_StageTimer":
        super().__enter__()
        self.started_at = _now()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status, _, _ = _status(exc_type, exc)
        duration = _duration_ms(self.started_at)
        self.trace.stages.append({"name": self.name, "duration_ms": duration, "status": status, **self.recorded})
        self.trace.stage_totals_ms[self.name] = round(
            self.trace.stage_totals_ms.get(self.name, 0.0) + duration,
            2,
        )
        return super().__exit__(exc_type, exc, tb)


@dataclass
class VoiceTrace:
    session_id: str
    user_id: str
    source_lang: str
    target_lang: str
    query: str
    provider: Optional[str] = None
    process_id: Optional[str] = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=_now)
    enabled: bool = field(default_factory=lambda: bool(getattr(settings, "enable_voice_tracing", True)))
    langfuse_client: Any | None = None
    route: Optional[str] = None
    outcome: Optional[str] = None
    #: A failure the turn caught and answered with the trouble line, as its reason.
    failure: Optional[str] = None
    #: The trace id of the request that took the session over, read when this one
    #: found itself stale (turn.v1's amul.superseded_by).
    superseded_by: Optional[str] = None
    root_observation: Any | None = None
    #: The root's amul.* keys (turn.v1), set by VoiceTelemetry before the root opens.
    root_metadata: dict[str, str] = field(default_factory=dict)
    environment: Optional[str] = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    response_parts: list[str] = field(default_factory=list)
    nudge_recorded: bool = False
    finished: bool = False

    def __post_init__(self) -> None:
        if self.enabled:
            try:
                from app.observability import get_langfuse_client

                self.langfuse_client = get_langfuse_client()
            except Exception as exc:
                logger.debug("Langfuse client lookup failed: %s", exc)
        self.metadata.update(
            {
                **forward_chat_telemetry_metadata(),
                "trace_id": self.trace_id,
                "provider": self.provider,
                "process_id": self.process_id,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "query": text_length(self.query),
            }
        )

    def text(self, value: Optional[str]) -> Optional[str]:
        """``value`` as the trace carries it: see ``trace_text``."""
        return trace_text(value)

    def attach_stage_timing(self, name: str, duration_ms: float, **metadata: Any) -> None:
        self.stages.append({"name": name, "duration_ms": round(duration_ms, 2), "status": "ok", **metadata})
        self.stage_totals_ms[name] = round(self.stage_totals_ms.get(name, 0.0) + duration_ms, 2)

    def stage(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        model: str | None = None,
        observe: bool = True,
    ) -> _StageTimer:
        """A timed step: its observation (unless ``observe=False``), and its entry in ``stages``."""
        return _StageTimer(self, name, as_type=as_type, input=input, metadata=metadata, model=model, observe=observe)

    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        model: str | None = None,
        under_root: bool = False,
    ) -> _Observation:
        """A step's observation alone, for a step voice times itself; ``under_root``
        for one opened as the root's direct child while another step is current."""
        return _Observation(self, name, as_type=as_type, input=input, model=model, under_root=under_root)

    @contextmanager
    def request_context(self) -> Iterator[None]:
        """Open the turn's root for the full streaming request.

        Trace-level fields live on the root only: ``propagate_attributes`` sets
        them on the current span and on every span opened inside its context, so
        it is entered and left before any step opens. A step that carried them
        would be a trace-level event, and a late one would rewrite the turn
        (amul-telemetry docs/TRANSFORM.md). The surface's Langfuse environment is
        the one thing propagated to every span (#422); it is not a trace-level
        field. ``finish`` writes what is known only at the end while the root is
        still the current span, so it runs inside this context.
        """
        if not self.enabled or self.langfuse_client is None:
            yield
            return

        try:
            from langfuse import propagate_attributes

            from app.observability import root_trace_id, surface_environment
        except Exception:
            yield
            return

        # Langfuse uses the active OTel context. Keeping this context open
        # across the async generator makes moderation, translation, tools, and
        # pydantic-ai child spans attach to one turn trace.
        stack = ExitStack()
        try:
            self.environment = surface_environment("voice")
            stack.enter_context(propagate_attributes(environment=self.environment))
            # The trace id made when the request claimed the session: the request it
            # took over from names it as amul.superseded_by.
            with root_trace_id(self.trace_id):
                self.root_observation = stack.enter_context(
                    self.langfuse_client.start_as_current_observation(
                        name=TURN_ROOT,
                        as_type="span",
                        metadata=self.root_metadata,
                        end_on_exit=False,
                    )
                )
            opened_as = getattr(self.root_observation, "trace_id", self.trace_id)
            if opened_as != self.trace_id:
                logger.warning(
                    "Voice turn root did not open with its claimed trace id; amul.superseded_by and "
                    "amul.utterance_id that name it point to no trace - session_id=%s",
                    self.session_id,
                )
            with propagate_attributes(
                environment=self.environment,
                user_id=(self.user_id or "")[:200] or None,  # the caller's number as given
                session_id=(self.session_id or "")[:200] or None,
                metadata=self.root_metadata,
            ):
                pass
            question = self.text(self.query)
            if question is not None:
                # The raw question; the languages and the call are amul.* keys.
                self.langfuse_client.set_current_trace_io(input=question)
        except Exception as exc:
            logger.debug("Voice Langfuse request context setup failed: %s", exc)
            stack.close()
            yield
            return

        try:
            yield
        finally:
            try:
                stack.close()
            except Exception as exc:
                logger.debug("Voice Langfuse request context close failed: %s", exc)

    def set_language(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.metadata["source_lang"] = source_lang
        self.metadata["target_lang"] = target_lang

    def set_route(self, route: str) -> None:
        self.route = route
        self.metadata["route"] = route
        source = _MILK_READOUT_SOURCES.get(route)
        if source is not None:
            # An outbound call that read the farmer's milk collection out, as
            # turn.v1's milk_readout step; the other routes are the turn's outcome.
            self.record_child_observation(name="milk_readout", output={"source": source})

    def set_outcome(self, outcome: str) -> None:
        self.outcome = outcome
        self.metadata["outcome"] = outcome

    def record_failure(self, error: BaseException) -> None:
        """A failure the sink caught and spoke the trouble line for: nothing is raised out
        of the turn, so this is the reason its root gives. The first one is the cause."""
        if self.failure is None:
            self.failure = _reason(error)

    def set_moderation(self, verdict: Any | None) -> None:
        def _pick(name: str, fallback: Any = None) -> Any:
            if verdict is None:
                return fallback
            value = getattr(verdict, name, fallback)
            return fallback if value is None else value

        attempts = _pick("attempts", [])
        fallback_used = _pick("fallback_used", None)
        if fallback_used is None:
            fallback_used = bool(
                isinstance(attempts, list)
                and len(attempts) > 1
                and isinstance(attempts[0], dict)
                and attempts[0].get("status") == "error"
            )

        payload: dict[str, Any] = {
            "available": verdict is not None,
            "requested_tier": _pick("requested_tier"),
            "requested_provider": _pick("requested_provider"),
            "requested_model": _pick("requested_model"),
            "actual_tier": _pick("actual_tier"),
            "actual_provider": _pick("actual_provider"),
            "actual_model": _pick("actual_model"),
            "fallback_used": fallback_used,
            "attempts": attempts if isinstance(attempts, list) else [],
        }
        if verdict is not None:
            payload.update(
                {
                    "category": getattr(verdict, "category", None),
                    "rejected": bool(getattr(verdict, "rejected", False)),
                    "failed_open": bool(getattr(verdict, "failed_open", False)),
                    "failed_closed": bool(getattr(verdict, "failed_closed", False)),
                    "reason": text_length(getattr(verdict, "reason", None)),
                }
            )
        self.metadata["moderation"] = payload

    def record_child_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any | None = None,
        output: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """A step recorded once it is over: opened and closed here, with its output.

        A direct child of the root: the checks resolve while the agent step is the
        current span, and they are not part of the agent's run. Without a root (a
        module recording on its own) it is opened under the current span."""
        if not self.enabled or self.langfuse_client is None:
            return
        try:
            if self.root_observation is None:
                with self.observation(name, as_type=as_type, input=input, model=model) as step:
                    step.record(output=output, metadata=metadata, level=level, status_message=status_message)
                return
            fields: dict[str, Any] = {"input": input, "output": output, "metadata": metadata or None}
            if as_type == "generation":
                fields["model"] = model
            self.root_observation.start_observation(
                name=name,
                as_type=as_type,
                level=level or "DEFAULT",
                status_message=status_message,
                **{key: value for key, value in fields.items() if value is not None},
            ).end()
        except Exception as exc:
            logger.debug("Langfuse child observation failed for %s: %s", name, exc)

    def set_pretranslation(
        self,
        *,
        text: str,
        provider: str,
        fallback_used: bool,
        requested_tier: Optional[str] = None,
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
        actual_tier: Optional[str] = None,
        actual_provider: Optional[str] = None,
        actual_model: Optional[str] = None,
        attempts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "text": text_length(text),
            "provider": provider,
            "fallback_used": fallback_used,
        }
        if requested_tier is not None:
            payload["requested_tier"] = requested_tier
        if requested_provider is not None:
            payload["requested_provider"] = requested_provider
        if requested_model is not None:
            payload["requested_model"] = requested_model
        if actual_tier is not None:
            payload["actual_tier"] = actual_tier
        if actual_provider is not None:
            payload["actual_provider"] = actual_provider
        if actual_model is not None:
            payload["actual_model"] = actual_model
        if attempts is not None:
            payload["attempts"] = attempts
        self.metadata["pretranslation"] = payload

    def set_farmer_context(
        self,
        *,
        source: Any = None,
        stale: Any = None,
        unions: Optional[list[str]] = None,
        farmer_info_chars: int = 0,
        technician_info_chars: int = 0,
    ) -> dict[str, Any]:
        """Record the farmer block built for the agent; returned as the
        farmer_context step's output."""
        self.metadata["farmer_context"] = {
            "source": source,
            "stale": stale,
            "unions": unions or [],
            "farmer_info_chars": farmer_info_chars,
            "technician_info_chars": technician_info_chars,
        }
        return self.metadata["farmer_context"]

    def set_agent(
        self,
        *,
        signed_in: bool,
        output: str,
        new_messages: Optional[list[Any]] = None,
        requested_tier: Optional[str] = None,
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
        actual_tier: Optional[str] = None,
        actual_provider: Optional[str] = None,
        actual_model: Optional[str] = None,
        first_token_committed_tier: Optional[str] = None,
        first_token_committed_provider: Optional[str] = None,
        first_token_committed_model: Optional[str] = None,
        fallback_used: Optional[bool] = None,
        attempts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        tool_calls = 0
        for msg in new_messages or []:
            for part in getattr(msg, "parts", None) or []:
                if getattr(part, "tool_name", None):
                    tool_calls += 1
        payload: dict[str, Any] = {
            "signed_in": signed_in,
            "output_chars": len(output or ""),
            "new_message_count": len(new_messages or []),
            "tool_call_count": tool_calls,
        }
        if requested_tier is not None:
            payload["requested_tier"] = requested_tier
        if requested_provider is not None:
            payload["requested_provider"] = requested_provider
        if requested_model is not None:
            payload["requested_model"] = requested_model
        if actual_tier is not None:
            payload["actual_tier"] = actual_tier
        if actual_provider is not None:
            payload["actual_provider"] = actual_provider
        if actual_model is not None:
            payload["actual_model"] = actual_model
        if first_token_committed_tier is not None:
            payload["first_token_committed_tier"] = first_token_committed_tier
        if first_token_committed_provider is not None:
            payload["first_token_committed_provider"] = first_token_committed_provider
        if first_token_committed_model is not None:
            payload["first_token_committed_model"] = first_token_committed_model
        if fallback_used is not None:
            payload["fallback_used"] = fallback_used
        if attempts is not None:
            payload["attempts"] = attempts
        self.metadata["agent"] = payload

    def set_nudge(self, **values: Any) -> None:
        current = self.metadata.setdefault("nudge", {})
        current.update(values)
        if values.get("sent"):
            # The nudge step starts when the nudge went out.
            self._record_nudge()

    def _record_nudge(self) -> None:
        """turn.v1's nudge step, once per turn: when the nudge was sent, else when
        the turn ends with what it recorded about the nudge that did not go out."""
        nudge = self.metadata.get("nudge")
        if not nudge or self.nudge_recorded:
            return
        self.nudge_recorded = True
        self.record_child_observation(
            name="nudge", output={key: nudge[key] for key in _NUDGE_OUTPUT_KEYS if key in nudge}
        )

    def increment(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by

    def mark(self, name: str) -> None:
        self.timings_ms.setdefault(name, _duration_ms(self.started_at))

    def record_emit(self, text: Any, *, kind: str = "assistant") -> None:
        if not isinstance(text, str):
            return
        self.mark("ttft_ms")
        if kind == "assistant" and text.strip():
            self.mark("ttfr_ms")
            self.response_parts.append(text)

    def finish(
        self,
        outcome: Optional[str] = None,
        *,
        error: BaseException | None = None,
        served_tier: Optional[str] = None,
    ) -> None:
        """Close the turn: what it knows only at its end goes on the root, which is
        the current span again here (inside ``request_context``), and the summary
        goes to the log."""
        if self.finished:
            return
        self.finished = True
        if outcome:
            self.set_outcome(outcome)
        elif self.outcome is None:
            self.set_outcome("success")
        if error is not None:
            self.metadata["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:300],
            }

        summary = {
            **self.metadata,
            "session_id": self.session_id,
            "total_ms": _duration_ms(self.started_at),
            "timings_ms": self.timings_ms,
            "stage_totals_ms": self.stage_totals_ms,
            "stages": self.stages[-50:],
            "counters": self.counters,
            "response": text_length("".join(self.response_parts)),
        }
        if self.root_observation is not None:
            self._close_root(error=error, served_tier=served_tier, total_ms=summary["total_ms"])
        if getattr(settings, "voice_trace_log_summary", True):
            try:
                logger.info("VOICE_TRACE_SUMMARY %s", json.dumps(summary, ensure_ascii=False, default=str))
            except Exception as exc:
                logger.debug("Voice trace summary logging failed: %s", exc)

    def _error_reason(self, error: BaseException | None) -> str:
        if error is not None:
            return _reason(error)
        return self.failure or self.route or "the turn ended without an answer"

    def _close_root(self, *, error: BaseException | None, served_tier: Optional[str], total_ms: float) -> None:
        self._record_nudge()
        try:
            from langfuse import propagate_attributes

            # The outcome, the tier that answered and the timings are trace-level
            # fields: set on the root, the current span, and nowhere else. Each stage's
            # time is amul.<stage>_ms: not every stage is a step (history_load and the
            # session's ownership run before the root opens). For the same reason the
            # turn's whole time is amul.total_ms, as the summary log has it.
            fields = turn_v1_fields(
                outcome=self.outcome,
                # Only a stale request names the one that took over (turn.v1).
                superseded_by=self.superseded_by if self.outcome == "stale_request" else None,
                served_tier=served_tier,
                total_ms=_whole_ms(total_ms),
                ttft_ms=_whole_ms(self.timings_ms.get("ttft_ms")),
                ttfr_ms=_whole_ms(self.timings_ms.get("ttfr_ms")),
                first_agent_text_ms=_whole_ms(self.timings_ms.get("first_agent_text_ms")),
                first_translation_chunk_ms=_whole_ms(self.timings_ms.get("first_translation_chunk_ms")),
                **{f"{stage}_ms": _whole_ms(ms) for stage, ms in self.stage_totals_ms.items()},
            )
            if fields:
                with propagate_attributes(environment=self.environment, metadata=fields):
                    pass
            answer = self.text("".join(self.response_parts))
            if answer is not None:
                # The answer as the caller heard it: every chunk spoken, in order.
                self.langfuse_client.set_current_trace_io(output=answer)
        except Exception as exc:
            logger.debug("Voice Langfuse root close failed: %s", exc)
        if error is not None or self.outcome == "error":
            # level = ERROR whenever the outcome is error (turn.v1), with the reason: what
            # the turn raised, else the failure the sink caught and spoke the trouble line
            # for, else the path that answered it.
            _safe_update(self.root_observation, level="ERROR", status_message=self._error_reason(error))
        try:
            self.root_observation.end()
        except Exception as exc:
            logger.debug("Langfuse root observation end failed: %s", exc)


def _reason(error: BaseException) -> str:
    return (str(error) or type(error).__name__)[:300]


def _whole_ms(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(round(value))


def create_voice_trace(
    *,
    session_id: str,
    user_id: str,
    query: str,
    source_lang: str,
    target_lang: str,
    provider: Optional[str],
    process_id: Optional[str],
    trace_id: Optional[str] = None,
) -> VoiceTrace:
    # Without one made when the request claimed the session, the trace makes its own.
    return VoiceTrace(
        session_id=session_id,
        user_id=user_id,
        query=query,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider,
        process_id=process_id,
        **({"trace_id": trace_id} if trace_id else {}),
    )


_current: ContextVar[Optional[VoiceTrace]] = ContextVar("voice_trace", default=None)


def current_trace() -> VoiceTrace:
    """The trace of the voice turn running now.

    Outside one (a voice module called on its own, as its tests do) it is a
    trace that is never sent, so callers record without checking.
    """
    trace = _current.get()
    if trace is None:
        return VoiceTrace(
            session_id="", user_id="", source_lang="", target_lang="", query="", enabled=False
        )
    return trace
