"""Voice's trace: the ``turn`` root of a call turn and what it records.

From voice-oan-api (``app/services/voice_trace.py`` at amul-dev ``3b19835``). It
sends turn.v1 (amul-telemetry ``telemetry/turn.v1.yaml``): one root named ``turn``
with flat ``amul.*`` metadata, the blocks voice recorded on the root written as the
output of the step each describes (moderation, pre_translation, agent, farmer_context,
nudge, ...), and the caller's words on the trace only in the ``full`` text mode.

What voice recorded stays recorded: ``metadata``, ``stages``, ``timings_ms`` and
``counters`` are the turn's in-memory record, and the ``VOICE_TRACE_SUMMARY`` log line
holds them hash-only, whatever the text mode. Only what goes to Langfuse changed.

``current_trace`` is how voice's modules reach the turn's trace. Voice passed its
trace around by hand; on the seam, ``VoiceTelemetry`` (``app/voice/telemetry.py``)
sets it for the length of the turn's root.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from app.config import settings
from app.llm_core import runtime
from app.llm_core import trace as pipeline_trace
from app.services.telemetry_stamps import TURN_ROOT, forward_chat_telemetry_metadata, turn_v1_fields

logger = logging.getLogger(__name__)

_VALID_TEXT_MODES = {"preview_hash", "full", "none"}
_VALID_USER_ID_MODES = {"hash", "phone"}
_HASH_SALT = "voice-oan-api"

# Voice's milk readout routes, as turn.v1's milk_readout step names its source.
_MILK_READOUT_SOURCES = {"outbound_milk_readout": "cache", "outbound_milk_readout_cold": "cold"}
# What the nudge block says, as the nudge step's output.
_NUDGE_OUTPUT_KEYS = ("armed", "sent", "trigger", "cancel_reason")


def _now() -> float:
    return time.perf_counter()


def _duration_ms(started_at: float) -> float:
    return round((_now() - started_at) * 1000.0, 2)


def _hash(value: str) -> str:
    return hashlib.sha256(f"{_HASH_SALT}:{value}".encode("utf-8")).hexdigest()


def _user_id_mode() -> str:
    mode = runtime.surface_setting(
        "voice", "trace_user_id_mode", getattr(settings, "voice_trace_user_id_mode", "hash")
    )
    return mode if mode in _VALID_USER_ID_MODES else "hash"


def _langfuse_user_id(user_id: Optional[str], user_id_hash: str) -> str:
    """The caller's hash by default. ``phone`` sends the caller's id as given, so
    it is only for a Langfuse approved to hold phone numbers."""
    if _user_id_mode() == "phone":
        return (user_id or "anonymous")[:200]
    return user_id_hash


def _text_mode() -> str:
    mode = runtime.surface_setting(
        "voice", "trace_text_mode", getattr(settings, "voice_trace_text_mode", "none")
    )
    return mode if mode in _VALID_TEXT_MODES else "none"


def trace_text(text: Optional[str]) -> Optional[str]:
    """``text`` as the trace carries it: the words themselves in the ``full`` text
    mode, else nothing. turn.v1 has no place for a preview or a hash (the summary
    log keeps the hash), so ``preview_hash`` sends no text, like ``none``."""
    if text and _text_mode() == "full":
        return text
    return None


def _safe_update(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse observation update failed: %s", exc)


def sanitize_text(text: Optional[str], *, mode: Optional[str] = None) -> dict[str, Any]:
    """Return trace-safe text metadata for the summary log.

    By default only the length and a hash. ``preview_hash`` and ``full`` add the
    caller's words to the record; the log drops them (``_hash_only``), and the
    trace carries them only in ``full`` (``trace_text``).
    """
    value = text or ""
    active_mode = mode or _text_mode()
    if active_mode not in _VALID_TEXT_MODES:
        active_mode = "none"

    payload: dict[str, Any] = {
        "chars": len(value),
        "sha256": _hash(value) if value else None,
    }
    if active_mode == "full":
        payload["text"] = value
    elif active_mode == "preview_hash":
        preview_chars = max(0, int(getattr(settings, "voice_trace_preview_chars", 120)))
        payload["preview"] = value[:preview_chars]
    return payload


def _hash_only(value: Any) -> Any:
    """``value`` with every sanitized text cut down to its length and hash. Logs
    never hold the caller's words, whatever text mode the trace is sent with."""
    if isinstance(value, dict):
        sanitized = "chars" in value and "sha256" in value
        return {
            key: _hash_only(item)
            for key, item in value.items()
            if not (sanitized and key in ("preview", "text"))
        }
    if isinstance(value, list):
        return [_hash_only(item) for item in value]
    return value


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
    step calls nests under it.

    What a step knows only once it ran (its output, its ``amul.*`` metadata, an
    error it recorded without raising) is given to ``record`` and written when the
    step closes, with level ERROR and the reason when the step raised.
    """

    def __init__(
        self,
        trace: "VoiceTrace",
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        model: str | None = None,
    ) -> None:
        self.trace = trace
        self.name = name
        self.as_type = as_type
        self.input = input
        self.model = model
        self.output: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.level: str | None = None
        self.status_message: str | None = None
        self.observation: Any | None = None
        self._cm: Any | None = None

    def __enter__(self) -> "_Observation":
        if self.trace.enabled and self.trace.langfuse_client is not None:
            try:
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
        return self

    def record(
        self,
        *,
        output: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
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
        return False


class _StageTimer(_Observation):
    """A step voice times: its observation, and its entry in the turn's ``stages``
    record (name, duration, status and the ``metadata`` given here, which is the
    record's, not the observation's)."""

    def __init__(
        self,
        trace: "VoiceTrace",
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
        model: str | None = None,
    ) -> None:
        super().__init__(trace, name, as_type=as_type, input=input, model=model)
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
                "user_id_hash": _hash(self.user_id or "anonymous"),
                "query": sanitize_text(self.query),
            }
        )

    @property
    def sends_phone(self) -> bool:
        """Whether this deployment sends the caller's number (VOICE_TRACE_USER_ID_MODE=phone)."""
        return _user_id_mode() == "phone"

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
    ) -> _StageTimer:
        """A timed step: its observation, and its entry in ``stages``."""
        return _StageTimer(self, name, as_type=as_type, input=input, metadata=metadata, model=model)

    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        model: str | None = None,
    ) -> _Observation:
        """A step's observation alone, for a step voice times itself."""
        return _Observation(self, name, as_type=as_type, input=input, model=model)

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

            from app.observability import surface_environment
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
            self.root_observation = stack.enter_context(
                self.langfuse_client.start_as_current_observation(
                    name=TURN_ROOT,
                    as_type="span",
                    metadata=self.root_metadata,
                    end_on_exit=False,
                )
            )
            with propagate_attributes(
                environment=self.environment,
                user_id=_langfuse_user_id(self.user_id, self.metadata["user_id_hash"]),
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
                    "reason": sanitize_text(getattr(verdict, "reason", None)),
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
            "text": sanitize_text(text),
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
            "response": sanitize_text("".join(self.response_parts)),
        }
        if self.root_observation is not None:
            self._close_root(error=error, served_tier=served_tier)
        if getattr(settings, "voice_trace_log_summary", True):
            try:
                logger.info("VOICE_TRACE_SUMMARY %s", json.dumps(_hash_only(summary), ensure_ascii=False, default=str))
            except Exception as exc:
                logger.debug("Voice trace summary logging failed: %s", exc)

    def _close_root(self, *, error: BaseException | None, served_tier: Optional[str]) -> None:
        self._record_nudge()
        try:
            from langfuse import propagate_attributes

            # The outcome, the tier that answered and the timings are trace-level
            # fields: set on the root, the current span, and nowhere else.
            fields = turn_v1_fields(
                outcome=self.outcome,
                served_tier=served_tier,
                ttft_ms=_whole_ms(self.timings_ms.get("ttft_ms")),
                ttfr_ms=_whole_ms(self.timings_ms.get("ttfr_ms")),
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
            # level = ERROR whenever the outcome is error (turn.v1), with the reason
            # when the turn raised; a failure the sink spoke the trouble line for has
            # its reason in the log.
            _safe_update(self.root_observation, level="ERROR", status_message=str(error)[:300] if error else None)
        try:
            self.root_observation.end()
        except Exception as exc:
            logger.debug("Langfuse root observation end failed: %s", exc)


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
) -> VoiceTrace:
    return VoiceTrace(
        session_id=session_id,
        user_id=user_id,
        query=query,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider,
        process_id=process_id,
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
