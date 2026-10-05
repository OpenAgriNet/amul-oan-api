"""Voice's trace: the ``agent_journey`` root of a call turn and what it records.

From voice-oan-api (``app/services/voice_trace.py`` at amul-dev ``3b19835``),
unchanged except for the ``voice.turn.v1`` stamps from voice-oan-api#308. The
``service`` stamp is this repo's, so turns served from here are told apart.

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
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from app.config import settings
from app.services.telemetry_stamps import forward_voice_telemetry_metadata

logger = logging.getLogger(__name__)

_VALID_TEXT_MODES = {"preview_hash", "full", "none"}
_HASH_SALT = "voice-oan-api"


def _now() -> float:
    return time.perf_counter()


def _duration_ms(started_at: float) -> float:
    return round((_now() - started_at) * 1000.0, 2)


def _hash(value: str) -> str:
    return hashlib.sha256(f"{_HASH_SALT}:{value}".encode("utf-8")).hexdigest()


def _safe_update(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse observation update failed: %s", exc)


def sanitize_text(text: Optional[str], *, mode: Optional[str] = None) -> dict[str, Any]:
    """Return trace-safe text metadata.

    By default only the length and a hash. ``preview_hash`` and ``full`` put the
    caller's words in the trace, so they are only for a Langfuse approved to hold
    them, with a documented retention and access policy.
    """
    value = text or ""
    active_mode = mode or getattr(settings, "voice_trace_text_mode", "none")
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


@dataclass
class _StageRecord:
    name: str
    started_at: float
    as_type: str = "span"
    metadata: dict[str, Any] = field(default_factory=dict)
    observation: Any | None = None


class _StageTimer:
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
        self.trace = trace
        self.name = name
        self.as_type = as_type
        self.input = input
        self.metadata = metadata or {}
        self.model = model
        self.record: _StageRecord | None = None
        self._cm: Any | None = None

    def __enter__(self) -> "_StageTimer":
        observation = None
        if self.trace.enabled and self.trace.langfuse_client is not None:
            try:
                self._cm = self.trace.langfuse_client.start_as_current_observation(
                    name=self.name,
                    as_type=self.as_type,
                    input=self.input,
                    metadata=self.metadata,
                    model=self.model,
                )
                observation = self._cm.__enter__()
            except Exception as exc:
                logger.debug("Langfuse stage start failed for %s: %s", self.name, exc)
                self._cm = None

        self.record = _StageRecord(
            name=self.name,
            started_at=_now(),
            as_type=self.as_type,
            metadata=dict(self.metadata),
            observation=observation,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.record is None:
            return False
        status = "ok"
        level = "DEFAULT"
        status_message = None
        if exc_type is not None:
            status = "cancelled" if exc_type.__name__ == "CancelledError" else "error"
            level = "ERROR"
            status_message = str(exc)[:300] if exc else exc_type.__name__

        duration = _duration_ms(self.record.started_at)
        item = {
            "name": self.name,
            "duration_ms": duration,
            "status": status,
            **self.record.metadata,
        }
        self.trace.stages.append(item)
        self.trace.stage_totals_ms[self.name] = round(
            self.trace.stage_totals_ms.get(self.name, 0.0) + duration,
            2,
        )
        _safe_update(
            self.record.observation,
            metadata={"duration_ms": duration, "status": status, **self.record.metadata},
            level=level,
            status_message=status_message,
        )
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc, tb)
            except Exception as close_exc:
                logger.debug("Langfuse stage close failed for %s: %s", self.name, close_exc)
        return False


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
    stages: list[dict[str, Any]] = field(default_factory=list)
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    response_parts: list[str] = field(default_factory=list)
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
                **forward_voice_telemetry_metadata(),
                "trace_id": self.trace_id,
                "provider": self.provider,
                "process_id": self.process_id,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "user_id_hash": _hash(self.user_id or "anonymous"),
                "query": sanitize_text(self.query),
            }
        )

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
        return _StageTimer(self, name, as_type=as_type, input=input, metadata=metadata, model=model)

    @contextmanager
    def request_context(self) -> Iterator[None]:
        """Open the root Langfuse observation for the full streaming request.

        The resolved-pipeline-config compact keys (``pipeline_profile`` /
        ``pipeline_flags`` / ``pc_<step>``) are added to ``self.metadata`` by
        ``stream_voice_message`` BEFORE this opens, so they ride along in the
        ``metadata=self.metadata`` passed to the root observation below (the path
        that actually lands — a nested blob via ``update_current_trace`` does not,
        this SDK has no such method)."""
        if not self.enabled or self.langfuse_client is None:
            yield
            return

        try:
            from langfuse import propagate_attributes
        except Exception:
            yield
            return

        # Langfuse uses the active OTel context. Keeping this context open
        # across the async generator makes moderation, translation, tools, and
        # pydantic-ai child spans attach to one agent_journey trace.
        stack = ExitStack()
        try:
            self.root_observation = stack.enter_context(
                self.langfuse_client.start_as_current_observation(
                    name="agent_journey",
                    as_type="span",
                    input=self.metadata["query"],
                    metadata=self.metadata,
                    end_on_exit=False,
                )
            )
            stack.enter_context(
                propagate_attributes(
                    # The caller's hash, never their phone number.
                    user_id=self.metadata["user_id_hash"],
                    session_id=(self.session_id or "")[:200] or None,
                    metadata={
                        "process_id": str(self.process_id or "")[:200],
                        "trace_id": self.trace_id,
                        "provider": str(self.provider or ""),
                    },
                    tags=[
                        "voice",
                        str(self.provider or "api"),
                        # Mirror chat's `pipeline_profile:<name>` trace tag so voice
                        # sessions are sliceable per profile (N-way) in Langfuse.
                        # pipeline_profile is set on metadata before this opens.
                        f"pipeline_profile:{self.metadata.get('pipeline_profile') or 'managed'}",
                    ],
                    trace_name="agent_journey",
                )
            )
        except Exception as exc:
            logger.debug("Voice Langfuse request context setup failed: %s", exc)
            stack.close()
            with nullcontext():
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
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        if not self.enabled or self.langfuse_client is None:
            return
        try:
            with self.langfuse_client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=input,
                metadata=metadata,
                model=model,
            ) as observation:
                _safe_update(
                    observation,
                    output=output,
                    metadata=metadata,
                    level=level,
                    status_message=status_message,
                )
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
    ) -> None:
        self.metadata["farmer_context"] = {
            "source": source,
            "stale": stale,
            "unions": unions or [],
            "farmer_info_chars": farmer_info_chars,
            "technician_info_chars": technician_info_chars,
        }

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

    def finish(self, outcome: Optional[str] = None, *, error: BaseException | None = None) -> None:
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
        _safe_update(
            self.root_observation,
            output=summary["response"],
            metadata=summary,
            level="ERROR" if error is not None else "DEFAULT",
            status_message=str(error)[:300] if error is not None else None,
        )
        if self.root_observation is not None:
            try:
                self.root_observation.end()
            except Exception as exc:
                logger.debug("Langfuse root observation end failed: %s", exc)
        if getattr(settings, "voice_trace_log_summary", True):
            try:
                logger.info("VOICE_TRACE_SUMMARY %s", json.dumps(_hash_only(summary), ensure_ascii=False, default=str))
            except Exception as exc:
                logger.debug("Voice trace summary logging failed: %s", exc)


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
