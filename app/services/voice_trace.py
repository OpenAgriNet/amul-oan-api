from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from app.config import settings

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

    The default keeps enough data to debug routing and regressions without
    storing full caller text in logs or Langfuse.
    """
    value = text or ""
    active_mode = mode or getattr(settings, "voice_trace_text_mode", "preview_hash")
    if active_mode not in _VALID_TEXT_MODES:
        active_mode = "preview_hash"

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
        """Open the root Langfuse observation for the full streaming request."""
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
        # pydantic-ai child spans attach to one voice_request trace.
        stack = ExitStack()
        try:
            self.root_observation = stack.enter_context(
                self.langfuse_client.start_as_current_observation(
                    name="voice_request",
                    as_type="span",
                    input=self.metadata["query"],
                    metadata=self.metadata,
                    end_on_exit=False,
                )
            )
            stack.enter_context(
                propagate_attributes(
                    user_id=(self.user_id or "anonymous")[:200],
                    session_id=(self.session_id or "")[:200] or None,
                    metadata={
                        "process_id": str(self.process_id or "")[:200],
                        "trace_id": self.trace_id,
                        "provider": str(self.provider or ""),
                    },
                    tags=[
                        "voice",
                        str(self.provider or "api"),
                        # Mirror chat's `variant:<oss|legacy>` trace tag so voice
                        # sessions are sliceable by pipeline variant in Langfuse.
                        # pipeline_variant is set on metadata before this opens.
                        f"variant:{self.metadata.get('pipeline_variant') or 'legacy'}",
                    ],
                    trace_name="voice_request",
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
        if verdict is None:
            self.metadata["moderation"] = {"available": False}
            return
        self.metadata["moderation"] = {
            "available": True,
            "category": getattr(verdict, "category", None),
            "rejected": bool(getattr(verdict, "rejected", False)),
            "failed_open": bool(getattr(verdict, "failed_open", False)),
            "reason": sanitize_text(getattr(verdict, "reason", None)),
        }

    def set_pretranslation(
        self,
        *,
        text: str,
        provider: str,
        fallback_used: bool,
    ) -> None:
        self.metadata["pretranslation"] = {
            "text": sanitize_text(text),
            "provider": provider,
            "fallback_used": fallback_used,
        }

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

    def set_agent(self, *, signed_in: bool, output: str, new_messages: Optional[list[Any]] = None) -> None:
        tool_calls = 0
        for msg in new_messages or []:
            for part in getattr(msg, "parts", None) or []:
                if getattr(part, "tool_name", None):
                    tool_calls += 1
        self.metadata["agent"] = {
            "signed_in": signed_in,
            "output_chars": len(output or ""),
            "new_message_count": len(new_messages or []),
            "tool_call_count": tool_calls,
        }

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
                logger.info("VOICE_TRACE_SUMMARY %s", json.dumps(summary, ensure_ascii=False, default=str))
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
