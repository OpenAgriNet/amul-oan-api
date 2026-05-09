from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from voice.helpers.utils import get_logger


logger = get_logger(__name__)

_SESSION_ID_VAR: ContextVar[str | None] = ContextVar("model_boundary_session_id", default=None)
_PROCESS_ID_VAR: ContextVar[str | None] = ContextVar("model_boundary_process_id", default=None)
_QUERY_VAR: ContextVar[str | None] = ContextVar("model_boundary_query", default=None)
_CALL_INDEX_VAR: ContextVar[int] = ContextVar("model_boundary_call_index", default=0)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def boundary_capture_enabled() -> bool:
    return _env_enabled("MODEL_BOUNDARY_CAPTURE_ENABLED", default=False)


def get_boundary_capture_dir() -> Path:
    return Path(os.getenv("MODEL_BOUNDARY_CAPTURE_DIR", "/tmp/voice_model_boundary")).resolve()


@contextmanager
def boundary_capture_context(
    *,
    session_id: str | None,
    process_id: str | None,
    user_query: str | None,
) -> Iterator[None]:
    session_token: Token[str | None] = _SESSION_ID_VAR.set(session_id)
    process_token: Token[str | None] = _PROCESS_ID_VAR.set(process_id)
    query_token: Token[str | None] = _QUERY_VAR.set(user_query)
    call_index_token: Token[int] = _CALL_INDEX_VAR.set(0)
    try:
        yield
    finally:
        _CALL_INDEX_VAR.reset(call_index_token)
        _QUERY_VAR.reset(query_token)
        _PROCESS_ID_VAR.reset(process_token)
        _SESSION_ID_VAR.reset(session_token)


def _next_call_index() -> int:
    current = _CALL_INDEX_VAR.get()
    next_index = current + 1
    _CALL_INDEX_VAR.set(next_index)
    return next_index


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def capture_model_boundary_payload(payload: dict[str, Any]) -> Path | None:
    if not boundary_capture_enabled():
        return None

    session_id = _SESSION_ID_VAR.get()
    if not session_id:
        return None

    call_index = _next_call_index()
    process_id = _PROCESS_ID_VAR.get()
    user_query = _QUERY_VAR.get()

    record = {
        "capture_type": "model_boundary",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "process_id": process_id,
        "query": user_query,
        "call_index": call_index,
        **_sanitize_payload(payload),
    }

    output_dir = get_boundary_capture_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}__{call_index:02d}.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Model boundary payload captured - session_id=%s process_id=%s call_index=%s path=%s",
        session_id,
        process_id,
        call_index,
        output_path,
    )
    return output_path
