"""The debug view: what a chat turn did, step by step, for the dev UI's Debug panel.

On a development deployment with DEBUG_VIEW_ENABLED on, every chat turn's OpenTelemetry spans are
kept for a day under the turn's trace id (the chat response's X-Trace-Id). They are the spans
Langfuse gets: the turn, moderation, translation, the agent, its model calls and its tools. Only
the user who asked can read them back (GET /api/debug/turns/{trace_id}, app/routers/debug.py).

Kept small, and kept away from anyone else:
- The system prompt and the conversation so far are left out (Langfuse has them, behind its
  sign-in), and so are pydantic-ai's tool definitions and bookkeeping.
- Attributes named like a secret or a user id are dropped, and token-like values are masked.
- Long values are cut, JSON values pruned so a tool's result stays readable, and a turn keeps
  at most MAX_SPANS spans.

Nothing here can fail a chat turn: capture errors are logged and dropped, and Redis is written
from a thread of its own, never from the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode

from app.config import settings

logger = logging.getLogger(__name__)

TTL_SECONDS = 24 * 3600
# How long after its request a turn's spans are still taken: the suggestions run after the response.
WATCH_SECONDS = 15 * 60
MAX_SPANS = 300
MAX_VALUE_CHARS = 4000
MAX_STRING_IN_JSON = 600
MAX_LIST_ITEMS = 50
MAX_SPAN_CHARS = 24_000

# Left out on purpose: the system prompt and the conversation so far (the Langfuse trace has them),
# and pydantic-ai's tool definitions and bookkeeping, which repeat on every model call.
DROPPED_ATTRIBUTES = frozenset(
    {
        "gen_ai.system_instructions",
        "gen_ai.input.messages",
        "pydantic_ai.all_messages",
        "gen_ai.tool.definitions",
        "model_request_parameters",
        "logfire.json_schema",
    }
)
_SECRET_WORDS = frozenset(
    {
        "authorization",
        "apikey",
        "api_key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "secret",
        "password",
        "passwd",
        "cookie",
        "credential",
        "credentials",
    }
)
_SECRET_SUFFIXES = ("_secret", "_password", "_api_key", "_apikey", "_token")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")


def active() -> bool:
    """On only where it is asked for and the deployment says it is development (dev, previews).

    ENVIRONMENT defaults to production, so a deployment that sets nothing never has it.
    """
    return settings.debug_view_enabled and settings.environment == "development"


def is_hidden_key(key: str) -> bool:
    """Whether an attribute (or a JSON key inside one) is left out: a secret by its name, or a user id.

    Usage counts such as gen_ai.usage.input_tokens are kept: no part of them is a secret word.
    """
    parts = [part.lower() for part in re.split(r"[.\s]", key) if part]
    if not parts:
        return False
    if parts[-2:] == ["user", "id"] or parts[-1] in {"user_id", "userid"}:
        return True
    return any(part in _SECRET_WORDS or part.endswith(_SECRET_SUFFIXES) for part in parts)


def _mask(text: str) -> str:
    return _BEARER.sub("Bearer [masked]", _JWT.sub("[masked token]", text))


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


def _prune(value: Any, max_string: int, max_items: int, depth: int = 0) -> Any:
    """A JSON value cut down so it still parses: long strings and lists shortened, hidden keys dropped."""
    if depth > 8:
        return "…"
    if isinstance(value, str):
        return _cut(_mask(value), max_string)
    if isinstance(value, list):
        items = [_prune(item, max_string, max_items, depth + 1) for item in value[:max_items]]
        if len(value) > max_items:
            items.append(f"… (+{len(value) - max_items} more)")
        return items
    if isinstance(value, dict):
        return {
            str(k): _prune(v, max_string, max_items, depth + 1)
            for k, v in value.items()
            if not is_hidden_key(str(k))
        }
    return value


def _pruned_json(text: str) -> Optional[str]:
    """``text`` as JSON pruned to fit MAX_VALUE_CHARS (smaller strings and lists until it does), else None."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    pruned = ""
    for max_string, max_items in ((MAX_STRING_IN_JSON, MAX_LIST_ITEMS), (200, 20), (80, 10)):
        pruned = json.dumps(_prune(value, max_string, max_items), ensure_ascii=False)
        if len(pruned) <= MAX_VALUE_CHARS:
            return pruned
    return _cut(pruned, MAX_VALUE_CHARS)


def clean_value(value: Any) -> Any:
    """An attribute value as kept: masked and cut, and a JSON string pruned so it stays JSON."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = [clean_value(item) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"… (+{len(value) - MAX_LIST_ITEMS} more)")
        return items
    text = value if isinstance(value, str) else str(value)
    if text.lstrip()[:1] in ("{", "["):
        pruned = _pruned_json(text)
        if pruned is not None:
            return pruned
    return _cut(_mask(text), MAX_VALUE_CHARS)


def _error_of(span: ReadableSpan, attributes: dict) -> Optional[str]:
    if attributes.get("langfuse.observation.level") == "ERROR":
        return str(attributes.get("langfuse.observation.status_message") or "error")
    if span.status.status_code is not StatusCode.ERROR:
        return None
    for event in span.events:
        if event.name == "exception":
            kind = (event.attributes or {}).get("exception.type", "")
            message = (event.attributes or {}).get("exception.message", "")
            return _cut(_mask(f"{kind}: {message}".strip(": ")), 1000) or "error"
    return _cut(_mask(span.status.description or "error"), 1000)


def compact(span: ReadableSpan) -> dict:
    """One span as the debug view keeps it."""
    attributes: dict[str, Any] = {}
    for key, value in (span.attributes or {}).items():
        if key in DROPPED_ATTRIBUTES or is_hidden_key(key):
            continue
        attributes[key] = clean_value(value)
    cut = False
    if len(json.dumps(attributes, ensure_ascii=False)) > MAX_SPAN_CHARS:
        # Largest first, down to 1000 characters each, until the span fits.
        for key in sorted(attributes, key=lambda k: len(json.dumps(attributes[k], ensure_ascii=False)), reverse=True):
            if isinstance(attributes[key], str):
                attributes[key] = _cut(attributes[key], 1000)
                cut = True
            if len(json.dumps(attributes, ensure_ascii=False)) <= MAX_SPAN_CHARS:
                break
    error = _error_of(span, attributes)
    record = {
        "id": format(span.context.span_id, "016x"),
        "parent": format(span.parent.span_id, "016x") if span.parent else None,
        "name": span.name,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": "error" if error else "ok",
        "attributes": attributes,
    }
    if error:
        record["error"] = error
    if cut:
        record["cut"] = True
    return record


class _Watched:
    """The trace ids of the turns this process is taking spans for, each with its span count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, list] = {}

    def add(self, trace_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            for old in [t for t, (until, _) in self._turns.items() if until < now]:
                del self._turns[old]
            self._turns[trace_id] = [now + WATCH_SECONDS, 0]

    def count(self, trace_id: str) -> Optional[int]:
        """The span's number in its turn (1, 2, ...), or None when the turn is not watched."""
        with self._lock:
            entry = self._turns.get(trace_id)
            if entry is None or entry[0] < time.monotonic():
                return None
            entry[1] += 1
            return entry[1]


watched = _Watched()

# sink(trace_id, record, ends_turn): where a kept span goes.
Sink = Callable[[str, dict, bool], None]


class DebugSpanProcessor(SpanProcessor):
    """Keeps the spans of watched turns, cut down, through ``sink``. Never raises into the tracer."""

    def __init__(self, sink: Sink, turns: _Watched = watched) -> None:
        self._sink = sink
        self._turns = turns
        self._warned_at = 0.0

    def on_end(self, span: ReadableSpan) -> None:
        try:
            trace_id = format(span.context.trace_id, "032x")
            number = self._turns.count(trace_id)
            if number is None:
                return
            if number > MAX_SPANS:
                if number == MAX_SPANS + 1:
                    self._sink(trace_id, {"name": f"(only the first {MAX_SPANS} spans are kept)", "dropped": True}, False)
                return
            self._sink(trace_id, compact(span), span.parent is None)
        except Exception:  # never let the debug view break tracing
            now = time.monotonic()
            if now - self._warned_at > 60:
                self._warned_at = now
                logger.exception("debug view: a span could not be kept")


def _keys(trace_id: str) -> tuple[str, str, str]:
    from app.core.cache import build_cache_key

    return (
        build_cache_key(trace_id, "debug-view:turn"),
        build_cache_key(trace_id, "debug-view:spans"),
        build_cache_key(trace_id, "debug-view:done"),
    )


class RedisSink:
    """Writes kept spans to Redis from a thread of its own, in batches; drops them if Redis is slow."""

    def __init__(self, client: Any = None) -> None:
        self._client = client
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._warned_at = 0.0
        threading.Thread(target=self._run, name="debug-view-writer", daemon=True).start()

    def __call__(self, trace_id: str, record: dict, ends_turn: bool) -> None:
        try:
            self._queue.put_nowait((trace_id, record, ends_turn))
        except queue.Full:
            self._warn("the write queue is full, a span was dropped")

    def _connect(self) -> Any:
        import redis

        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            username=settings.redis_username,
            password=settings.redis_password,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            socket_timeout=settings.redis_socket_timeout,
            decode_responses=True,
        )
        return self._client

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < 200:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                self.write(batch)
            except Exception as error:
                self._warn(f"{len(batch)} spans could not be written to Redis: {error}")

    def write(self, batch: list) -> None:
        client = self._client or self._connect()
        pipe = client.pipeline(transaction=False)
        for trace_id, record, ends_turn in batch:
            _, spans_key, done_key = _keys(trace_id)
            pipe.rpush(spans_key, json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            pipe.expire(spans_key, TTL_SECONDS)
            if ends_turn:
                pipe.set(done_key, "1", ex=TTL_SECONDS)
        pipe.execute()

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._warned_at > 60:
            self._warned_at = now
            logger.warning("debug view: %s", message)


_installed = False


def install(provider: Any = None, sink: Optional[Sink] = None) -> bool:
    """Adds the span processor to the tracer provider. Off, or with tracing off: does nothing."""
    global _installed
    if _installed or not active():
        return _installed
    if provider is None:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        logger.warning("debug view: tracing is off (no Langfuse), so there are no turns to show")
        return False
    provider.add_span_processor(DebugSpanProcessor(sink or RedisSink()))
    _installed = True
    logger.info("debug view: on, chat turns are kept for %s hours", TTL_SECONDS // 3600)
    return True


def owner_of(user_info: Optional[dict]) -> Optional[str]:
    """Whose turns these are: the JWT's sub. Server-side callers (X-API-Key, WhatsApp) have none."""
    if not user_info or user_info.get("auth_type") == "api_key":
        return None
    sub = user_info.get("sub")
    return sub if isinstance(sub, str) and sub else None


def _client(client: Any) -> Any:
    if client is not None:
        return client
    from app.core.cache import redis_client

    return redis_client


async def register_turn(
    trace_id: str,
    *,
    owner: Optional[str],
    session_id: str,
    client: Any = None,
) -> None:
    """Watches a chat turn about to start, for its owner (the JWT's sub). Never raises."""
    if not active() or not owner:
        return
    try:
        from app.services.telemetry_stamps import chat_telemetry_release

        watched.add(trace_id)
        meta = {
            "owner": owner,
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": {
                "environment": settings.langfuse_tracing_environment or settings.environment,
                "release": chat_telemetry_release(),
            },
        }
        meta_key, _, _ = _keys(trace_id)
        # A slow Redis may cost the turn a second at most, never the answer.
        await asyncio.wait_for(_client(client).set(meta_key, json.dumps(meta), ex=TTL_SECONDS), timeout=1.0)
    except Exception:
        logger.exception("debug view: turn %s could not be registered", trace_id)


async def read_turn(trace_id: str, owner: Optional[str], client: Any = None) -> Optional[dict]:
    """The turn as kept, or None when there is none for this owner (another user's reads as none)."""
    if not owner:
        return None
    redis = _client(client)
    meta_key, spans_key, done_key = _keys(trace_id)
    raw = await redis.get(meta_key)
    if not raw:
        return None
    meta = json.loads(raw)
    if meta.get("owner") != owner:
        return None
    spans = [json.loads(item) for item in await redis.lrange(spans_key, 0, -1)]
    return {
        "trace_id": trace_id,
        "session_id": meta.get("session_id"),
        "started_at": meta.get("started_at"),
        "backend": meta.get("backend", {}),
        "complete": bool(await redis.exists(done_key)),
        "spans": spans,
    }
