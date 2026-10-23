import logging
import re
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from app.config import settings
from app.services.telemetry_stamps import chat_telemetry_release

logger = logging.getLogger(__name__)

# Track if any OTEL exporter is configured.
has_otel_exporter = False

# The trace id the next root span opens with, inside ``root_trace_id``.
_root_trace_id: ContextVar[Optional[int]] = ContextVar("root_trace_id", default=None)
_TRACE_ID = re.compile(r"[0-9a-f]{32}")


@contextmanager
def root_trace_id(trace_id: Optional[str]) -> Iterator[None]:
    """Open the root span started inside with ``trace_id`` (32 lowercase hex).

    A call turn's trace id is made when its request claims the session, before the
    turn opens, so the request it takes over from can name it. OpenTelemetry asks the
    id generator for a trace id only for a span with no parent, so the root takes it
    and every span under it shares it; enter this around opening the root alone.
    Passing Langfuse ``trace_context`` instead would give the root a parent that is
    never sent, and Langfuse would keep it as the root's parent observation. Any other
    value leaves the id to OpenTelemetry.
    """
    if not trace_id or not _TRACE_ID.fullmatch(trace_id):
        yield
        return
    token = _root_trace_id.set(int(trace_id, 16))
    try:
        yield
    finally:
        _root_trace_id.reset(token)


def _trace_ids():
    """OpenTelemetry's random ids, but a root span opened inside ``root_trace_id`` gets its id."""
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    class _TraceIds(RandomIdGenerator):
        def generate_trace_id(self) -> int:
            pinned = _root_trace_id.get()
            return pinned if pinned is not None else super().generate_trace_id()

    return _TraceIds()


def _process_environment() -> str:
    """The Langfuse environment of anything not inside a surface's turn."""
    return (
        settings.langfuse_tracing_environment
        or settings.environment
        or "voice-development"
    )


def surface_environment(surface: str) -> str:
    """The Langfuse environment of one surface's turns ("chat" or "voice").

    One process can serve both surfaces, so the surface, not the process, says
    where a turn's trace lands: its pipeline settings' langfuse_environment, else
    the process environment. The turn roots pass this to
    ``propagate_attributes(environment=...)``, which stamps it on every span
    opened inside the turn, pydantic-ai's included, and on the scores made there.
    """
    from app.llm_core import runtime

    return runtime.surface_settings(surface).langfuse_environment or _process_environment()


# Conditionally configure Langfuse if env vars are set.
langfuse_public_key = settings.langfuse_public_key
langfuse_secret_key = settings.langfuse_secret_key
langfuse_client = None

if langfuse_public_key and langfuse_secret_key:
    try:
        # Labels for Langfuse. The release is the build that served the trace:
        # LANGFUSE_RELEASE if the deployment sets it, else the running commit,
        # the same value as the amul.release stamp.
        release = settings.langfuse_release or chat_telemetry_release()
        environment = _process_environment()
        # Langfuse SDK v4 reads LANGFUSE_BASE_URL. Keep LANGFUSE_HOST as a
        # backward-compatible input because older deployments may still set it.
        host = (
            settings.langfuse_base_url
            or settings.langfuse_host
            or "https://cloud.langfuse.com"
        )

        print(
            f"Langfuse initializing: host={host}, release={release}, environment={environment}",
            flush=True,
        )

        from langfuse import Langfuse

        langfuse_client = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=host,
            release=release,
            environment=environment,
            # Langfuse uses it when it sets up OpenTelemetry, which it does here, first;
            # it logs a warning if a tracer provider was already registered.
            id_generator=_trace_ids(),
        )

        # Verify connection before enabling pydantic-ai OTEL instrumentation.
        if langfuse_client.auth_check():
            print("Langfuse initialized successfully - authentication verified", flush=True)
            has_otel_exporter = True
        else:
            print("Langfuse authentication failed - traces will not be sent", flush=True)
    except ImportError as e:
        print(f"Langfuse package not available - tracing disabled ({e})", flush=True)
else:
    print(
        "Langfuse not configured - LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set",
        flush=True,
    )

# Enable Pydantic AI instrumentation if at least one exporter is configured.
if has_otel_exporter:
    from pydantic_ai.agent import Agent

    Agent.instrument_all()
    print("Pydantic AI instrumentation enabled", flush=True)


def get_langfuse_client():
    """Return the configured Langfuse client, or None when tracing is disabled."""
    return langfuse_client


def set_trace_io(
    *, input: Any | None = None, output: Any | None = None
) -> None:
    """Set the current Langfuse TURN's input and/or output.

    ⚠️ CALL THIS ONLY FROM THE ORCHESTRATOR, AT TURN LEVEL. Never from inside a
    tool, a backend call, or any nested observation.

    "Trace" here means the whole turn — the farmer's question and the answer they
    received. A tool calling this does not annotate itself; it OVERWRITES the
    turn's question and answer with its own parameters and result.

    This was live in production for five days (#212). The booking tools called it
    from inside tool execution, so a successful AI-insemination booking recorded
    ``trace_input`` as ``{"union_code": ..., "farmer_code": ...}`` and
    ``trace_output`` as ``{"success": true, "ticket_number": ...}`` — while the
    farmer had asked "ગાય" and been answered in Gujarati. ~140 turns/day, and
    100% of bookings: exactly the records consulted when a booking is disputed.

    It went unnoticed for a reason worth remembering: before the turn root span
    (#200) there was no active span, so these writes were silently DISCARDED.
    #200 made them land. Fixing the drop created the clobber.

    To annotate a tool, update its own observation instead — see
    ``start_observation`` below, whose ``update(output=...)`` is where tool
    input/output belongs.

    No-op when tracing is disabled, and never raises — tracing must not break the
    caller.
    """
    if langfuse_client is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if kwargs:
            langfuse_client.set_current_trace_io(**kwargs)
    except Exception:
        pass


@contextmanager
def non_turn_root(
    kind: str,
    metadata: dict[str, str],
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    environment: str | None = None,
) -> Iterator[Any | None]:
    """Open the root of a trace that is not a turn: a background job or a frontend
    event, named after its kind (amul-telemetry telemetry/turn.v1.yaml).

    As on a turn's root, the trace-level fields (the amul.* metadata, the session, the
    user) are written on the root only: ``propagate_attributes`` is entered and left
    before anything opens under it, so the model or API calls inside carry none. The
    root has no input or output; turn.v1 keeps those for turns, so what the job read
    and wrote belongs on the observation that did the work.

    ``environment`` is the Langfuse environment of every span inside, for work that
    belongs to a surface but runs outside its turn; without it the process's applies.
    Yields None when tracing is disabled.
    """
    if langfuse_client is None:
        yield None
        return

    from langfuse import propagate_attributes

    with ExitStack() as stack:
        if environment:
            stack.enter_context(propagate_attributes(environment=environment))
        root = stack.enter_context(
            langfuse_client.start_as_current_observation(name=kind, as_type="span", metadata=metadata)
        )
        with propagate_attributes(
            session_id=session_id[:200] if session_id else None,
            user_id=user_id[:200] if user_id else None,
            metadata=metadata,
        ):
            pass
        yield root


@contextmanager
def start_observation(
    name: str,
    *,
    input: Any | None = None,
    output: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "span",
) -> Iterator[Any | None]:
    """Start a Langfuse observation when the client is configured.

    Generic helpers default to a span. Callers should pass as_type="generation"
    only for model calls so Langfuse renders model latency and token metadata
    correctly.
    """
    if langfuse_client is None:
        yield None
        return

    with langfuse_client.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input,
        output=output,
        model=model,
        metadata=metadata or {},
    ) as observation:
        yield observation
