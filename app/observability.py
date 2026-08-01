import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Track if any OTEL exporter is configured.
has_otel_exporter = False

# Conditionally configure Langfuse if env vars are set.
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_client = None

if langfuse_public_key and langfuse_secret_key:
    try:
        from app.config import settings

        # Labels for Langfuse: identify all traces as from this service.
        release = (
            os.getenv("LANGFUSE_RELEASE")
            or settings.langfuse_release
            or "voice-oan-api"
        )
        environment = (
            os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
            or settings.langfuse_tracing_environment
            or settings.environment
            or "voice-development"
        )
        # Langfuse SDK v4 reads LANGFUSE_BASE_URL. Keep LANGFUSE_HOST as a
        # backward-compatible input because older deployments may still set it.
        host = (
            os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or (settings.langfuse_base_url if settings.langfuse_base_url else None)
            or "https://cloud.langfuse.com"
        )

        os.environ.setdefault("LANGFUSE_BASE_URL", host)

        print(
            f"Langfuse initializing: host={host}, release={release}, environment={environment}",
            flush=True,
        )

        from langfuse import get_client

        langfuse_client = get_client()

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
    """Set the current Langfuse trace's input and/or output.

    Centralized wrapper around set_current_trace_io so tools/backends don't reach
    for the langfuse client directly. No-op when tracing is disabled, and never
    raises — tracing must not break the caller.
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
