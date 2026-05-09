import os
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Track if any OTEL exporter is configured
has_otel_exporter = False

# Conditionally configure Langfuse if env vars are set
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_client = None

if langfuse_public_key and langfuse_secret_key:
    try:
        from voice.app.config import settings

        # Labels for Langfuse: identify all traces as from this service
        release = (
            os.getenv("LANGFUSE_RELEASE")
            or settings.langfuse_release
            or "voice-oan-api"
        )
        environment = (
            os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
            or settings.langfuse_environment
            or settings.environment
            or "voice-development"
        )
        # SDK v3 uses LANGFUSE_HOST env var, but we also support LANGFUSE_BASE_URL
        host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or (settings.langfuse_base_url if settings.langfuse_base_url else None)
            or "https://cloud.langfuse.com"
        )

        # Set environment variables for SDK v3 auto-configuration
        os.environ.setdefault("LANGFUSE_HOST", host)

        print(f"🔍 Langfuse initializing: host={host}, release={release}, environment={environment}", flush=True)

        from langfuse import get_client

        langfuse_client = get_client()

        # Verify connection, but never fail app startup if the host is unreachable.
        try:
            if langfuse_client.auth_check():
                print("✅ Langfuse initialized successfully - authentication verified", flush=True)
                has_otel_exporter = True
            else:
                print("❌ Langfuse authentication failed - traces will not be sent", flush=True)
        except Exception as exc:
            print(f"ℹ️  Langfuse auth check skipped ({exc})", flush=True)
    except ImportError as e:
        print(f"ℹ️  Langfuse package not available - tracing disabled ({e})", flush=True)
else:
    print("ℹ️  Langfuse not configured - LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set", flush=True)

# Enable Pydantic AI instrumentation if at least one exporter is configured
if has_otel_exporter:
    from pydantic_ai.agent import Agent
    Agent.instrument_all()
    print("📊 Pydantic AI instrumentation enabled", flush=True)


@contextmanager
def start_observation(
    name: str,
    *,
    input: Any | None = None,
    output: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "generation",
) -> Iterator[Any | None]:
    """Start a Langfuse observation when the client is configured."""
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
