import os
from dotenv import load_dotenv

load_dotenv()

# Track if any OTEL exporter is configured
has_otel_exporter = False

# Conditionally configure Langfuse if env vars are set
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
if langfuse_public_key and langfuse_secret_key:
    from langfuse import Langfuse

    # Langfuse reads LANGFUSE_TRACING_ENVIRONMENT only from os.environ; if unset, the UI
    # shows "default". Pydantic settings are not wired here — use the explicit env var or
    # this repo default so chat traces consistently land under chat-development.
    _langfuse_tracing_environment = (
        os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or "chat-development"
    )
    Langfuse(environment=_langfuse_tracing_environment)
    has_otel_exporter = True

# Enable Pydantic AI instrumentation if at least one exporter is configured
if has_otel_exporter:
    from pydantic_ai.agent import Agent
    Agent.instrument_all()
