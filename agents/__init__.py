import os
from dotenv import load_dotenv

load_dotenv()

# Track if any OTEL exporter is configured
has_otel_exporter = False

# Conditionally configure Logfire if token is set
if os.getenv("LOGFIRE_TOKEN"):
    import logfire
    logfire.configure(scrubbing=False)
    has_otel_exporter = True

# Conditionally configure Langfuse if env vars are set
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
if langfuse_public_key and langfuse_secret_key:
    from langfuse import get_client
    # Initialize Langfuse client - this registers the OTEL span processor
    get_client()
    has_otel_exporter = True

# Enable Pydantic AI instrumentation if at least one exporter is configured
if has_otel_exporter:
    from pydantic_ai.agent import Agent
    Agent.instrument_all()