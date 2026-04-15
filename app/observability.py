"""Langfuse client + Pydantic AI instrumentation (lazy, idempotent)."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_initialized = False

# Set by init_observability(); default None until init (lazy import paths see safe defaults)
langfuse_client = None
has_otel_exporter = False


def init_observability() -> None:
    """Initialize Langfuse and Pydantic AI tracing once per process."""
    global _initialized, langfuse_client, has_otel_exporter

    if _initialized:
        return

    _initialized = True
    # Allow local `python -m app.observability` runs to pick up .env values.
    load_dotenv()

    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not langfuse_public_key or not langfuse_secret_key:
        msg = (
            "Observability disabled: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set"
        )
        logger.info(msg)
        print(f"ℹ️  {msg}", flush=True)
        return

    from app.config import settings

    release = (
        os.getenv("LANGFUSE_RELEASE")
        or (settings.langfuse_release or None)
        or "amul-oan-api"
    )
    environment = (
        os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
        or (settings.langfuse_tracing_environment or None)
        or (settings.environment or None)
        or "development"
    )
    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or (settings.langfuse_base_url if settings.langfuse_base_url else None)
        or "https://cloud.langfuse.com"
    )
    os.environ.setdefault("LANGFUSE_HOST", host)

    logger.info(
        "Langfuse initializing: host=%s, release=%s, environment=%s",
        host,
        release,
        environment,
    )
    print(
        f"🔍 Langfuse initializing: host={host}, release={release}, environment={environment}",
        flush=True,
    )

    from langfuse import get_client

    client = get_client()

    try:
        auth_ok = client.auth_check()
    except Exception as exc:
        warn_msg = f"Langfuse auth check failed ({exc}); traces will not be sent"
        logger.warning(warn_msg)
        print(f"❌ {warn_msg}", flush=True)
        langfuse_client = None
        has_otel_exporter = False
        return

    if auth_ok:
        ok_msg = "Langfuse initialized successfully (authentication verified)"
        logger.info(ok_msg)
        print(f"✅ {ok_msg}", flush=True)
        langfuse_client = client
        has_otel_exporter = True
    else:
        warn_msg = "Langfuse authentication failed; traces will not be sent"
        logger.warning(warn_msg)
        print(f"❌ {warn_msg}", flush=True)
        langfuse_client = None
        has_otel_exporter = False

    if has_otel_exporter:
        from pydantic_ai.agent import Agent

        Agent.instrument_all()
        inst_msg = "Pydantic AI instrumentation enabled"
        logger.info(inst_msg)
        print(f"📊 {inst_msg}", flush=True)

if __name__ == "__main__":
    init_observability()