"""Async Postgres engine/session for partner webhook storage.

Lazy initialization: the engine is created only when first used and only when
``WEBHOOK_DB_URL`` is configured.
"""
from __future__ import annotations

import contextlib
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.models.webhook import WebhookBase
from helpers.utils import get_logger

logger = get_logger(__name__)

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker] = None


def webhook_db_configured() -> bool:
    """True when webhook persistence DB URL is configured."""
    return bool(settings.webhook_db_url)


def _get_sessionmaker() -> async_sessionmaker:
    global _engine, _sessionmaker
    if _sessionmaker is not None:
        return _sessionmaker
    if not settings.webhook_db_url:
        raise RuntimeError("WEBHOOK_DB_URL is not configured")

    _engine = create_async_engine(
        settings.webhook_db_url,
        pool_size=settings.webhook_db_pool_size,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=10,
        connect_args={"timeout": 10, "command_timeout": 10},
        # A DB error's text otherwise carries the bound values (farmer_contact,
        # payload_raw), and the router logs that text.
        hide_parameters=True,
        future=True,
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    logger.info(
        "Webhook DB engine initialized (pool_size=%s)",
        settings.webhook_db_pool_size,
    )
    return _sessionmaker


@contextlib.asynccontextmanager
async def get_webhook_session() -> AsyncIterator[AsyncSession]:
    """Yield a webhook DB session and rollback on errors."""
    sm = _get_sessionmaker()
    session = sm()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_webhook_tables() -> None:
    """Create webhook tables if missing (idempotent)."""
    if not settings.webhook_db_url:
        logger.info("init_webhook_tables skipped: WEBHOOK_DB_URL not configured")
        return
    _get_sessionmaker()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(WebhookBase.metadata.create_all)
    logger.info("Webhook tables ensured (create_all)")
