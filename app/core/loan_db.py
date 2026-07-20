"""Async Postgres engine/session for the micro-loan feature.

Lazy by design: the engine is only built on first use, and only when
``LOAN_DB_URL`` is set. Services that never touch the loan flow (or run without
the DB configured) are unaffected — importing this module has no side effects.
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
from app.models.loan import Base
from helpers.utils import get_logger

logger = get_logger(__name__)

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker] = None


def loan_db_configured() -> bool:
    """True when a loan DB URL is present. Callers should gate on this before
    attempting any persistence and degrade gracefully when it is False."""
    return bool(settings.loan_db_url)


def _get_sessionmaker() -> async_sessionmaker:
    global _engine, _sessionmaker
    if _sessionmaker is not None:
        return _sessionmaker
    if not settings.loan_db_url:
        raise RuntimeError("LOAN_DB_URL is not configured")
    _engine = create_async_engine(
        settings.loan_db_url,
        pool_size=settings.loan_db_pool_size,
        pool_pre_ping=True,
        future=True,
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    logger.info("Loan DB engine initialized (pool_size=%s)", settings.loan_db_pool_size)
    return _sessionmaker


@contextlib.asynccontextmanager
async def get_loan_session() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding a session; rolls back on error."""
    sm = _get_sessionmaker()
    session = sm()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_loan_tables() -> None:
    """Create loan tables if missing (idempotent).

    Ops normally applies ``migrations/loan/001_init.sql`` directly against each
    env's Postgres; this helper exists so local/dev bring-up is a single call.
    No-op when the DB is not configured.
    """
    if not settings.loan_db_url:
        logger.info("init_loan_tables skipped: LOAN_DB_URL not configured")
        return
    _get_sessionmaker()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Loan tables ensured (create_all)")
