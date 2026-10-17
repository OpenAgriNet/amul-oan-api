"""Background worker that enforces webhook data retention.

Deletes rows older than ``WEBHOOK_RETENTION_HOURS`` from the dedicated webhook
Postgres table at a fixed cadence (``WEBHOOK_CLEANUP_INTERVAL_SECONDS``).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete

from app.config import settings
from app.core.webhook_db import get_webhook_session, webhook_db_configured
from app.models.webhook import HeatAlertWebhookEvent
from helpers.utils import get_logger

logger = get_logger(__name__)

_worker_task: Optional[asyncio.Task] = None


async def _cleanup_once() -> int:
    """Delete rows older than the configured retention horizon."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.webhook_retention_hours)
    stmt = delete(HeatAlertWebhookEvent).where(HeatAlertWebhookEvent.received_at < cutoff)
    async with get_webhook_session() as session:
        result = await session.execute(stmt)
        await session.commit()
    return int(result.rowcount or 0)


async def _run_loop() -> None:
    interval = max(5, settings.webhook_cleanup_interval_seconds)
    logger.info(
        "Webhook cleanup worker started (retention_hours=%s interval_seconds=%s)",
        settings.webhook_retention_hours,
        interval,
    )
    while True:
        try:
            deleted = await _cleanup_once()
            if deleted > 0:
                logger.info("Webhook cleanup deleted %s expired row(s)", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Webhook cleanup worker iteration failed")
        await asyncio.sleep(interval)


async def start_webhook_cleanup_worker() -> None:
    """Start retention worker only when webhook DB is configured."""
    global _worker_task
    if not webhook_db_configured():
        logger.info("Webhook cleanup worker not started: WEBHOOK_DB_URL not configured")
        return
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run_loop())


async def stop_webhook_cleanup_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Webhook cleanup worker failed during shutdown")
    finally:
        _worker_task = None
        logger.info("Webhook cleanup worker stopped")
