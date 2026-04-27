"""Scheduler bootstrap for union scheme refresh tasks."""

from __future__ import annotations

import asyncio
from zoneinfo import ZoneInfo

from helpers.utils import get_logger

from app.config import settings
from app.services.scheme_ingestion import (
    get_scheme_sources,
    refresh_all_scheme_sources,
    refresh_scheme_source,
    source_cache_exists,
)

logger = get_logger(__name__)

_scheme_scheduler = None


def get_scheme_scheduler():
    return _scheme_scheduler


def _create_scheduler():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ModuleNotFoundError:
        logger.exception("APScheduler dependency is unavailable for scheme scheduler")
        raise

    logger.info("Creating scheme scheduler timezone=%s", settings.timezone)
    scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.timezone))
    scheduler.add_job(
        refresh_all_scheme_sources,
        trigger=CronTrigger(hour=0, minute=0, second=0, timezone=ZoneInfo(settings.timezone)),
        id="refresh_milk_producer_schemes",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    logger.info("Registered scheme scheduler job id=refresh_milk_producer_schemes cron=00:00:00 timezone=%s", settings.timezone)
    return scheduler


async def schedule_startup_scheme_refreshes(create_task_fn=asyncio.create_task) -> list[str]:
    queued_sources: list[str] = []
    logger.info("Checking startup scheme refresh requirements")
    for source in get_scheme_sources():
        logger.info("Evaluating startup scheme refresh source=%s", source.cache_key)
        if await source_cache_exists(source.cache_key):
            logger.info("Skipping startup scheme refresh because cache already exists source=%s", source.cache_key)
            continue
        create_task_fn(refresh_scheme_source(source))
        queued_sources.append(source.cache_key)
        logger.info("Queued startup scheme refresh source=%s", source.cache_key)
    logger.info("Completed startup scheme refresh evaluation queued_count=%s", len(queued_sources))
    return queued_sources


async def start_scheme_scheduler() -> None:
    global _scheme_scheduler
    logger.info("Starting scheme scheduler")
    if _scheme_scheduler is None:
        try:
            _scheme_scheduler = _create_scheduler()
            _scheme_scheduler.start()
            logger.info("Scheme scheduler started")
        except ModuleNotFoundError:
            logger.exception("Scheme scheduler could not start because APScheduler is unavailable")
            _scheme_scheduler = None
            return
        except Exception:
            logger.exception("Scheme scheduler failed during startup")
            _scheme_scheduler = None
            return
    else:
        logger.info("Scheme scheduler already initialized")

    try:
        queued_sources = await schedule_startup_scheme_refreshes()
    except Exception:
        logger.exception("Failed while scheduling startup scheme refreshes")
        return
    if queued_sources:
        logger.info("Queued startup scheme refreshes for sources=%s", queued_sources)
    else:
        logger.info("No startup scheme refreshes were needed")


async def stop_scheme_scheduler() -> None:
    global _scheme_scheduler
    logger.info("Stopping scheme scheduler")
    if _scheme_scheduler is None:
        logger.info("Scheme scheduler stop skipped because scheduler is not running")
        return
    try:
        _scheme_scheduler.shutdown(wait=False)
    except Exception:
        logger.exception("Scheme scheduler failed during shutdown")
    finally:
        _scheme_scheduler = None
    logger.info("Scheme scheduler stopped")
