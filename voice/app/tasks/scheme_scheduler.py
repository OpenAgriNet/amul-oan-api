"""Scheduler bootstrap for union scheme refresh tasks."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from voice.app.config import settings
from voice.app.services.scheme_ingestion import (
    get_scheme_sources,
    refresh_all_scheme_sources,
    refresh_scheme_source,
    source_cache_exists,
)
from voice.helpers.utils import get_logger

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

    scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.timezone))
    scheduler.add_job(
        refresh_all_scheme_sources,
        trigger=CronTrigger(hour=0, minute=0, second=0, timezone=ZoneInfo(settings.timezone)),
        id="refresh_milk_producer_schemes",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    return scheduler


async def refresh_missing_scheme_sources() -> dict[str, bool]:
    refresh_results: dict[str, bool] = {}
    for source in get_scheme_sources():
        if await source_cache_exists(source.cache_key):
            logger.info("Startup scheme refresh skipped because cache exists source=%s", source.cache_key)
            continue

        logger.info("Startup scheme cache missing; running refresh source=%s", source.cache_key)
        refresh_results[source.cache_key] = await refresh_scheme_source(source)
        if refresh_results[source.cache_key]:
            logger.info("Startup scheme refresh completed source=%s", source.cache_key)
        else:
            logger.warning("Startup scheme refresh did not populate cache source=%s", source.cache_key)
    return refresh_results


async def schedule_startup_scheme_refreshes() -> list[str]:
    """Backward-compatible wrapper returning sources that were refreshed successfully."""
    results = await refresh_missing_scheme_sources()
    return [source_key for source_key, refreshed in results.items() if refreshed]


async def start_scheme_scheduler() -> None:
    global _scheme_scheduler
    if _scheme_scheduler is None:
        try:
            _scheme_scheduler = _create_scheduler()
            _scheme_scheduler.start()
        except ModuleNotFoundError:
            _scheme_scheduler = None
            return
        except Exception:
            logger.exception("Scheme scheduler failed during startup")
            _scheme_scheduler = None
            return

    try:
        refresh_results = await refresh_missing_scheme_sources()
    except Exception:
        logger.exception("Failed while scheduling startup scheme refreshes")
        return
    if refresh_results:
        logger.info("Startup scheme refresh results=%s", refresh_results)
    else:
        logger.info("No startup scheme refreshes were needed")


async def stop_scheme_scheduler() -> None:
    global _scheme_scheduler
    if _scheme_scheduler is None:
        return
    try:
        _scheme_scheduler.shutdown(wait=False)
    except Exception:
        logger.exception("Scheme scheduler failed during shutdown")
    finally:
        _scheme_scheduler = None
