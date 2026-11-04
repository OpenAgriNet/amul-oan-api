"""Background refresher for the verified cattle listings snapshot in Redis.

Every pod runs one; each tick is a cheap Redis check, and the shared NX lock in
agents.tools.cattle_listings_cache makes sure only one pod calls the provider.
It refreshes only while the listings tool was used within
CATTLE_LISTINGS_IDLE_SECONDS (see refresh_if_in_use).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from helpers.utils import get_logger

logger = get_logger(__name__)

TICK_SECONDS = 60.0

_worker_task: Optional[asyncio.Task] = None


async def _run_loop() -> None:
    # Imported here so importing this module stays side-effect-free (see
    # farmer_refresh_worker for the import-cycle reason).
    from agents.tools.cattle_listings_cache import refresh_if_in_use

    logger.info("Cattle listings refresher started")
    while True:
        try:
            await refresh_if_in_use()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Cattle listings refresh tick failed: %s", type(exc).__name__)
        await asyncio.sleep(TICK_SECONDS)


async def start_cattle_listings_refresher() -> None:
    global _worker_task
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run_loop())


async def stop_cattle_listings_refresher() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    finally:
        _worker_task = None
        logger.info("Cattle listings refresher stopped")
