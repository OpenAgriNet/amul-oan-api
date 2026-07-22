"""Background worker that drains the farmer-data refresh queue.

Stale-while-revalidate: the request path serves cached farmer data immediately
and enqueues stale phones (see agents.services.farmer_cache.
enqueue_farmer_refresh). This worker drains that Redis-backed queue off the
request path and refreshes each record, so slow/unreliable upstream PashuGPT
APIs never block a call. refresh_farmer_data self-dedupes via its NX lock, so
running one worker per pod is safe.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from helpers.utils import get_logger

logger = get_logger(__name__)

# Seconds to wait before re-polling when the queue is empty.
IDLE_SLEEP_SECONDS = 5.0
# Phones to refresh per drain iteration.
DRAIN_BATCH = 20

_worker_task: Optional[asyncio.Task] = None


async def _run_loop() -> None:
    # Imported here (not at module scope) so importing this worker module is
    # side-effect-free and never triggers the farmer_cache <-> tools import
    # cycle at app startup (main.py imports the worker before the routers).
    from agents.services.farmer_cache import drain_farmer_refresh_queue_once

    logger.info("Farmer refresh worker started")
    while True:
        try:
            processed = await drain_farmer_refresh_queue_once(batch=DRAIN_BATCH)
            if processed == 0:
                await asyncio.sleep(IDLE_SLEEP_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Farmer refresh worker iteration failed")
            await asyncio.sleep(IDLE_SLEEP_SECONDS)


async def start_farmer_refresh_worker() -> None:
    global _worker_task
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run_loop())


async def stop_farmer_refresh_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Farmer refresh worker failed during shutdown")
    finally:
        _worker_task = None
        logger.info("Farmer refresh worker stopped")
