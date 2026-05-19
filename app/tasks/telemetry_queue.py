import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.models.telemetry import CanonicalTelemetryEvent
from app.services.langfuse_telemetry_writer import write_canonical_event_to_langfuse
from helpers.utils import get_logger

logger = get_logger(__name__)


EnqueueStatus = Literal["enqueued", "dropped_queue_full", "not_started"]

_QUEUE_MAX_SIZE = settings.telemetry_queue_max_size
_MAX_RETRIES = settings.telemetry_queue_max_retries
_BASE_DELAY_MS = settings.telemetry_queue_retry_base_delay_ms
_MAX_DELAY_MS = settings.telemetry_queue_retry_max_delay_ms
_DEAD_LETTER_MAX = settings.telemetry_dead_letter_max


@dataclass
class DeadLetterRecord:
    event_name: str
    session_id: str | None
    question_id: str | None
    reason: str
    attempts: int


_queue: asyncio.Queue[CanonicalTelemetryEvent | None] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
_worker_task: asyncio.Task | None = None
_dead_letter: deque[DeadLetterRecord] = deque(maxlen=_DEAD_LETTER_MAX)
_stats = {
    "enqueued": 0,
    "dropped_queue_full": 0,
    "dropped_not_started": 0,
    "processed_ok": 0,
    "processed_skipped": 0,
    "processed_failed": 0,
}


def enqueue_canonical_telemetry_event(
    canonical: CanonicalTelemetryEvent,
) -> tuple[EnqueueStatus, str | None]:
    if _worker_task is None or _worker_task.done():
        _stats["dropped_not_started"] += 1
        return "not_started", "telemetry worker not running"

    try:
        _queue.put_nowait(canonical)
        _stats["enqueued"] += 1
        return "enqueued", None
    except asyncio.QueueFull:
        _stats["dropped_queue_full"] += 1
        return "dropped_queue_full", "telemetry queue is full"


async def start_telemetry_worker() -> None:
    global _worker_task
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_telemetry_worker_loop(), name="telemetry-worker")
    logger.info(
        "Telemetry worker started queue_max=%s retries=%s base_delay_ms=%s max_delay_ms=%s",
        _QUEUE_MAX_SIZE,
        _MAX_RETRIES,
        _BASE_DELAY_MS,
        _MAX_DELAY_MS,
    )


async def stop_telemetry_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return

    # Best-effort semantics: shutdown is intentionally lossy.
    # We enqueue a sentinel and stop the worker once it is observed; events queued
    # after the sentinel are not guaranteed to be processed before process exit.
    pending_before_stop = _queue.qsize()
    if pending_before_stop > 0:
        logger.warning(
            "Telemetry worker stopping with best-effort semantics pending_events=%s",
            pending_before_stop,
        )

    try:
        await _queue.put(None)
        await _worker_task
    finally:
        _worker_task = None
    logger.info("Telemetry worker stopped")


def get_telemetry_queue_stats() -> dict:
    return {
        **_stats,
        "queue_size": _queue.qsize(),
        "queue_max_size": _QUEUE_MAX_SIZE,
        "dead_letter_size": len(_dead_letter),
        "worker_running": _worker_task is not None and not _worker_task.done(),
    }


async def _telemetry_worker_loop() -> None:
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            break

        await _process_with_retries(item)
        _queue.task_done()


async def _process_with_retries(event: CanonicalTelemetryEvent) -> None:
    attempts = 0
    while True:
        attempts += 1
        status, reason = await write_canonical_event_to_langfuse(event)

        if status == "queued":
            _stats["processed_ok"] += 1
            return

        if status == "skipped_unconfigured":
            _stats["processed_skipped"] += 1
            logger.info(
                "Telemetry skipped event_name=%s sid=%s qid=%s reason=%s",
                event.event_name,
                event.session_id,
                event.question_id,
                reason,
            )
            return

        if attempts > _MAX_RETRIES:
            _stats["processed_failed"] += 1
            failure_reason = reason or "unknown_error"
            _dead_letter.append(
                DeadLetterRecord(
                    event_name=event.event_name,
                    session_id=event.session_id,
                    question_id=event.question_id,
                    reason=failure_reason,
                    attempts=attempts,
                )
            )
            logger.warning(
                "Telemetry dead-lettered event_name=%s sid=%s qid=%s attempts=%s reason=%s",
                event.event_name,
                event.session_id,
                event.question_id,
                attempts,
                failure_reason,
            )
            return

        delay_ms = min(_BASE_DELAY_MS * (2 ** (attempts - 1)), _MAX_DELAY_MS)
        logger.warning(
            "Telemetry write retry event_name=%s sid=%s qid=%s attempt=%s/%s in_ms=%s reason=%s",
            event.event_name,
            event.session_id,
            event.question_id,
            attempts,
            _MAX_RETRIES + 1,
            delay_ms,
            reason,
        )
        await asyncio.sleep(delay_ms / 1000)
