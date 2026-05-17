from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.services.telemetry_normalizer import normalize_telemetry_payload
from app.tasks.telemetry_queue import enqueue_canonical_telemetry_event, get_telemetry_queue_stats
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/observability-service", tags=["telemetry"])


@router.post("/action/data/v3/telemetry")
async def ingest_telemetry(request: Request) -> JSONResponse:
    """
    Compatibility telemetry ingest endpoint.

    Accepts frontend telemetry payloads (single events or Sunbird SDK batch envelopes),
    validates supported schemas, normalizes them, and enqueues for async Langfuse delivery.
    """
    try:
        payload: Any = await request.json()
    except Exception:
        logger.warning("telemetry_ingest invalid_json=True")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "detail": "Invalid JSON payload"},
        )

    if not isinstance(payload, dict):
        logger.warning("telemetry_ingest invalid_payload_type=%s", type(payload).__name__)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "detail": "Payload must be a JSON object"},
        )

    normalize_status, canonical_events, reason = normalize_telemetry_payload(payload)
    if normalize_status == "invalid":
        logger.warning(
            "telemetry_ingest accepted=False validation_error=True reason=%s keys=%s",
            reason,
            sorted(payload.keys()),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "detail": "Unsupported or invalid telemetry schema"},
        )

    if normalize_status == "ignored_unknown":
        logger.info(
            "telemetry_ingest accepted=True normalized=False reason=%s keys=%s",
            reason,
            sorted(payload.keys()),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "ignored", "reason": reason},
        )

    enqueued_count = 0
    event_names = []
    last_enqueue_status = None
    last_enqueue_reason = None

    for canonical in (canonical_events or []):
        enqueue_status, enqueue_reason = enqueue_canonical_telemetry_event(canonical)
        last_enqueue_status = enqueue_status
        last_enqueue_reason = enqueue_reason
        event_names.append(canonical.event_name)
        if enqueue_status == "enqueued":
            enqueued_count += 1

    queue_stats = get_telemetry_queue_stats()

    logger.info(
        "telemetry_ingest accepted=True normalized=True events=%s enqueued=%s/%s queue_size=%s",
        event_names,
        enqueued_count,
        len(canonical_events or []),
        queue_stats.get("queue_size"),
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "normalized_events": event_names,
            "enqueued_count": enqueued_count,
            "total_events": len(canonical_events or []),
            "enqueue_status": last_enqueue_status,
            "enqueue_reason": last_enqueue_reason,
        },
    )
