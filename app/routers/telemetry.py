import json
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.services.telemetry_normalizer import normalize_telemetry_payload
from app.tasks.telemetry_queue import enqueue_canonical_telemetry_event, get_telemetry_queue_stats
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/observability-service", tags=["telemetry"])


def _truncate_string(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[:max_len]


def _sanitize_telemetry_payload(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_telemetry_payload(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_telemetry_payload(item, parent_key) for item in value]
    if isinstance(value, str):
        if parent_key == "questionText":
            return _truncate_string(value, settings.telemetry_ingest_max_question_text_len)
        if parent_key == "answerText":
            return _truncate_string(value, settings.telemetry_ingest_max_answer_text_len)
        if parent_key == "feedbackText":
            return _truncate_string(value, settings.telemetry_ingest_max_feedback_text_len)
        if parent_key == "errorText":
            return _truncate_string(value, settings.telemetry_ingest_max_error_text_len)
        return _truncate_string(value, settings.telemetry_ingest_max_string_len_default)
    return value


@router.post("/action/data/v3/telemetry")
async def ingest_telemetry(
    request: Request,
    _user_info: dict = Depends(get_chat_user),
) -> JSONResponse:
    """
    Compatibility telemetry ingest endpoint.

    Accepts frontend telemetry payloads (single events or Sunbird SDK batch envelopes),
    validates supported schemas, normalizes them, and enqueues for async Langfuse delivery.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.telemetry_ingest_max_body_bytes:
                logger.warning("telemetry_ingest rejected=body_too_large content_length=%s", content_length)
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"status": "error", "detail": "Payload too large"},
                )
        except ValueError:
            logger.warning("telemetry_ingest invalid_content_length=%s", content_length)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "detail": "Invalid Content-Length header"},
            )

    try:
        raw_body = await request.body()
        if len(raw_body) > settings.telemetry_ingest_max_body_bytes:
            logger.warning("telemetry_ingest rejected=body_too_large bytes=%s", len(raw_body))
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"status": "error", "detail": "Payload too large"},
            )

        payload: Any = json.loads(raw_body)
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

    sanitized_payload = _sanitize_telemetry_payload(payload)
    normalize_status, canonical_events, reason = normalize_telemetry_payload(sanitized_payload)
    if normalize_status == "invalid":
        logger.warning(
            "telemetry_ingest accepted=False validation_error=True reason=%s keys=%s",
            reason,
            sorted(sanitized_payload.keys()),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "detail": "Unsupported or invalid telemetry schema"},
        )

    if normalize_status == "ignored_unknown":
        logger.info(
            "telemetry_ingest accepted=True normalized=False reason=%s keys=%s",
            reason,
            sorted(sanitized_payload.keys()),
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
