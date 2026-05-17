import json
from contextlib import nullcontext
from typing import Literal

from app.models.telemetry import CanonicalTelemetryEvent
from app.services.telemetry_mapper import map_canonical_event_to_langfuse
from helpers.utils import get_logger

try:
    from langfuse import propagate_attributes, get_client as get_langfuse_client
except ImportError:
    propagate_attributes = None
    get_langfuse_client = None

logger = get_logger(__name__)


WriteStatus = Literal["queued", "skipped_unconfigured", "failed"]


def _stringify_metadata_for_propagation(metadata: dict) -> dict[str, str]:
    """
    propagate_attributes metadata is OTEL-style baggage and expects string values.
    Keep full metadata for observation writes, but pass a string-safe copy for propagation.
    """
    safe: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            safe[key] = value
        elif isinstance(value, (int, float, bool)):
            safe[key] = str(value)
        else:
            safe[key] = json.dumps(value, ensure_ascii=True, default=str)
    return safe


def write_canonical_event_to_langfuse(
    canonical: CanonicalTelemetryEvent,
) -> tuple[WriteStatus, str | None]:
    """
    Best-effort writer for telemetry canonical events to Langfuse.

    The ingest endpoint must not fail if Langfuse is unavailable, so this function
    always returns a status instead of raising.
    """
    if get_langfuse_client is None:
        return "skipped_unconfigured", "langfuse package not available"

    try:
        langfuse = get_langfuse_client()
    except Exception as exc:
        return "skipped_unconfigured", f"langfuse client not configured: {exc}"

    mapped = map_canonical_event_to_langfuse(canonical)
    tags = mapped.get("tags", [])
    metadata = mapped.get("metadata", {})
    propagation_metadata = _stringify_metadata_for_propagation(metadata)
    session_id = canonical.session_id
    user_id = canonical.user_id

    trace_ctx = (
        propagate_attributes(
            session_id=(session_id or "")[:200] if session_id else None,
            user_id=(user_id or "")[:200] if user_id else None,
            metadata=propagation_metadata,
            tags=tags,
        )
        if propagate_attributes is not None
        else nullcontext()
    )

    try:
        with trace_ctx:
            with langfuse.start_as_current_observation(
                name=mapped["observation_name"],
                as_type="generation",
                input=mapped.get("input"),
                output=mapped.get("output"),
                metadata=metadata,
            ) as observation:
                if mapped.get("input") is not None:
                    langfuse.set_current_trace_io(input=mapped["input"])
                if mapped.get("output") is not None:
                    langfuse.set_current_trace_io(output=mapped["output"])

                # Best-effort score capture for feedback events.
                score = mapped.get("score")
                if score and score.get("value") is not None:
                    if hasattr(observation, "score"):
                        observation.score(
                            name=score["name"],
                            value=score["value"],
                            comment=score.get("comment"),
                        )
                    elif hasattr(langfuse, "score_current_trace"):
                        langfuse.score_current_trace(
                            name=score["name"],
                            value=score["value"],
                            comment=score.get("comment"),
                        )

        # Langfuse export is async/OTEL-backed; enqueue success does not guarantee remote ingest.
        # Try best-effort flush to surface immediate transport errors when supported.
        if hasattr(langfuse, "flush"):
            try:
                langfuse.flush()
            except Exception as flush_exc:
                logger.warning(
                    "Langfuse flush failed event_name=%s sid=%s qid=%s error=%s",
                    canonical.event_name,
                    canonical.session_id,
                    canonical.question_id,
                    flush_exc,
                )
                return "failed", str(flush_exc)

        return "queued", None
    except Exception as exc:
        logger.warning(
            "Langfuse telemetry write failed event_name=%s sid=%s qid=%s error=%s",
            canonical.event_name,
            canonical.session_id,
            canonical.question_id,
            exc,
        )
        return "failed", str(exc)
