import asyncio
from typing import Literal

from app.models.telemetry import CanonicalTelemetryEvent
from app.observability import is_trace_id, non_turn_root, surface_environment
from app.services.telemetry_mapper import map_canonical_event_to_langfuse
from app.services.telemetry_stamps import FRONTEND_EVENT, non_turn_v1_metadata
from helpers.utils import get_logger

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

logger = get_logger(__name__)


WriteStatus = Literal["queued", "skipped_unconfigured", "failed"]


async def write_canonical_event_to_langfuse(
    canonical: CanonicalTelemetryEvent,
) -> tuple[WriteStatus, str | None]:
    """
    Best-effort writer for telemetry canonical events to Langfuse.

    Each event is a turn.v1 frontend_event trace (amul-telemetry
    telemetry/turn.v1.yaml): a root named frontend_event with the amul.* keys,
    the session and the user, and under it the ``frontend.<event>`` observation
    with what the frontend sent. turn.v1 keeps a trace's input and output for
    turns. A feedback's ``user_feedback`` score goes on the turn when the frontend
    sends the turn's trace id (the chat response's X-Trace-Id) with it, else on
    this trace.

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
    root_metadata = non_turn_v1_metadata(
        FRONTEND_EVENT,
        # The frontend telemetry is the web app's; WhatsApp has none.
        pipeline="webapp",
        event=canonical.event_name,
    )

    try:
        with non_turn_root(
            FRONTEND_EVENT, root_metadata, session_id=canonical.session_id, user_id=canonical.user_id
        ):
            with langfuse.start_as_current_observation(
                name=mapped["observation_name"],
                as_type="generation",
                input=mapped.get("input"),
                output=mapped.get("output"),
                metadata=mapped.get("metadata", {}),
            ) as observation:
                # Best-effort score capture for feedback events.
                score = mapped.get("score")
                if score and score.get("value") is not None:
                    if is_trace_id(canonical.trace_id):
                        # On the turn's trace, in chat's environment as the turn is.
                        langfuse.create_score(
                            trace_id=canonical.trace_id,
                            name=score["name"],
                            value=score["value"],
                            comment=score.get("comment"),
                            environment=surface_environment("chat"),
                        )
                    elif hasattr(observation, "score"):
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
        # flush() can block due to network/export work, so run it off the event loop.
        if hasattr(langfuse, "flush"):
            try:
                await asyncio.to_thread(langfuse.flush)
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
