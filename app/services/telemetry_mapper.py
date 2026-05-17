from typing import Any

from app.models.telemetry import CanonicalTelemetryEvent


def map_canonical_event_to_langfuse(canonical: CanonicalTelemetryEvent) -> dict[str, Any]:
    """
    Convert canonical telemetry event into a Langfuse-friendly payload shape.

    This function is intentionally provider-agnostic and does not call Langfuse SDK
    directly. It only extracts standardized fields from canonical payloads.
    """
    mapped: dict[str, Any] = {
        "event_name": canonical.event_name,
        "session_id": canonical.session_id,
        "question_id": canonical.question_id,
        "user_id": canonical.user_id,
        "pipeline": canonical.pipeline,
        "ts": canonical.ts,
        "trace_name": "frontend.telemetry",
        "observation_name": f"frontend.{canonical.event_name}",
        "tags": [f"frontend-event:{canonical.event_name}"],
        "metadata": {
            "schema_version": canonical.schema_version,
        },
    }

    if canonical.pipeline:
        mapped["tags"].append(f"pipeline:{canonical.pipeline}")

    payload = canonical.payload
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    target_type = target.get("type")
    mapped["metadata"]["target_type"] = target_type

    if canonical.event_name == "question":
        details = target.get("questionsDetails", {}) if isinstance(target, dict) else {}
        mapped["input"] = {
            "question": details.get("questionText"),
            "channel": payload.get("channel"),
        }
        return mapped

    if canonical.event_name == "question_response":
        details = target.get("questionsDetails", {}) if isinstance(target, dict) else {}
        performance = target.get("performance", {}) if isinstance(target, dict) else {}
        mapped["input"] = {
            "question": details.get("questionText"),
        }
        mapped["output"] = {
            "answer": details.get("answerText"),
        }
        mapped["metadata"]["performance"] = performance
        return mapped

    if canonical.event_name == "error":
        details = target.get("errorDetails", {}) if isinstance(target, dict) else {}
        mapped["input"] = {
            "question_id": canonical.question_id,
        }
        mapped["output"] = {
            "error": details.get("errorText"),
        }
        mapped["metadata"]["error"] = True
        return mapped

    if canonical.event_name == "feedback":
        details = target.get("feedbackDetails", {}) if isinstance(target, dict) else {}
        mapped["input"] = {
            "question": details.get("questionText"),
            "answer": details.get("answerText"),
        }
        mapped["output"] = {
            "feedback_text": details.get("feedbackText"),
            "feedback_type": details.get("feedbackType"),
            "rating": details.get("rating"),
        }
        mapped["score"] = {
            "name": "user_feedback",
            "value": _feedback_score_value(details.get("feedbackType"), details.get("rating")),
            "comment": details.get("feedbackText"),
        }
        return mapped

    # anonymous_token_issued
    mapped["input"] = {
        "sid": payload.get("sid"),
        "uid": payload.get("uid"),
        "did": payload.get("did"),
    }
    mapped["metadata"]["eid"] = payload.get("eid")
    return mapped


def _feedback_score_value(feedback_type: Any, rating: Any) -> float | None:
    if isinstance(rating, (int, float)):
        return float(rating)
    if isinstance(feedback_type, str):
        lowered = feedback_type.lower()
        if lowered == "like":
            return 1.0
        if lowered == "dislike":
            return 0.0
    return None
