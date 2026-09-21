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

    if canonical.event_name == "chat_trace_bootstrap":
        mapped["observation_name"] = "backend.chat.trace_bootstrap"
        mapped["tags"].append("backend-event:chat_trace_bootstrap")
        trace_input = payload.get("trace_input")
        if isinstance(trace_input, dict):
            mapped["input"] = trace_input
        else:
            mapped["input"] = {}
        pipeline_profile = payload.get("pipeline_profile")
        if pipeline_profile is not None:
            mapped["metadata"]["pipeline_profile"] = pipeline_profile
            # Preserve prior chat-path score semantics: categorical + session-sticky
            # upsert via deterministic score_id (variant-{session_id}).
            session_id_safe = (canonical.session_id or "")[:200]
            score_id = payload.get("score_id") or (
                f"variant-{session_id_safe}" if session_id_safe else None
            )
            mapped["score"] = {
                "name": "pipeline_profile",
                "value": pipeline_profile,
                "data_type": "CATEGORICAL",
                "comment": "Sticky pipeline variant for this session",
            }
            if score_id:
                mapped["score"]["score_id"] = score_id
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
