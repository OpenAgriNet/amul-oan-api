from typing import Any, Literal

from pydantic import ValidationError

from app.models.telemetry import (
    AnonymousTokenIssuedEvent,
    CanonicalTelemetryEvent,
    TelemetrySdkEvent,
)


NormalizeStatus = Literal["normalized", "ignored_unknown", "invalid"]


def normalize_telemetry_payload(
    payload: dict[str, Any],
) -> tuple[NormalizeStatus, list[CanonicalTelemetryEvent] | None, str | None]:
    """
    Validate incoming frontend telemetry payload and map to canonical event(s).

    The Sunbird SDK sends a batch envelope: { id, ver, ets, events: [...] }.
    Each event inside wraps our data under edata.eks.
    Direct-fetch payloads (e.g. OE_ANONYMOUS_TOKEN_ISSUED) arrive as flat objects.

    Returns:
    - ("normalized", [events], None) when at least one event matched
    - ("ignored_unknown", None, reason) when nothing matched
    - ("invalid", None, reason) when a supported shape fails validation
    """
    if not isinstance(payload, dict):
        return "invalid", None, "Payload must be a JSON object"

    # Branch 0: Sunbird SDK batch envelope — unwrap and normalize each event.
    if isinstance(payload.get("events"), list) and payload.get("id") is not None:
        events = payload["events"]
        if not events:
            return "ignored_unknown", None, "Batch envelope with empty events array"

        all_canonical: list[CanonicalTelemetryEvent] = []
        last_reason: str | None = None

        for raw_event in events:
            if not isinstance(raw_event, dict):
                continue
            unwrapped = _unwrap_sdk_event(raw_event)
            status, canonical, reason = _normalize_single_event(unwrapped)
            if status == "normalized" and canonical is not None:
                all_canonical.append(canonical)
            elif status == "invalid":
                last_reason = reason
            elif status == "ignored_unknown":
                last_reason = reason

        if all_canonical:
            return "normalized", all_canonical, None
        return "ignored_unknown", None, last_reason or "No supported events in batch"

    # Branch 1+2: single event (direct fetch or standalone SDK event).
    unwrapped = _unwrap_sdk_event(payload)
    status, canonical, reason = _normalize_single_event(unwrapped)
    if status == "normalized" and canonical is not None:
        return "normalized", [canonical], None
    return status, None, reason


def _unwrap_sdk_event(raw: dict[str, Any]) -> dict[str, Any]:
    """
    The Sunbird SDK wraps the actual payload under edata.eks.
    If that structure exists, lift eks fields to root level so the normalizer
    can find target/qid/sid/channel in the expected places.
    Also preserve top-level sid/uid/channel/did from the SDK envelope.
    """
    eks = None
    edata = raw.get("edata")
    if isinstance(edata, dict):
        eks = edata.get("eks")

    if not isinstance(eks, dict):
        return raw

    merged: dict[str, Any] = {}
    # Start with eks fields (contains target, qid, type, etc.)
    merged.update(eks)
    # Overlay envelope-level fields that the normalizer needs
    for key in ("sid", "uid", "did", "channel", "eid", "ets", "mid", "pdata", "ver"):
        if key in raw:
            merged[key] = raw[key]
    # Keep original edata for anonymous-token style events
    if "edata" in raw:
        merged.setdefault("edata", raw["edata"])
    return merged


def _normalize_single_event(
    payload: dict[str, Any],
) -> tuple[NormalizeStatus, CanonicalTelemetryEvent | None, str | None]:
    """Normalize a single unwrapped telemetry event."""
    if not isinstance(payload, dict):
        return "invalid", None, "Payload must be a JSON object"

    # Branch 1: explicit event-id based payloads (currently anonymous token issued).
    if payload.get("eid") is not None:
        if payload.get("eid") == "OE_ANONYMOUS_TOKEN_ISSUED":
            try:
                event = AnonymousTokenIssuedEvent.model_validate(payload)
            except ValidationError as exc:
                return "invalid", None, str(exc)

            canonical = CanonicalTelemetryEvent(
                event_name="anonymous_token_issued",
                session_id=event.sid,
                user_id=event.uid or (event.edata.eks.uid if event.edata and event.edata.eks else None),
                ts=event.ets,
                payload=event.model_dump(),
            )
            return "normalized", canonical, None

        # SDK-generated events have eid like OE_ITEM_RESPONSE, OE_START, OE_END.
        # For these, the actual data is already unwrapped above — check for target.
        if isinstance(payload.get("target"), dict) and payload["target"].get("type") is not None:
            return _normalize_target_event(payload)

        # OE_START and OE_END don't carry question/response data; safe to ignore.
        eid = payload.get("eid")
        if eid in ("OE_START", "OE_END"):
            return "ignored_unknown", None, f"Lifecycle event eid={eid} (no data to forward)"

        return "ignored_unknown", None, f"Unsupported eid={eid}"

    # Branch 2: SDK response() payloads with target.type discriminator.
    if isinstance(payload.get("target"), dict) and payload["target"].get("type") is not None:
        return _normalize_target_event(payload)

    return "ignored_unknown", None, "Unsupported telemetry payload shape"


def _normalize_target_event(
    payload: dict[str, Any],
) -> tuple[NormalizeStatus, CanonicalTelemetryEvent | None, str | None]:
    """Normalize events that carry a target.type discriminator."""
    target_type = payload["target"].get("type")
    if target_type not in {"Question", "QuestionResponse", "Error", "Feedback"}:
        return "ignored_unknown", None, f"Unsupported target.type={target_type}"

    try:
        event = TelemetrySdkEvent.model_validate(payload)
    except ValidationError as exc:
        return "invalid", None, str(exc)

    if target_type == "Question":
        details = event.target.questionsDetails
        canonical = CanonicalTelemetryEvent(
            event_name="question",
            session_id=event.sid,
            question_id=event.qid,
            pipeline=details.pipeline,
            payload=event.model_dump(),
        )
        return "normalized", canonical, None

    if target_type == "QuestionResponse":
        details = event.target.questionsDetails
        canonical = CanonicalTelemetryEvent(
            event_name="question_response",
            session_id=event.sid,
            question_id=event.qid,
            pipeline=details.pipeline,
            payload=event.model_dump(),
        )
        return "normalized", canonical, None

    if target_type == "Error":
        canonical = CanonicalTelemetryEvent(
            event_name="error",
            session_id=event.sid,
            question_id=event.qid,
            payload=event.model_dump(),
        )
        return "normalized", canonical, None

    details = event.target.feedbackDetails
    canonical = CanonicalTelemetryEvent(
        event_name="feedback",
        session_id=event.sid,
        question_id=event.qid,
        pipeline=details.pipeline,
        payload=event.model_dump(),
    )
    return "normalized", canonical, None
