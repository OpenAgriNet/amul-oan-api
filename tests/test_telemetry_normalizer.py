"""The frontend's telemetry events, as the ingest normalizes them: a Feedback event may
name the turn it is about with the trace id the chat response carried."""
from app.services.telemetry_normalizer import normalize_telemetry_payload


def _sdk_batch(eks):
    return {
        "id": "ekstep.telemetry", "ver": "3.0", "ets": 1760000000000,
        "events": [{"eid": "OE_ITEM_RESPONSE", "ets": 1760000000000, "sid": "web-session", "edata": {"eks": eks}}],
    }


def _feedback(**more):
    return {
        "qid": "q-1",
        "target": {"type": "Feedback", "feedbackDetails": {
            "feedbackText": "useful", "sessionId": "web-session", "questionText": "q", "answerText": "a",
            "feedbackType": "like", "pipeline": "default",
        }},
        **more,
    }


def test_a_feedback_carries_the_trace_id_it_names():
    status, events, reason = normalize_telemetry_payload(_sdk_batch(_feedback(trace_id="b" * 32)))

    assert (status, reason) == ("normalized", None)
    (event,) = events
    assert (event.event_name, event.session_id, event.question_id, event.trace_id) == ("feedback", "web-session", "q-1", "b" * 32)


def test_a_feedback_without_a_trace_id_has_none():
    status, events, _ = normalize_telemetry_payload(_sdk_batch(_feedback()))

    assert status == "normalized" and events[0].trace_id is None
