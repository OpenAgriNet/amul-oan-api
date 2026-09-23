from app.services.telemetry_stamps import forward_chat_telemetry_metadata


def test_forward_telemetry_metadata_stamps_schema_service_and_release():
    assert forward_chat_telemetry_metadata("test-release-sha") == {
        "amul.schema_version": "chat.turn.v1",
        "service": "amul-oan-api",
        "release": "test-release-sha",
    }


def test_forward_telemetry_metadata_marks_an_unknown_release():
    assert forward_chat_telemetry_metadata(None)["release"] == "unknown"
