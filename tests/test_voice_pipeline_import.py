"""Inc 7.4b — smoke test that the ported voice pipeline imports cleanly.

voice.py is 1876 lines ported verbatim; its real risk is an import-chain break
(a symbol that resolves by name but fails at runtime import). Importing the module
exercises the whole chain (agents.voice -> agent build, farmer_cache, translation,
moderation, voice_trace, stt_signals, app.utils, ...). Functional behavior is
covered by the voice test suite in 7.6.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def test_voice_pipeline_imports_cleanly():
    import app.services.voice as voice

    # public entry point the router (7.5) will call
    assert hasattr(voice, "stream_voice_message")
    # a couple of internal helpers that came with it
    assert hasattr(voice, "_is_signed_in_session")
    assert hasattr(voice, "get_or_fetch_farmer_data")
