"""Smoke test that the app assembles.

The only test that imported ``main`` was deleted with the voice router, which
left router wiring, lifespan setup and the refresh worker's import surface with
zero coverage.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def test_app_assembles_and_registers_chat_routes():
    import main

    paths = {getattr(r, "path", None) for r in main.app.routes}
    assert "/api/chat/" in paths
    assert not any(p and p.startswith("/api/voice") for p in paths)
