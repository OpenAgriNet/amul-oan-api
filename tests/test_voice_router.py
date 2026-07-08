"""Inc 7.5 — the voice router is wired and the app assembles with it live.

Importing main builds the FastAPI app and includes every router (which imports the
voice pipeline), so this proves the whole service assembles with /voice registered.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def test_voice_router_exposes_voice_route():
    from app.routers.voice import router

    paths = [getattr(r, "path", "") for r in router.routes]
    assert any("/voice" in p for p in paths)


def test_app_assembles_with_voice_route_registered():
    import main

    paths = [getattr(r, "path", "") for r in main.app.routes]
    assert any("/voice" in p for p in paths), f"no /voice route in {paths}"
