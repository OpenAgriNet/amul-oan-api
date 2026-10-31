"""GET /api/ui-flags/: which screens the UI shows on this deployment (#487)."""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.config import settings
from app.routers import ui_flags


def _flags():
    app = FastAPI()
    app.include_router(ui_flags.router, prefix=settings.api_prefix)
    response = TestClient(app).get("/api/ui-flags/")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(autouse=True)
def everything_off(monkeypatch):
    monkeypatch.setattr(settings, "test_cases_route_enabled", False)
    monkeypatch.setattr(settings, "doctor_persona_enabled", False)
    monkeypatch.setattr(settings, "chat_persona_override_enabled", False)


def test_everything_is_off_by_default_and_no_token_is_needed():
    assert _flags() == {"test_case_entry": False, "doctor_persona_selector": False}


def test_the_test_case_screen_follows_the_route_flag(monkeypatch):
    monkeypatch.setattr(settings, "test_cases_route_enabled", True)

    assert _flags()["test_case_entry"] is True


@pytest.mark.parametrize(
    "doctor, override, shown",
    [(True, False, False), (False, True, False), (True, True, True)],
)
def test_the_persona_picker_needs_both_persona_flags(monkeypatch, doctor, override, shown):
    monkeypatch.setattr(settings, "doctor_persona_enabled", doctor)
    monkeypatch.setattr(settings, "chat_persona_override_enabled", override)

    assert _flags()["doctor_persona_selector"] is shown
