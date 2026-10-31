"""Which screens the UI shows on this deployment (GET /api/ui-flags/, #487).

The UI used to decide this from build-time env vars, so turning a screen on or off
needed a new build. No auth: the flags aren't secret, and the UI needs them before
sign-in (the test case icon shows on the login screen).
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(prefix="/ui-flags", tags=["ui-flags"])


class UIFlags(BaseModel):
    test_case_entry: bool
    doctor_persona_selector: bool


@router.get("/", response_model=UIFlags)
async def ui_flags() -> UIFlags:
    return UIFlags(
        # The screen and POST /api/test-cases/ are on together.
        test_case_entry=settings.test_cases_route_enabled,
        # resolve_chat_persona (app/personas.py) ignores the picker unless both are on.
        doctor_persona_selector=settings.doctor_persona_enabled and settings.chat_persona_override_enabled,
    )
