from types import SimpleNamespace

import pytest

from agents.services import beckn_amul
from agents.tools import ai_call, beckn_network, health_call
from app.models.ai_call import AISpecies
from app.models.health_call import HealthCaseType


async def _in_scope():
    return True


def _ctx():
    return SimpleNamespace(
        deps=SimpleNamespace(
            session_id=None,
            mobile="9000000000",
            farmer_unions=["kaira"],
            ensure_in_scope=_in_scope,
        ),
        tool_call_id="tool-1",
    )


def _callback_mode(monkeypatch, module):
    monkeypatch.setattr(module.settings, "enable_network", True)
    monkeypatch.setattr(module.settings, "beckn_callback_transactions_enabled", True)


@pytest.mark.asyncio
async def test_ai_confirm_uses_canonical_owned_account_and_discovered_technician(monkeypatch):
    _callback_mode(monkeypatch, ai_call)
    monkeypatch.setattr(ai_call.settings, "ai_call_booking_guard_enabled", False)
    account = beckn_amul.AuthenticatedFarmerAccount("CANON-U", "CANON-S", "CANON-F")

    async def resolve(mobile, **kwargs):
        assert mobile == "9000000000"
        assert kwargs["union_code"] == "MODEL-U"
        return account

    async def technicians(**kwargs):
        assert kwargs["union_code"] == "CANON-U"
        return [beckn_amul.AITechnicianRecord(userId="TECH-1", fullName="Technician")]

    captured = {}
    async def confirm(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return beckn_network.NetworkBookingResult(True, "TICKET", "booked successfully")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_amul, "search_ai_technicians", technicians)
    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", confirm)

    result = await ai_call.create_ai_call(
        _ctx(), "MODEL-U", "MODEL-S", "MODEL-F", "TECH-1", AISpecies.COW
    )

    assert "booked successfully" in result
    assert captured["args"][:5] == ("CANON-U", "CANON-S", "CANON-F", "TECH-1", "cow")


@pytest.mark.asyncio
async def test_ai_confirm_is_not_sent_for_unowned_account(monkeypatch):
    _callback_mode(monkeypatch, ai_call)
    monkeypatch.setattr(ai_call.settings, "ai_call_booking_guard_enabled", False)

    async def resolve(*args, **kwargs):
        return None

    async def confirm(*args, **kwargs):
        raise AssertionError("confirm must not be sent for an unowned account")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", confirm)

    result = await ai_call.create_ai_call(
        _ctx(), "MODEL-U", "MODEL-S", "MODEL-F", "TECH-1", AISpecies.COW
    )

    assert "does not belong" in result


@pytest.mark.asyncio
async def test_health_confirm_uses_canonical_owned_account(monkeypatch):
    _callback_mode(monkeypatch, health_call)
    account = beckn_amul.AuthenticatedFarmerAccount("CANON-U", "CANON-S", "CANON-F")

    async def resolve(*args, **kwargs):
        return account

    captured = {}
    async def confirm(*args, **kwargs):
        captured["args"] = args
        return beckn_network.NetworkBookingResult(True, "HEALTH-1", "booked successfully")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_network, "network_create_health_call_result", confirm)

    result = await health_call.create_health_call(
        _ctx(),
        "MODEL-U",
        "MODEL-S",
        "MODEL-F",
        AISpecies.BUFFALO,
        HealthCaseType.NORMAL,
        "not eating",
    )

    assert "booked successfully" in result
    assert captured["args"][:3] == ("CANON-U", "CANON-S", "CANON-F")
