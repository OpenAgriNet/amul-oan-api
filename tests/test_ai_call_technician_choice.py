"""With two or more technicians, create_ai_call books only once the farmer has picked one.

On dev on 5 Nov 2026 (fixture farmer F2) the farmer was shown Mahesh and Dinesh, answered
only "for the buffalo", and the model booked Mahesh itself; her "send Dinesh Patel" then
booked a second visit.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from agents.tools import ai_call
from agents.tools.beckn import amul as beckn_amul
from agents.tools.beckn import network as beckn_network
from agents.tools.models.ai_call import AISpecies

MAHESH = "MaheshChaudharyAIT0003=="
DINESH = "DineshPatelAITech00004=="
TECHNICIANS = [
    beckn_amul.AITechnicianRecord(userId=MAHESH, fullName="Mahesh Chaudhary",
                                  gujaratiFullName="મહેશ ચૌધરી", mobileNumber="5550000203"),
    beckn_amul.AITechnicianRecord(userId=DINESH, fullName="Dinesh Patel",
                                  gujaratiFullName="દિનેશ પટેલ", mobileNumber="5550000204"),
]


async def _in_scope():
    return True


def _ctx(*farmer_said):
    return SimpleNamespace(
        deps=SimpleNamespace(session_id=None, mobile="5550000002", farmer_unions=["dudhsagar"],
                             farmer_profile_status="found", ensure_in_scope=_in_scope),
        tool_call_id="tool-1",
        messages=[ModelRequest(parts=[UserPromptPart(content=text)]) for text in farmer_said],
    )


def _book(monkeypatch, ctx, user_id, technicians=TECHNICIANS):
    monkeypatch.setattr(ai_call.settings, "ai_call_booking_guard_enabled", False)
    booked = []

    async def resolve(mobile, **kwargs):
        return beckn_amul.AuthenticatedFarmerAccount("EVU02", "EV9902", "EV990201")

    async def search(**kwargs):
        return technicians

    async def confirm(*args, **kwargs):
        booked.append(args[3])
        return beckn_network.NetworkBookingResult(True, "99051120260001", "booked successfully")

    monkeypatch.setattr(beckn_amul, "resolve_authenticated_account", resolve)
    monkeypatch.setattr(beckn_amul, "search_ai_technicians", search)
    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", confirm)
    result = asyncio.run(ai_call.create_ai_call(ctx, "EVU02", "EV9902", "EV990201", user_id, AISpecies.BUFFALO))
    return result, booked


@pytest.mark.parametrize("farmer_said", [
    ("I want to book a technician visit for artificial insemination.", "For the buffalo. Laxmi has come into heat."),
    ("Send Dinesh Patel.",),  # she named Dinesh, the model picked Mahesh
])
def test_no_booking_with_a_technician_she_did_not_pick(monkeypatch, farmer_said):
    result, booked = _book(monkeypatch, _ctx(*farmer_said), MAHESH)
    assert booked == []
    assert result.startswith("No booking was made yet")
    assert "મહેશ ચૌધરી (5550000203)" in result and "દિનેશ પટેલ (5550000204)" in result


@pytest.mark.parametrize("pick", [
    "Send Dinesh Patel.", "દિનેશભાઈને મોકલજો.", "Book the one on 5550000204.", "The second one.", "2",
])
def test_booking_once_she_picks(monkeypatch, pick):
    result, booked = _book(monkeypatch, _ctx("For the buffalo.", pick), DINESH)
    assert booked == [DINESH]
    assert "booked successfully" in result


def test_a_first_name_picks_a_full_record_name(monkeypatch):
    technicians = [TECHNICIANS[0], beckn_amul.AITechnicianRecord(
        userId=DINESH, fullName="DINESHBHAI RAMESHBHAI PATEL", mobileNumber="5550000204")]
    result, booked = _book(monkeypatch, _ctx("For the buffalo.", "Send Dinesh."), DINESH, technicians)
    assert booked == [DINESH]
    assert "booked successfully" in result


def test_a_single_technician_needs_no_pick(monkeypatch):
    result, booked = _book(monkeypatch, _ctx("Book an AI visit for my buffalo."), DINESH, TECHNICIANS[1:])
    assert booked == [DINESH]
    assert "booked successfully" in result
