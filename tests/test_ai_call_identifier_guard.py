"""create_ai_call must refuse identifiers that cannot be real.

With no farmer/technician context the model does not stop — it invents them and
books anyway. On chat-production in the 30d to 2026-09-08 that reached
CreateAICall three times ({"union_code":"null",...}, "not_available", and
U11223/S67890/F12345/T001) and 500'd every time. Over the same window the
patterns below reject 4 of 2,728 booking attempts — all four already failed
upstream — and none of the 2,423 that succeeded.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agents.tools import ai_call as ai_mod
from app.models.ai_call import AISpecies

VALID_TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="  # 24 base64 chars — the real prod shape


# identifiers actually sent to the booking API by the model, in both channels
@pytest.mark.parametrize("identifiers", [
    ("null", "null", "null", "null"),
    ("not_available", "not_available", "not_available", "773"),
    ("U11223", "S67890", "F12345", "T001"),
    ("2103", "01185", "159", "tkxUMxk2v8VAO2j41wqaA=="),  # 23 chars — truncated id
    ("159", "00002", "Rathod Sanjay Shri Jagats", "/cT4TzbfxFOo+L+ZN9x1ZQ=="),
    ("", "", "", ""),
])
def test_invented_identifiers_are_rejected(identifiers):
    assert ai_mod._invalid_booking_identifier(*identifiers) is not None


# real triples from successful prod bookings — codes are NOT always numeric
@pytest.mark.parametrize("codes", [
    ("159", "00243", "0067"), ("M001", "2169", "0092"),
    ("2021", "NA4192", "NA0001"), ("2004", "55", "NA01"),
])
def test_real_prod_identifiers_pass(codes):
    assert ai_mod._invalid_booking_identifier(*codes, VALID_TECH_ID) is None


def test_invented_booking_reaches_neither_route(monkeypatch):
    """Refused before the Beckn identity reads and before PashuGPT."""
    async def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an invented booking must not leave the process")

    monkeypatch.setattr(ai_mod, "_book_via_network", boom)
    monkeypatch.setattr(ai_mod, "_book_direct", boom)

    async def in_scope():
        return True

    ctx = SimpleNamespace(deps=SimpleNamespace(session_id="s1", ensure_in_scope=in_scope,
                                               farmer_unions=[], mobile="9999999999"))
    out = asyncio.run(ai_mod.create_ai_call(ctx, "U11223", "S67890", "F12345", "T001",
                                            AISpecies.COW))
    assert out == ai_mod.INVALID_IDENTIFIERS_MESSAGE
