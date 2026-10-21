"""Chat turns send turn.v1 as telemetry/contracts/turn.v1/chat.json says, step by step.

tests/test_chat_telemetry_schema_stamps.py checks the root's key names in chat.py;
this one runs chat turns through the real run_turn with a recording Langfuse
client, as tests/test_voice_telemetry_contract.py does for voice, and compares
the steps Langfuse was handed with the contract.
"""

import asyncio
import json
from pathlib import Path
from types import MappingProxyType

from app.channels.chat import WEB
from app.services import chat as chat_service
from app.turn.types import Turn
from tests.test_chat_turn_contract import patch_turn
from tests.test_voice_telemetry import _Langfuse

CONTRACT = Path(__file__).resolve().parents[1] / "telemetry" / "contracts" / "turn.v1" / "chat.json"


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="chat-contract",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="chat-contract",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


def _recorded_turn(monkeypatch, turn=None):
    """One chat turn with every outside stage stubbed, recorded as Langfuse would get it."""
    patch_turn(monkeypatch)
    client = _Langfuse()
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(chat_service, "propagate_attributes", client.propagate_attributes)

    async def _go():
        stream = chat_service.run_turn(turn or _turn(), chat_service.CHAT_SURFACE, scheduler=_Scheduler())
        return [emission async for emission in stream]

    asyncio.run(_go())
    return client


def test_the_moderation_step_sends_the_contracts_output(monkeypatch):
    """Chat's model gives a category and an action, no reason: the action goes under its own name,
    not as turn.v1's reason, which is a model's reason (voice's)."""
    client = _recorded_turn(monkeypatch)

    moderation = client.step("moderation").sent
    assert moderation["output"] == {"rejected": False, "category": "valid_agricultural", "action": "allow"}
    assert set(moderation["output"]) == set(_contract()["steps"]["moderation"]["output"])
