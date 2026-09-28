"""telemetry/mappings/chat.yaml must only read what a chat contract sends.

test_chat_telemetry_schema_stamps.py checks chat.py against its contract. This
checks the reading side: every field the mapping reads for a stamped version has
to come from a key that version sends. Without it, renaming a key in chat.py and
the contract passes CI while the mapping quietly reads nothing.
"""

import json
from pathlib import Path

import pytest

from app.services.telemetry_era_adapters import load_chat_mappings

CONTRACTS = Path(__file__).resolve().parents[1] / "telemetry" / "contracts"

# Set on every chat turn but not listed in chat.turn.v1: the session goes through
# propagate_attributes and the answer through set_current_trace_io.
TRACE_FIELDS = {"sessionId", "session_id", "output"}


def _contracts():
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(CONTRACTS.glob("chat.turn.v*.json"))}


def _sent(path, contract):
    """Whether this contract sends what a mapping path reads."""
    if path in TRACE_FIELDS:
        return True
    head, _, key = path.partition(".")
    if head == "input":
        return key in contract["trace_input"]["required"]
    if head == "metadata":
        return key in contract["metadata"]["required"] + contract["metadata"].get("optional", [])
    return False


@pytest.mark.parametrize("version", sorted(_contracts()))
def test_every_chat_contract_has_a_mapping(version):
    assert version in load_chat_mappings(), (
        f"{version} has no entry in telemetry/mappings/chat.yaml. Add it in the same change as the contract, "
        "or the adapters reject these traces."
    )


@pytest.mark.parametrize("version", sorted(_contracts()))
def test_every_mapped_chat_field_is_sent(version):
    contract = _contracts()[version]
    mapping = load_chat_mappings()[version]
    unreadable = {field: list(paths) for field, paths in mapping.fields.items() if not any(_sent(path, contract) for path in paths)}

    assert not unreadable, (
        f"{version} reads {unreadable} but the contract doesn't send it. Point each field at a key that is sent, "
        "and if a key was renamed, release it as a new version."
    )


def test_a_path_is_read_from_the_right_part_of_the_trace():
    contract = _contracts()["chat.turn.v1"]

    assert _sent("metadata.user_id", contract)
    assert _sent("input.query", contract)
    assert _sent("output", contract)
    assert not _sent("metadata.variant", contract)
    assert not _sent("input.user_id", contract)
    assert not _sent("no_such_field", contract)
