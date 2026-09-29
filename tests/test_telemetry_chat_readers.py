"""telemetry/mappings/chat.yaml must only read what a chat contract sends.

test_chat_telemetry_schema_stamps.py checks chat.py against its contract. This
checks the reading side: every field or attribute the mapping reads for a stamped
version has to come from a key that version sends. Without it, renaming a key in
chat.py and the contract passes CI while the mapping quietly reads nothing. It
also checks the score names chat.py sends against the contract.
"""

import ast
import json
from pathlib import Path

import pytest

from app.services.telemetry_era_adapters import load_chat_mappings
from app.services.telemetry_stamps import CHAT_TELEMETRY_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "telemetry" / "contracts"
CHAT_PY = REPO / "app" / "services" / "chat.py"

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
    read = {**mapping.fields, **{f"attributes.{name}": paths for name, paths in mapping.attributes.items()}}
    unreadable = {field: list(paths) for field, paths in read.items() if not any(_sent(path, contract) for path in paths)}

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


def test_attributes_are_only_mapped_on_stamped_versions():
    # Historical chat eras are read by the era adapters, which don't fill attributes.
    ignored = sorted(version for version, mapping in load_chat_mappings().items() if mapping.attributes and not version.startswith("chat.turn."))

    assert not ignored, f"{ignored} list attributes, but only stamped chat.turn.vN versions fill them. Move them there."


def _score_names_in_chat_py():
    """Names chat.py gives its Langfuse scores, e.g. score_current_trace(name="turn_outcome")."""
    return {
        keyword.value.value
        for node in ast.walk(ast.parse(CHAT_PY.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"score_current_trace", "score_current_span", "create_score"}
        for keyword in node.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    }


def test_chat_scores_match_the_contract():
    contract = _contracts()[CHAT_TELEMETRY_SCHEMA_VERSION]
    listed = {name for names in contract["scores"].values() for name in names}
    found = _score_names_in_chat_py()

    assert found == listed, (
        f"chat.py sends the scores {sorted(found)}, but {CHAT_TELEMETRY_SCHEMA_VERSION} lists {sorted(listed)}. "
        "A score added, renamed or removed is a new schema version: bump CHAT_TELEMETRY_SCHEMA_VERSION, add "
        "telemetry/contracts/chat.turn.vN.json with the change, and add the version to telemetry/mappings/chat.yaml."
    )
