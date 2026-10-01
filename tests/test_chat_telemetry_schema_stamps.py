import ast
import hashlib
import json
from pathlib import Path

from app.services import telemetry_stamps
from app.services.telemetry_stamps import (
    CHAT_TELEMETRY_SCHEMA_VERSION,
    CHAT_TELEMETRY_SERVICE,
    CHAT_TURN_V1_ROOT,
    chat_turn_v1_input,
    chat_turn_v1_metadata,
    forward_chat_telemetry_metadata,
)


CONTRACTS = Path(__file__).resolve().parents[1] / "telemetry" / "contracts"
CONTRACT = CONTRACTS / f"{CHAT_TELEMETRY_SCHEMA_VERSION}.json"
NEW_VERSION = (
    f"{CHAT_TELEMETRY_SCHEMA_VERSION} can't change once released, so this needs a new schema version: "
    "bump CHAT_TELEMETRY_SCHEMA_VERSION in app/services/telemetry_stamps.py, copy "
    f"telemetry/contracts/{CHAT_TELEMETRY_SCHEMA_VERSION}.json to the new version's file and make the change "
    f"there, and add the new version to telemetry/mappings/chat.yaml (it can extend {CHAT_TELEMETRY_SCHEMA_VERSION})."
)

# Key order, spacing and a "note" don't count.
RELEASED_CONTRACTS = {
    "chat.turn.v1": "47af4bd14b99f3c896e23c8bdb0c4218f4443e7990252042d65181d14a436193",
}


def _fingerprint(path):
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.pop("note", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_forward_telemetry_metadata_stamps_schema_service_and_release(monkeypatch, tmp_path):
    # No checkout here, so the build arg is used. A real .git would win over it.
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: tmp_path)
    monkeypatch.setenv("GIT_SHA", "test-release-sha")
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata() == {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "service": "amul-oan-api",
        "release": "test-release-sha",
    }


def test_forward_telemetry_metadata_marks_an_unknown_release(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: Path("missing-repository"))
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata()["release"] == "unknown"


def test_forward_telemetry_metadata_reads_a_git_head_file(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("test-git-sha\n", encoding="utf-8")
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: tmp_path)
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata()["release"] == "test-git-sha"


def test_chat_turn_v1_contract_matches_what_chat_py_sends():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    chat_source = Path(__file__).resolve().parents[1] / "app" / "services" / "chat.py"
    tree = ast.parse(chat_source.read_text(encoding="utf-8"))

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    metadata_call = next(call for call in calls if call.func.id == "chat_turn_v1_metadata")
    input_call = next(call for call in calls if call.func.id == "chat_turn_v1_input")
    metadata_keys = {keyword.arg for keyword in metadata_call.keywords}
    input_keys = {keyword.arg for keyword in input_call.keywords}

    assert contract["schema_version"] == CHAT_TELEMETRY_SCHEMA_VERSION
    assert contract["root"] == CHAT_TURN_V1_ROOT
    assert set(contract["metadata"]["required"]) - {
        "amul.schema_version", "service", "release"
    } == metadata_keys, f"chat.py's metadata keys don't match the contract. {NEW_VERSION}"
    assert set(contract["trace_input"]["required"]) == input_keys, (
        f"chat.py's trace input keys don't match the contract. {NEW_VERSION}"
    )
    assert CHAT_TELEMETRY_SERVICE == "amul-oan-api"


def test_released_contracts_never_change():
    for version, fingerprint in RELEASED_CONTRACTS.items():
        path = CONTRACTS / f"{version}.json"
        assert path.exists(), f"{version} is released: keep telemetry/contracts/{version}.json, old traces follow it."
        # Compared as a bool so a failure doesn't print the new fingerprint to paste in.
        unchanged = _fingerprint(path) == fingerprint
        assert unchanged, (
            f"telemetry/contracts/{version}.json is released and can't change: traces already in Langfuse "
            f"follow it. Undo the edit and put the change in a new version. {NEW_VERSION}"
        )


def test_forward_telemetry_metadata_reads_a_packed_branch_ref(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{'f' * 40} refs/heads/other\n"
        f"{'a' * 40} refs/heads/main\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: tmp_path)
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata()["release"] == "a" * 40
