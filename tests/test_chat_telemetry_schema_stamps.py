import ast
import hashlib
import json
import re
import typing
from pathlib import Path

from app.models.telemetry import CanonicalTelemetryEvent
from app.services import telemetry_stamps
from app.services.telemetry_mapper import map_canonical_event_to_langfuse
from app.services.telemetry_stamps import (
    BACKGROUND_JOB,
    CHAT_TELEMETRY_SCHEMA_VERSION,
    CHAT_TELEMETRY_SERVICE,
    FRONTEND_EVENT,
    TURN_ROOT,
    forward_chat_telemetry_metadata,
)


REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "telemetry" / "contracts"
CONTRACT = CONTRACTS / CHAT_TELEMETRY_SCHEMA_VERSION / "chat.json"
NON_TURN_CONTRACT = CONTRACTS / CHAT_TELEMETRY_SCHEMA_VERSION / "non_turn.json"
STAMPS = {"amul.schema_version", "amul.service", "amul.release", "amul.kind"}
NEW_VERSION = (
    f"{CHAT_TELEMETRY_SCHEMA_VERSION} is the shape every pipeline sends (amul-telemetry telemetry/turn.v1.yaml). "
    "A key added or removed is a registry change there first; then change telemetry/contracts/"
    f"{CHAT_TELEMETRY_SCHEMA_VERSION}/chat.json and the calls in chat.py together."
)

# Key order, spacing and a "note" don't count.
RELEASED_CONTRACTS = {
    "chat.turn.v1": "47af4bd14b99f3c896e23c8bdb0c4218f4443e7990252042d65181d14a436193",
}

_GENERIC_NAMES = {"data", "id", "result", "score", "status", "time", "type", "value"}
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")


def _fingerprint(path):
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.pop("note", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _calls(path):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _keyword_names(calls, function):
    return {
        keyword.arg
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == function
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _observation_names(calls):
    """The constant names given to start_as_current_observation in a module."""
    return {
        keyword.value.value
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "start_as_current_observation"
        for keyword in call.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    }


def test_forward_telemetry_metadata_stamps_schema_service_and_release(monkeypatch, tmp_path):
    # No checkout here, so the build arg is used. A real .git would win over it.
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: tmp_path)
    monkeypatch.setenv("GIT_SHA", "test-release-sha")
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata() == {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "amul.service": "amul-oan-api",
        "amul.release": "test-release-sha",
    }


def test_forward_telemetry_metadata_marks_an_unknown_release(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: Path("missing-repository"))
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata()["amul.release"] == "unknown"


def test_forward_telemetry_metadata_reads_a_git_head_file(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("test-git-sha\n", encoding="utf-8")
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(telemetry_stamps, "_repository_root", lambda: tmp_path)
    telemetry_stamps.chat_telemetry_release.cache_clear()

    assert forward_chat_telemetry_metadata()["amul.release"] == "test-git-sha"


def test_turn_v1_contract_matches_what_chat_py_sends():
    """The contract lists exactly the amul.* keys chat.py builds (turn_v1_metadata at the
    root, turn_v1_fields when it closes), the steps it, translation.py and the suggestions
    task open, and its scores."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    chat_calls = _calls("app/services/chat.py")
    sent = STAMPS | {
        f"amul.{name}" for name in _keyword_names(chat_calls, "turn_v1_metadata") | _keyword_names(chat_calls, "turn_v1_fields")
    }

    assert contract["schema_version"] == CHAT_TELEMETRY_SCHEMA_VERSION == "turn.v1"
    assert contract["root"] == TURN_ROOT == "turn"
    assert sent == set(contract["metadata_keys"]), f"chat.py's metadata keys don't match the contract. {NEW_VERSION}"
    opened = {
        name
        for path in ("app/services/chat.py", "app/services/translation.py", "app/tasks/suggestions.py")
        for name in _observation_names(_calls(path))
    } - {TURN_ROOT}
    assert opened == set(contract["steps"]), f"the steps chat opens don't match the contract. {NEW_VERSION}"
    assert CHAT_TELEMETRY_SERVICE == "amul-oan-api"


def _score_names_in_chat_py():
    """Names chat.py gives its Langfuse scores, e.g. score_current_trace(name="turn_outcome")."""
    return {
        keyword.value.value
        for node in _calls("app/services/chat.py")
        if isinstance(node.func, ast.Attribute)
        and node.func.attr in {"score_current_trace", "score_current_span", "score_trace", "create_score"}
        for keyword in node.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    }


def test_chat_scores_match_the_contract():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert _score_names_in_chat_py() == set(contract["scores"]) == {"turn_outcome"}, (
        f"chat.py sends the scores {sorted(_score_names_in_chat_py())}, but the contract lists {contract['scores']}. "
        "turn.v1 records the outcome as amul.outcome and as the turn_outcome score; served_tier and pipeline_profile "
        "are metadata, not scores."
    )


def _non_turn_calls(function):
    """Every call to ``function`` (non_turn_v1_metadata or non_turn_root) in app/ and agents/."""
    paths = sorted([*REPO.glob("app/**/*.py"), *REPO.glob("agents/**/*.py")])
    return [
        call
        for path in paths
        for call in _calls(path.relative_to(REPO))
        if isinstance(call.func, ast.Name) and call.func.id == function
    ]


def _non_turn_sends():
    """kind -> job -> the amul.* keys each non_turn_v1_metadata call builds. The frontend
    writer's one call names no job: its event is the event's."""
    sent: dict[str, dict] = {}
    for call in _non_turn_calls("non_turn_v1_metadata"):
        kind = getattr(telemetry_stamps, call.args[0].id)
        job = next((kw.value.value for kw in call.keywords if kw.arg == "job" and isinstance(kw.value, ast.Constant)), None)
        sent.setdefault(kind, {})[job] = STAMPS | {f"amul.{kw.arg}" for kw in call.keywords}
    return sent


def test_non_turn_contract_matches_what_the_jobs_and_the_frontend_writer_send():
    """Every background job and frontend event is in telemetry/contracts/turn.v1/non_turn.json
    with exactly the amul.* keys its call builds; amul-telemetry checks the file against the shape."""
    contract = json.loads(NON_TURN_CONTRACT.read_text(encoding="utf-8"))
    sent = _non_turn_sends()

    assert contract["schema_version"] == CHAT_TELEMETRY_SCHEMA_VERSION
    assert set(contract) - {"schema_version", "note"} == set(sent) == {BACKGROUND_JOB, FRONTEND_EVENT}
    jobs = {job: set(entry["metadata_keys"]) for job, entry in contract[BACKGROUND_JOB].items()}
    assert jobs == sent[BACKGROUND_JOB], f"the background jobs don't match the contract. {NEW_VERSION}"
    (writer_keys,) = sent[FRONTEND_EVENT].values()
    events = set(typing.get_args(CanonicalTelemetryEvent.model_fields["event_name"].annotation))
    assert set(contract[FRONTEND_EVENT]) == events
    assert all(set(entry["metadata_keys"]) == writer_keys for entry in contract[FRONTEND_EVENT].values())


def test_non_turn_trace_fields_and_scores_match_the_contract():
    contract = json.loads(NON_TURN_CONTRACT.read_text(encoding="utf-8"))
    passed: dict[str, set] = {}
    for call in _non_turn_calls("non_turn_root"):
        passed.setdefault(call.args[0].id, set()).update(kw.arg for kw in call.keywords if kw.arg in ("session_id", "user_id"))

    # No turn caused a job here, so none passes a session or a user.
    assert passed["BACKGROUND_JOB"] == set()
    assert all(entry["trace_fields"] == [] for entry in contract[BACKGROUND_JOB].values())
    assert {field for entry in contract[FRONTEND_EVENT].values() for field in entry["trace_fields"]} == passed["FRONTEND_EVENT"]
    scored = {
        event: {mapped["score"]["name"]}
        for event in contract[FRONTEND_EVENT]
        if (mapped := map_canonical_event_to_langfuse(CanonicalTelemetryEvent(event_name=event))).get("score")
    }
    assert scored == {event: set(entry["scores"]) for event, entry in contract[FRONTEND_EVENT].items() if entry["scores"]}
    assert scored == {"feedback": {"user_feedback"}}


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


def test_names_are_specific_snake_case():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    names = [
        *(key.removeprefix("amul.") for key in contract["metadata_keys"]),
        *contract["trace_fields"],
        *contract["steps"],
        *(key.removeprefix("amul.") for step in contract["steps"].values() for key in step["metadata_keys"]),
        *contract["scores"],
        *contract["outcomes"],
    ]
    unclear = [name for name in names if not _SNAKE_CASE.fullmatch(name) or name in _GENERIC_NAMES]

    assert not unclear, f"Unclear names {unclear}."


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

    assert forward_chat_telemetry_metadata()["amul.release"] == "a" * 40


def test_the_image_build_passes_the_commit_for_the_release_stamp():
    # .dockerignore drops .git, so without this every prod trace says "unknown".
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build-and-push.yml"

    assert "GIT_SHA=${{ github.sha }}" in workflow.read_text(encoding="utf-8")


def test_the_image_is_checked_for_its_release_before_it_is_pushed():
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build-and-push.yml").read_text(
        encoding="utf-8"
    )
    build, check, push = (
        workflow.index(f"name: {name}")
        for name in ("Build Docker image", "Check the image's release stamp", "Push Docker image")
    )

    assert build < check < push
    assert "push: true" not in workflow[build:check]
    assert 'test "$release" = "$GITHUB_SHA"' in workflow[check:push]
    # The image that is checked and the one that is pushed are built the same way.
    assert workflow[build:check].count("GIT_SHA=${{ github.sha }}") == 1
    assert workflow[push:].count("GIT_SHA=${{ github.sha }}") == 1
