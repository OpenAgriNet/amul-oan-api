"""Stable provenance stamps for telemetry emitted by this service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Chat and voice send turn.v1: the one shape every pipeline is sent in
# (amul-telemetry telemetry/turn.v1.yaml). The root is named "turn", every
# trace-level key is flat under amul.*, and telemetry/contracts/turn.v1/ lists
# what each surface sends. One constant per surface, as amul-telemetry's contract
# check reads them.
CHAT_TELEMETRY_SCHEMA_VERSION = "turn.v1"
VOICE_TELEMETRY_SCHEMA_VERSION = "turn.v1"
CHAT_TELEMETRY_SERVICE = "amul-oan-api"
TURN_ROOT = "turn"
TURN_KIND = "turn"
# The traces that are not turns, each named after its kind (turn.v1's amul.kind).
BACKGROUND_JOB = "background_job"
FRONTEND_EVENT = "frontend_event"


@lru_cache(maxsize=1)
def chat_telemetry_release() -> str:
    """Return the deployed Git revision without a Git executable or LANGFUSE_RELEASE."""

    return _read_git_head(_repository_root()) or os.getenv("GIT_SHA") or "unknown"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_git_head(repository: Path) -> str | None:
    git_entry = repository / ".git"
    if git_entry.is_dir():
        git_dir = git_entry
    elif git_entry.is_file():
        prefix = "gitdir:"
        pointer = git_entry.read_text(encoding="utf-8").strip()
        if not pointer.startswith(prefix):
            return None
        git_dir = (repository / pointer.removeprefix(prefix).strip()).resolve()
    else:
        return None

    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref: "):
        return head or None
    ref = head.removeprefix("ref: ")
    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    # After git gc the branch may only be listed in packed-refs.
    try:
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        sha, _, name = line.partition(" ")
        if name == ref:
            return sha or None
    return None


def forward_chat_telemetry_metadata() -> dict[str, str]:
    """The stamps on every trace this service sends: which shape, which service, which build."""

    return {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "amul.service": CHAT_TELEMETRY_SERVICE,
        "amul.release": chat_telemetry_release(),
    }


def turn_v1_metadata(**fields: Any) -> dict[str, str]:
    """The trace-level metadata of a turn root as turn.v1 names it: the stamps, the
    kind, and each field under ``amul.<name>``. A field without a value is left out,
    so a key is present only when it is known. The contract tests read the field
    names from the calls in chat.py and app/voice.
    """

    return {**forward_chat_telemetry_metadata(), "amul.kind": TURN_KIND, **turn_v1_fields(**fields)}


def non_turn_v1_metadata(kind: str, **fields: Any) -> dict[str, str]:
    """The trace-level metadata of a trace that is not a turn (a background job or a
    frontend event), as turn.v1 names it: the stamps, the kind, and each field under
    ``amul.<name>``, left out when it has no value. The non-turn contract test reads the
    field names from these calls, per job and event.
    """

    return {**forward_chat_telemetry_metadata(), "amul.kind": kind, **turn_v1_fields(**fields)}


def turn_v1_fields(**fields: Any) -> dict[str, str]:
    """Fields written to the root after it opened (the outcome, the served tier), as ``amul.<name>``."""

    return {f"amul.{name}": str(value) for name, value in fields.items() if value not in (None, "")}
