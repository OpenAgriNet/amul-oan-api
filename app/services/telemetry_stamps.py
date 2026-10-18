"""Stable provenance stamps for telemetry emitted by this service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Chat sends turn.v1: the one shape every pipeline is sent in (amul-telemetry
# telemetry/turn.v1.yaml). The root is named "turn", every trace-level key is
# flat under amul.*, and telemetry/contracts/turn.v1.json lists what is sent.
CHAT_TELEMETRY_SCHEMA_VERSION = "turn.v1"
CHAT_TELEMETRY_SERVICE = "amul-oan-api"
TURN_ROOT = "turn"
TURN_KIND = "turn"

VOICE_TELEMETRY_SCHEMA_VERSION = "voice.turn.v1"
# Voice turns served from this repo are stamped with this service, not
# voice-oan-api's, so they can be told apart from turns served there.
VOICE_TELEMETRY_SERVICE = CHAT_TELEMETRY_SERVICE


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
    """The stamps on every chat trace: which shape, which service, which build."""

    return {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "amul.service": CHAT_TELEMETRY_SERVICE,
        "amul.release": chat_telemetry_release(),
    }


def forward_voice_telemetry_metadata() -> dict[str, str]:
    return {
        "amul.schema_version": VOICE_TELEMETRY_SCHEMA_VERSION,
        "service": VOICE_TELEMETRY_SERVICE,
        "release": chat_telemetry_release(),
    }


def turn_v1_metadata(**fields: Any) -> dict[str, str]:
    """The trace-level metadata of a chat turn root as turn.v1 names it: the stamps,
    the kind, and each field under ``amul.<name>``. A field without a value is left
    out, so a key is present only when it is known. The contract test reads the
    field names from the calls in chat.py.
    """

    return {**forward_chat_telemetry_metadata(), "amul.kind": TURN_KIND, **turn_v1_fields(**fields)}


def turn_v1_fields(**fields: Any) -> dict[str, str]:
    """Fields written to the root after it opened (the outcome, the served tier), as ``amul.<name>``."""

    return {f"amul.{name}": str(value) for name, value in fields.items() if value not in (None, "")}
