"""Stable provenance stamps for telemetry emitted by this service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

CHAT_TELEMETRY_SCHEMA_VERSION = "chat.turn.v1"
CHAT_TELEMETRY_SERVICE = "amul-oan-api"
CHAT_TURN_V1_ROOT = "chat.translation"


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
    return {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "service": CHAT_TELEMETRY_SERVICE,
        "release": chat_telemetry_release(),
    }


def chat_turn_v1_metadata(**fields: str) -> dict[str, str]:
    """The complete fixed metadata shape emitted for a chat.turn.v1 root."""

    return {**forward_chat_telemetry_metadata(), **fields}


def chat_turn_v1_input(**fields: Any) -> dict[str, Any]:
    """The trace input shape for chat.turn.v1."""

    return dict(fields)
