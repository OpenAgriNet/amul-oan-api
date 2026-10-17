"""Prefer Amul-provided local-script names over English transliterations."""

from __future__ import annotations

import re


def prefer_local_name(
    local: str | None,
    english: str | None = None,
) -> str | None:
    """Return a non-empty local (e.g. Gujarati) name when present, else English.

    Used for farmer-facing display so post-translation does not reinvent proper
    names Amul already localized in the API payload.
    """
    for candidate in (local, english):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def speakable_name(value: str | None) -> str | None:
    if not value:
        return value
    words = re.sub(r"[-_]", " ", re.sub(r"\d+", "", value)).split()
    if not words:
        return value
    if not any(ch.islower() for word in words for ch in word):
        words = [word.capitalize() for word in words]
    return " ".join(words)
