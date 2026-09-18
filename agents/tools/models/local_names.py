"""Prefer Amul-provided local-script names over English transliterations."""

from __future__ import annotations


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
