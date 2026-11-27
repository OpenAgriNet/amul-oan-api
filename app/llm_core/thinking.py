"""Gemma's thinking channel, kept out of the agent's answer (#612).

Prod's Gemma engine runs without vLLM's reasoning parser, so when the model
writes its thinking channel, the markers come back in the answer's text:
``<|channel>thought ... <channel|>``, or a lone ``<channel|>``. Left in, voice
sends them to the output translator as a batch of their own, and the caller
hears it ask for text to translate (4 of 8,327 answers, 11-18 Nov).
"""
from __future__ import annotations

import re
from typing import Iterable

OPEN = "<|channel>"
CLOSE = "<channel|>"
_BLOCK = re.compile(re.escape(OPEN) + r".*?(?:" + re.escape(CLOSE) + r"|\Z)", re.DOTALL)


def strip_thinking(text: str) -> str:
    """The text without thinking blocks or stray markers."""
    return _BLOCK.sub("", text).replace(CLOSE, "")


def strip_thinking_from_messages(messages: Iterable) -> None:
    """Strip the text parts of ``messages`` in place, as they are saved to the history."""
    for message in messages:
        for part in getattr(message, "parts", ()):
            if getattr(part, "part_kind", None) == "text" and (OPEN in part.content or CLOSE in part.content):
                part.content = strip_thinking(part.content)


def _marker_prefix_at_end(text: str) -> int:
    """How many trailing characters of ``text`` could start a marker that the
    next chunk completes."""
    for size in range(min(len(text), len(OPEN) - 1), 0, -1):
        tail = text[-size:]
        if OPEN.startswith(tail) or CLOSE.startswith(tail):
            return size
    return 0


class ThinkingFilter:
    """Streamed text with thinking blocks and stray markers taken out. A marker
    can arrive split across chunks, so text that may start one is held back
    until the next chunk, or :meth:`flush`."""

    def __init__(self) -> None:
        self._held = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        text, self._held = self._held + chunk, ""
        out: list[str] = []
        while text:
            if self._inside:
                end = text.find(CLOSE)
                if end < 0:
                    # Thinking text is dropped; only a marker start is kept.
                    self._held = text[len(text) - _marker_prefix_at_end(text):]
                    break
                text, self._inside = text[end + len(CLOSE):], False
                continue
            starts = [i for i in (text.find(OPEN), text.find(CLOSE)) if i >= 0]
            if not starts:
                keep = _marker_prefix_at_end(text)
                out.append(text[: len(text) - keep])
                self._held = text[len(text) - keep:]
                break
            at = min(starts)
            out.append(text[:at])
            if text.startswith(OPEN, at):
                text, self._inside = text[at + len(OPEN):], True
            else:
                text = text[at + len(CLOSE):]
        return "".join(out)

    def flush(self) -> str:
        """What is still held when the stream ends: text, unless inside a block."""
        held, self._held = self._held, ""
        return "" if self._inside else held
