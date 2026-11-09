"""Keep chat links intact while sentence batching and translating prose."""

import re


_URL = re.compile(
    r"https?://[^\s<>\"`]+|www\.[^\s<>\"`]+|"
    r"(?<![\w@/])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"[a-z]{2,63}(?::[0-9]+)?(?:[/#?][^\s<>\"`]*)?",
    re.I,
)
_AUTOLINK = re.compile(r"<https?://[^\s<>]+>", re.I)


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if not escaped:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return index + 1
        escaped = char == "\\" and not escaped
    return None


def markdown_link_spans(text: str, *, streaming: bool = False) -> list[tuple[int, int]]:
    """Inline link ranges, including unfinished labels/destinations in a stream.

    The same ranges guard batching and bypass translation. Nested brackets and
    multiline labels remain together, as do balanced destination parentheses.
    """
    spans = []
    offset = 0
    while offset < len(text):
        start = text.find("[", offset)
        if start < 0:
            break
        if start and text[start - 1] == "\\":
            offset = start + 1
            continue
        label_end = _balanced_end(text, start, "[", "]")
        link_start = start - 1 if start and text[start - 1] == "!" else start
        if label_end is None or label_end == len(text):
            if streaming:
                spans.append((link_start, len(text)))
            break
        if text[label_end] != "(":
            offset = label_end
            continue
        end = _balanced_end(text, label_end, "(", ")")
        if end is None:
            if streaming:
                spans.append((link_start, len(text)))
            break
        spans.append((link_start, end))
        offset = end
    return spans


def link_spans(text: str) -> list[tuple[int, int]]:
    """URL and inline Markdown-link ranges, including balanced URL parentheses."""
    spans = markdown_link_spans(text)
    spans.extend(match.span() for match in _AUTOLINK.finditer(text))
    for match in _URL.finditer(text):
        start, end = match.span()
        # Sentence punctuation and Markdown delimiters are not part of a URL.
        while end > start:
            value = text[start:end]
            last = value[-1]
            if last in ".,!?;:" or any(
                last == close and value.count(close) > value.count(opening)
                for opening, close in [("(", ")"), ("[", "]"), ("{", "}")]
            ):
                end -= 1
            else:
                break
        if end > start:
            spans.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def translation_parts(text: str) -> list[tuple[str, bool]]:
    """(text, verbatim) pieces. Keep whole Markdown links, including their labels.

    Links bypass the model entirely, so no placeholder can leak and a translator
    cannot invent a different destination. Whitespace next to a link is also
    copied so independently translated prose cannot glue itself to the address.
    """
    spans = link_spans(text)
    if not spans:
        return [(text, False)]
    parts = []

    def prose(value):
        if not value:
            return
        match = re.fullmatch(r"(\s*)(.*?)(\s*)", value, re.S)
        for piece, verbatim in zip(match.groups(), (True, False, True)):
            if piece:
                parts.append((piece, verbatim))

    offset = 0
    for start, end in spans:
        prose(text[offset:start])
        parts.append((text[start:end], True))
        offset = end
    prose(text[offset:])
    return parts
