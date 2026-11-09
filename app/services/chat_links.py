"""Keep chat links intact while sentence batching and translating prose."""

import re


_URL = re.compile(
    r"https?://[^\s<>\"`]+|www\.[^\s<>\"`]+|"
    r"(?<![\w@/])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"[a-z]{2,63}(?::[0-9]+)?(?:[/#?][^\s<>\"`]*)?",
    re.I,
)
_LINK_START = re.compile(r"!?\[[^\]\n]*\]\(")
_OPEN_LABEL = re.compile(r"!?\[[^\]\n]*(?:\])?$")
_AUTOLINK = re.compile(r"<https?://[^\s<>]+>", re.I)


def link_spans(text: str, *, streaming: bool = False) -> list[tuple[int, int]]:
    """URL and inline Markdown-link ranges, including balanced URL parentheses.

    While collecting a stream, an unfinished label/destination is protected too:
    a closing bracket or parenthesis can arrive in a later model chunk.
    """
    spans = [match.span() for match in _AUTOLINK.finditer(text)]
    for match in _LINK_START.finditer(text):
        depth = 1
        end = match.end()
        escaped = False
        while end < len(text) and depth:
            char = text[end]
            if not escaped:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            escaped = char == "\\" and not escaped
            end += 1
        if depth == 0:
            spans.append((match.start(), end))
        elif streaming:
            spans.append((match.start(), len(text)))
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
    if streaming:
        match = _OPEN_LABEL.search(text)
        if match:
            spans.append(match.span())
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def protect_sentence_boundaries(text: str, sentences: list[str]) -> list[str]:
    """Merge punctuation splits inside links or the stream's unfinished word."""
    spans = link_spans(text, streaming=True)
    # Even before a token looks like a URL ("pmki"), its next chunk may add a
    # domain suffix. Do not commit any boundary inside the final lexical token.
    final_word = re.search(r"\S+$", text)
    if final_word:
        spans.append(final_word.span())
    result = []
    pending = ""
    offset = 0
    for sentence in sentences:
        pending += sentence
        offset += len(sentence)
        if not any(start < offset < end for start, end in spans):
            result.append(pending)
            pending = ""
    if pending:
        result.append(pending)
    return result


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
