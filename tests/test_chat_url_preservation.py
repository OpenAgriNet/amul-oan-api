"""Chat URL regressions: no live models, credentials, or farmer data required."""

import asyncio
import os
import random
import re
import socket
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services import chat
from app.services.chat_links import translation_parts


URLS = [
    "https://pmkisan.gov.in",
    "https://pmfby.gov.in",
    "http://example.org:8080/2026/apply?a=1&b=two%20words#section",
    "https://example.com/a.b/c?next=https%3A%2F%2Fexample.org%2Fx#part",
    "https://example.org/wiki/Policy_(India)",
    "https://example.org/a_(b_(c))",
    "HTTPS://EXAMPLE.COM/CaseSensitive",
    "https://example.org/ગુજરાતી",
    "https://bücher.de/path",
    "https://xn--bcher-kva.de/path",
    "pmkisan.gov.in",
    "www.example.com/apply",
    "example.co.uk/path?q=1#part",
]
MARKDOWN_LINKS = [
    "[https://pmkisan.gov.in](https://pmkisan.gov.in)",
    "[https://pmfby.gov.in](https://pmfby.gov.in)",
    "[Apply here](https://pmfby.gov.in)",
    "[Apply now. Official website](https://pmfby.gov.in)",
    "[Ready? Apply now!](https://pmfby.gov.in)",
    "[Dr. Patel portal](https://pmfby.gov.in)",
    "[Apply [official] now](https://pmfby.gov.in)",
    "[Apply here.\nNow](https://pmfby.gov.in)",
    '[Apply](https://pmfby.gov.in "Official. Website")',
    "[Details](https://example.org/wiki/Policy_(India))",
    r"[Apply](https://example.org/path\(one\))",
    "![Official portal](https://example.org/logo.png)",
]
LINKS = URLS + MARKDOWN_LINKS + ["<https://pmfby.gov.in>"]


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("URL regression tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def run_stream(monkeypatch, chunks, *, target="gu", fail=False):
    calls = []
    failures = []

    async def translator(*, text, **kwargs):
        calls.append(text)
        if fail:
            raise RuntimeError("translator unavailable")
        # Corrupt every Latin letter, not just the known '.in' suffix. A URL
        # or Markdown label survives only if it bypasses the translation model.
        translated = re.sub(r"[A-Za-z]", "જ", text)
        for index in range(0, len(translated), 7):
            yield translated[index:index + 7]

    class Step:
        def __init__(self, *args):
            pass

        def open(self):
            return None

        def failed(self, error):
            failures.append(str(error))

        def record(self, given, delivered):
            pass

        def close(self):
            pass

    monkeypatch.setattr(chat, "translate_text_stream_fast", translator)
    monkeypatch.setattr(chat, "_PostTranslation", Step)

    async def source():
        for chunk in chunks:
            yield chunk

    async def collect():
        stored = []
        output = "".join([part async for part in chat._stream_to_client(
            source(), translate_to=target, max_output_chars=None,
            execution=SimpleNamespace(), output_chunks=stored,
        )])
        assert output == "".join(stored), "trace output differs from delivered text"
        return output

    return asyncio.run(collect()), calls, failures


@pytest.mark.parametrize("link", LINKS)
def test_every_incoming_split_position_preserves_link(monkeypatch, link):
    text = "Find the official application website here: " + link + "\n\nBring documents."
    for split in range(1, len(text)):
        output, calls, failures = run_stream(monkeypatch, [text[:split], text[split:]])
        assert output.count(link) == 1, (split, output)
        assert calls, "surrounding prose was not translated"
        assert "Bring documents" not in output
        assert not failures
        if link in MARKDOWN_LINKS:
            assert not any("[" in part or "](" in part for part in calls), calls


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_two_scheme_links_in_repeated_small_chunks(monkeypatch, size):
    text = (
        "* **Official PM-KISAN website:** " + MARKDOWN_LINKS[0]
        + "\n\n* **Official crop insurance website:** " + MARKDOWN_LINKS[1]
        + "\n\nWould you like more information?"
    )
    chunks = [text[index:index + size] for index in range(0, len(text), size)]
    output, _, _ = run_stream(monkeypatch, chunks)
    assert output.count(MARKDOWN_LINKS[0]) == output.count(MARKDOWN_LINKS[1]) == 1


@pytest.mark.parametrize("seed", range(6))
def test_random_chunk_schedules_preserve_markdown(monkeypatch, seed):
    text = "Find the official resource here: " + "\n".join(MARKDOWN_LINKS)
    rng = random.Random(seed)
    chunks = []
    offset = 0
    while offset < len(text):
        size = rng.randint(1, 23)
        chunks.append(text[offset:offset + size])
        offset += size
    output, _, _ = run_stream(monkeypatch, chunks)
    for link in MARKDOWN_LINKS:
        assert output.count(link) == 1


@pytest.mark.parametrize("link", LINKS)
def test_end_of_stream_and_translation_failure(monkeypatch, link):
    text = "Find the details here: " + link
    output, _, failures = run_stream(monkeypatch, list(text), fail=True)
    assert output.count(link) == 1
    assert "Find the details here:" in output
    assert failures, "the prose fallback path was not exercised"


@pytest.mark.parametrize("link", LINKS)
def test_link_only_never_calls_translator(monkeypatch, link):
    output, calls, _ = run_stream(monkeypatch, list(link))
    assert output == link
    assert calls == []


def test_english_passthrough_is_unchanged(monkeypatch):
    text = "Apply at " + MARKDOWN_LINKS[0] + "."
    output, calls, _ = run_stream(monkeypatch, list(text), target=None)
    assert output == text
    assert calls == []


@pytest.mark.parametrize("text", [
    "Use 2.5 litres daily. Then wash the bowl.",
    "1. First item.\n2. Second item.\n\nA final paragraph.",
    "One sentence! A question? Yes.",
    "First sentence.\tSecond sentence.\r\nThird sentence.",
])
def test_sentence_collection_loses_no_source_characters(text):
    buffer = ""
    collected = []
    for char in text:
        complete, buffer = chat.extract_complete_sentences(buffer + char)
        collected.extend(complete)
    assert "".join(collected) + buffer == text


def test_earlier_sentence_flushes_while_link_is_incomplete():
    complete, tail = chat.extract_complete_sentences(
        "This ordinary sentence is complete. Here is [https://pmfby.gov."
    )
    assert "This ordinary sentence is complete." in "".join(complete)
    assert "[https://pmfby.gov." in tail


def test_domain_dot_waits_for_following_character():
    complete, tail = chat.extract_complete_sentences("Apply here: https://pmfby.gov.")
    assert complete == []
    complete, tail = chat.extract_complete_sentences(tail + "in. Next sentence.")
    assert "https://pmfby.gov.in." in "".join(complete)
    assert tail == "Next sentence."


def test_bracketed_prose_does_not_block_later_sentences():
    complete, tail = chat.extract_complete_sentences("[Note] Bring documents. Then apply.")
    assert "".join(complete) == "[Note] Bring documents. "
    assert tail == "Then apply."


@pytest.mark.parametrize("text, protected", [
    ("Visit https://example.org/apply. Next.", "https://example.org/apply"),
    ("Visit (https://example.org/apply).", "https://example.org/apply"),
    ("See https://example.org/wiki/Policy_(India).", "https://example.org/wiki/Policy_(India)"),
    ("Read <https://example.org/apply>.", "<https://example.org/apply>"),
    ("See [Apply now](https://example.org/apply?q=1#part).", "[Apply now](https://example.org/apply?q=1#part)"),
])
def test_protected_ranges_leave_sentence_punctuation_outside(text, protected):
    parts = translation_parts(text)
    assert "".join(part for part, _ in parts) == text
    assert protected in [part for part, verbatim in parts if verbatim]
