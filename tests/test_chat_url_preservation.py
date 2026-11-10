"""Review cases for amul-oan-api #581: chat sentence batching + post-translation.

The fake translator does what the real one does to whitespace: the prompt carries
``text.strip()`` (translation._build_translation_instruction), so leading and trailing
whitespace of a batch never reaches the model. It returns the stripped text unchanged,
so any layout difference in the output comes from the pipeline, not the model.
"""

import asyncio
import os
import random
import re
import socket
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services import chat


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("no network in these tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def stream(monkeypatch, text, size_range, seed):
    calls = []

    async def translator(*, text, **kwargs):
        calls.append(text)
        if text.strip():
            yield text.strip()

    class Step:
        def __init__(self, *a): pass
        def open(self): return None
        def failed(self, e): pass
        def record(self, given, delivered): pass
        def close(self): pass

    monkeypatch.setattr(chat, "translate_text_stream_fast", translator)
    monkeypatch.setattr(chat, "_PostTranslation", Step)
    rnd = random.Random(seed)
    chunks, i = [], 0
    while i < len(text):
        n = rnd.randint(*size_range)
        chunks.append(text[i:i + n])
        i += n

    async def source():
        for c in chunks:
            yield c

    async def collect():
        return "".join([p async for p in chat._stream_to_client(
            source(), translate_to="gu", max_output_chars=None,
            execution=SimpleNamespace(), output_chunks=[],
        )])

    return asyncio.run(collect()), calls


SCHEDULES = [((1, 1), 0)] + [((1, 4), s) for s in range(5)] + [((2, 9), s) for s in range(5)]

PARAGRAPHS = (
    "Mastitis is an infection of the udder that reduces milk yield and quality in dairy animals.\n\n"
    "Keep the shed clean and dry every day, and wash the udder before milking.\n\n"
    "If you see swelling or clots in the milk, contact your nearest veterinary doctor quickly."
)
BOLD_SECTIONS = (
    "**Feeding:** Give green fodder and clean water to the animal twice a day without fail.\n\n"
    "**Housing:** Keep the shed dry, shaded and well ventilated during the summer months.\n\n"
    "**Health:** Vaccinate the animal on time as per the schedule from the doctor."
)
NUMBERED = (
    "To apply for the scheme, follow these steps:\n"
    "1. Visit the nearest milk society office.\n"
    "2. Submit your Aadhaar card and bank details.\n"
    "3. Wait for the confirmation message."
)


def lines(text):
    return [line.strip() for line in text.split("\n") if line.strip()]


# A. Layout: every non-empty source line comes out as its own line.
@pytest.mark.parametrize("size_range, seed", SCHEDULES)
@pytest.mark.parametrize("name, text", [
    ("paragraphs", PARAGRAPHS), ("bold_sections", BOLD_SECTIONS), ("numbered", NUMBERED),
])
def test_line_breaks_survive_batching(monkeypatch, name, text, size_range, seed):
    output, _ = stream(monkeypatch, text, size_range, seed)
    assert lines(output) == lines(text), repr(output)


# B. A numbered marker never ends up glued to its item ("1.Visit").
@pytest.mark.parametrize("size_range, seed", SCHEDULES)
def test_numbered_marker_keeps_its_space(monkeypatch, size_range, seed):
    output, _ = stream(monkeypatch, NUMBERED, size_range, seed)
    assert not re.search(r"(?m)^\d+\.\S", output), repr(output)


URL_TEXTS = [
    ("You can check your PM-KISAN status on pmkisan.gov.in using your registered mobile number. "
     "For crop insurance, visit https://pmfby.gov.in/ and enter your details.\n\nContact the helpline for more help.",
     ["pmkisan.gov.in", "https://pmfby.gov.in/"]),
    ("Read the [PM-KISAN portal](https://pmkisan.gov.in/) page. Then call 1962. Also see www.amul.coop for more.",
     ["https://pmkisan.gov.in/", "www.amul.coop"]),
    ("Visit https://pmfby.gov.in/. Done.\n1. Open pmkisan.gov.in on your phone.\n2. Enter details.",
     ["https://pmfby.gov.in/", "pmkisan.gov.in"]),
]


# C. The original bug: no translator call ever receives a piece of a URL.
@pytest.mark.parametrize("size_range, seed", SCHEDULES)
@pytest.mark.parametrize("text, urls", URL_TEXTS)
def test_no_call_gets_a_url_fragment(monkeypatch, text, urls, size_range, seed):
    output, calls = stream(monkeypatch, text, size_range, seed)
    for url in urls:
        host = re.sub(r"^https?://", "", url).split("/")[0]
        first_label = host.split(".")[0] + "."
        for call in calls:
            if first_label in call:
                assert host in call, (url, call)
        assert url in output


# D. A sentence that contains a URL reaches the translator in one call, so the
#    model can order the words around it (Gujarati is SOV).
@pytest.mark.parametrize("size_range, seed", SCHEDULES)
def test_sentence_with_url_is_translated_whole(monkeypatch, size_range, seed):
    text = "Visit pmkisan.gov.in for details about your installment. Keep your Aadhaar ready."
    _, calls = stream(monkeypatch, text, size_range, seed)
    assert any("Visit pmkisan.gov.in for details about your installment." in c for c in calls), calls


# E. An unclosed "[" in prose does not hold back the rest of the answer.
def test_unclosed_bracket_does_not_stop_batching():
    complete, tail = chat.extract_complete_sentences(
        "Use the [note without close for this. Sentence two is here. Sentence three"
    )
    assert "Sentence two is here." in "".join(complete), (complete, tail)


# F. Old behaviour kept: decimals and abbreviation-free sentence ends.
@pytest.mark.parametrize("text, expected", [
    ("Use 2.5 litres daily. Then wash", ["Use 2.5 litres daily. "]),
    ("One! Two? Three. Four", ["One! ", "Two? ", "Three. "]),
])
def test_basic_boundaries(text, expected):
    complete, _ = chat.extract_complete_sentences(text)
    assert complete == expected


@pytest.mark.parametrize("size_range, seed", SCHEDULES)
@pytest.mark.parametrize("url", [
    "https://pmkisan.gov.in",
    "https://pmfby.gov.in/",
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
])
def test_url_and_surrounding_sentence_share_a_translation_call(monkeypatch, url, size_range, seed):
    sentence = f"Visit {url} for details about the application."
    output, calls = stream(monkeypatch, sentence + " Bring your documents.", size_range, seed)
    assert any(sentence in call for call in calls), calls
    assert url in output


def test_domain_dot_waits_for_following_character():
    complete, tail = chat.extract_complete_sentences("Apply here: https://pmfby.gov.")
    assert complete == []
    complete, tail = chat.extract_complete_sentences(tail + "in. Next sentence.")
    assert "https://pmfby.gov.in." in "".join(complete)
    assert tail == "Next sentence."


def test_earlier_sentence_flushes_while_later_url_is_incomplete():
    complete, tail = chat.extract_complete_sentences(
        "This sentence is complete. Here is https://pmfby.gov."
    )
    assert "This sentence is complete." in "".join(complete)
    assert "https://pmfby.gov." in tail


@pytest.mark.parametrize("text", [
    "Use 2.5 litres daily. Then wash the bowl.",
    "1. First item.\n2. Second item.\n\nA final paragraph.",
    "One sentence! A question? Yes.",
])
def test_sentence_collection_loses_no_source_characters(text):
    buffer = ""
    collected = []
    for char in text:
        complete, buffer = chat.extract_complete_sentences(buffer + char)
        collected.extend(complete)
    assert "".join(collected) + buffer == text
