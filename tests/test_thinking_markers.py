"""Gemma's thinking markers are taken out of the streamed text (#612). The seam
test in test_voice_sink.py covers the translator and the history."""
import pytest

from app.llm_core.thinking import ThinkingFilter

ANSWER = "Wishing you healthy animals and good milk production."


def _streamed(chunks):
    thinking = ThinkingFilter()
    return "".join(thinking.feed(chunk) for chunk in chunks) + thinking.flush()


@pytest.mark.parametrize("text, expected", [
    # The three shapes prod's voice answers had, 11-18 Nov.
    (f"{ANSWER}\n\n<|channel>thought\n<channel|>", f"{ANSWER}\n\n"),
    (f"{ANSWER}\n\n<channel|>", f"{ANSWER}\n\n"),
    (f"<|channel>thought\n<channel|>{ANSWER}", ANSWER),
])
def test_a_marker_split_into_single_characters_is_still_taken_out(text, expected):
    assert _streamed(list(text)) == expected


@pytest.mark.parametrize("text", ["Keep the pH < 7.", "Text that ends like a marker start <chan"])
def test_text_without_markers_comes_out_whole(text):
    assert _streamed([text]) == text
    assert _streamed(list(text)) == text
