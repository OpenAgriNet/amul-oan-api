"""Guard against TranslateGemma hallucinating on degenerate fragments.

A streaming chunk boundary can isolate a pure-markdown token such as ``**`` as its
own translation call. TranslateGemma (greedy/temp-0) does not echo ``**`` — it
free-generates an unrelated canned English paragraph ("Here's a breakdown of the
key differences… Approach 1… Approach 2…"), which then gets concatenated into the
final answer. ``_is_untranslatable_fragment`` short-circuits those fragments so we
never call the model with nothing to translate.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import (
    _is_untranslatable_fragment,
    translate_text,
    translate_text_stream_fast,
)


@pytest.mark.parametrize("fragment", ["**", "*", "***", "---", ":", "•", "( )", "  -  "])
def test_pure_markup_is_untranslatable(fragment):
    assert _is_untranslatable_fragment(fragment) is True


@pytest.mark.parametrize("fragment", ["**bold**", "* point", "hello", "123", "ગાય", "5%"])
def test_real_content_is_translatable(fragment):
    assert _is_untranslatable_fragment(fragment) is False


@pytest.mark.asyncio
async def test_translate_text_returns_markup_verbatim_without_model_call():
    # If the guard fails, this would hit the network / TranslateGemma endpoint.
    assert await translate_text("**", "english", "gujarati") == "**"


@pytest.mark.asyncio
async def test_stream_yields_markup_verbatim_without_model_call():
    chunks = [c async for c in translate_text_stream_fast("**", "english", "gujarati")]
    assert chunks == ["**"]
