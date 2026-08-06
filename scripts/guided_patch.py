"""Guided-decoding helper for the bare-gemma4 bench.

Ports OpenAgriNet/bharat-oan-api#142 (Gautam-Rajeev) to the amul stack, with the
shared character set EXTENDED — #142's set omits the degree sign, so a
constrained answer cannot write "25°C" and every weather answer is silently
mangled. Additions over #142 are marked below.

Verified on this box against vLLM 0.15.1:
  - extra_body {"structured_outputs": {"regex": ...}}  -> HONORED
  - extra_body {"guided_regex": ...}                   -> SILENTLY IGNORED
  - tool_calls are byte-identical with and without the constraint
"""
from pydantic_ai.models.openai import OpenAIChatModelSettings

# Unicode block per language, exactly as in #142.
_SCRIPT_RANGES = {
    "kn": "ಀ-೿",  # Kannada
    "ta": "஀-௿",  # Tamil
    "ml": "ഀ-ൿ",  # Malayalam
    "te": "ఀ-౿",  # Telugu
    "bn": "ঀ-৿",  # Bengali
    "as": "ঀ-৿",  # Assamese (shares the Bengali block)
    "gu": "઀-૿",  # Gujarati
    "hi": "ऀ-ॿ",  # Devanagari (NOT in #142; added for our Hindi traffic)
}

# #142's shared set: escaped tab/newline/CR (raw control bytes crash xgrammar's
# EBNF parser) + printable ASCII + danda + dashes/quotes/bullet/rupee.
_SHARED_142 = "\\t\\n\\r -~" "।॥" "–—‘’“”•₹"

# EXTENSIONS over #142 — each one is a character our answers actually emit.
#   ° degree      -> "25°C", every weather answer
#   … ellipsis    -> truncation in voice
#   × multiply    -> dosage "2 × 5 ml"
#   ≥ ≤      -> thresholds in advisory text
#   → ←      -> arrows in chat formatting
#   ± plus-minus  -> tolerance ranges
#     NBSP        -> emitted by the model around units
#   ½¼¾ -> vulgar fractions, common in feed quantities
_SHARED_EXTRA = "°…×≥≤→←± ½¼¾"

SHARED_CHARS = _SHARED_142 + _SHARED_EXTRA


def guided_settings(lang_code, extended=True):
    """Return OpenAIChatModelSettings constraining output to the target script.

    extended=False reproduces #142 verbatim (no degree sign) so the bench can
    measure what the missing characters actually cost.
    Returns None for an unmapped language -> unconstrained, same as #142.
    """
    script_range = _SCRIPT_RANGES.get((lang_code or "").lower())
    if not script_range:
        return None
    shared = SHARED_CHARS if extended else _SHARED_142
    pattern = f"^[{script_range}{shared}]*$"
    return OpenAIChatModelSettings(
        extra_body={"structured_outputs": {"regex": pattern}}
    )
