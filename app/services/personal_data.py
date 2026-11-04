"""Hide phone and Aadhaar numbers, OTPs and farmer codes in test case text.

The raw test case store keeps what testers sent, unmasked. Anything that reads it
back out masks it first: the review list (#517) with ``mask_numbers``, and the
labeller (#488) with ``mask_personal_data`` before any LLM sees the text.

What counts as a phone or Aadhaar number:

- 7 or more digits in a row (a phone, an Aadhaar, an account number)
- digits in 2 to 4 groups split by a space or a hyphen that add up to a phone or
  an Aadhaar number (10 to 13 digits, one group of 4 or more), like
  ``98765 43210``, ``+91 98765-43210`` or ``1234 5678 9012``

Gujarati and Devanagari digits count the same. A masked number keeps its last 4
digits, like ``******3210``. Amounts up to 6 digits, dates, fat and SNF values,
litres and cycles like 2025-26 stay as they are.

Farmer codes and OTPs are short and look like any other word, so only the word in
front of them gives them away: ``farmer code 1234``, ``society code: NA4192``,
``OTP is 4821`` or ``સભાસદ કોડ ૧૨૩૪`` become ``[code]``.

No app imports, so the labeller script runs without the app's settings.
"""
import re
import unicodedata

# \d matches any Unicode decimal digit for str patterns.
_NUMBER = re.compile(r"(?<!\d)\+?\d+(?:[ -]\d+){0,3}(?!\d)")
_GROUP_SPLIT = re.compile(r"[ -]")
MASK = "******"

# The word that names the code, then optional fillers ("is", ":", "no."), then the
# code itself: letters, digits, "/" and "-", with at least one digit. Both parts can
# take "-", so both are bounded: unbounded, a long run of dashes made a match take
# seconds, and it holds the GIL while it runs.
_CODE = re.compile(
    r"(?i)((?:\b(?:otp\b|code\b|(?:farmer|member)\s+(?:id\b|number\b|no\b\.?))|કોડ|ઓટીપી|સભાસદ\s+નંબર)"
    r"(?:\s*(?:is|no\.?|number|[:#=-])){0,3}\s*)"
    r"((?:[A-Za-z/-]|\d){0,32}\d(?:[A-Za-z/-]|\d){0,32})"
)
CODE_MASK = "[code]"


def _ascii(digits: str) -> str:
    return "".join(str(unicodedata.decimal(char)) for char in digits)


def _hide(match: re.Match[str]) -> str:
    text = match.group(0)
    groups = _GROUP_SPLIT.split(text.lstrip("+"))
    digits = "".join(groups)
    if len(groups) == 1:
        private = len(digits) >= 7
    else:
        private = 10 <= len(digits) <= 13 and max(len(group) for group in groups) >= 4
    if not private:
        return text
    return MASK + _ascii(digits[-4:])


def mask_numbers(text: str) -> str:
    """``text`` with phone and Aadhaar numbers cut down to their last 4 digits."""
    return _NUMBER.sub(_hide, text) if text else text


def mask_codes(text: str) -> str:
    """``text`` with OTPs and farmer, member, society and union codes hidden."""
    return _CODE.sub(lambda match: match.group(1) + CODE_MASK, text) if text else text


def mask_personal_data(text: str) -> str:
    """Everything this module hides: codes, then phone and Aadhaar numbers."""
    return mask_numbers(mask_codes(text))
