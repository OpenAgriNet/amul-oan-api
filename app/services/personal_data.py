"""Hide phone and Aadhaar numbers in text that people read back (#517).

The raw test case store keeps what testers sent, unmasked. Anything that reads it
back out goes through ``mask_numbers`` first, so a farmer's phone or Aadhaar
number in a question doesn't reach a screen.

What counts as one of those numbers:

- 7 or more digits in a row (a phone, an Aadhaar, an account number)
- digits in 2 to 4 groups split by a space or a hyphen that add up to a phone or
  an Aadhaar number (10 to 13 digits, one group of 4 or more), like
  ``98765 43210``, ``+91 98765-43210`` or ``1234 5678 9012``

Gujarati and Devanagari digits count the same. A masked number keeps its last 4
digits, like ``******3210``. Amounts up to 6 digits, dates, fat and SNF values,
litres and short codes stay as they are.

Farmer codes are short and look like any other word, so a pattern can't find
them on their own; the labeller (#488) handles them with the words around them.
"""
import re

from helpers.utils import to_ascii_digits

# \d matches any Unicode decimal digit for str patterns.
_NUMBER = re.compile(r"(?<!\d)\+?\d+(?:[ -]\d+){0,3}(?!\d)")
_GROUP_SPLIT = re.compile(r"[ -]")
MASK = "******"


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
    return MASK + to_ascii_digits(digits[-4:])


def mask_numbers(text: str) -> str:
    """``text`` with phone and Aadhaar numbers cut down to their last 4 digits."""
    return _NUMBER.sub(_hide, text) if text else text
