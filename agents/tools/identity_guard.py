"""One place that decides whether an identifier the model supplied can be real.

Chat is far better behaved than voice here — 2 of 996 AI bookings and 3 of 190
health calls carried invented codes in 2026-09-01..09-14, against voice's 10-20%
— because the farmer context is almost always resolved before the turn. But the
tool signatures are the same, the guard was on `create_ai_call` only, and the
one tool without it is the one that leaks. See voice-oan-api#282.
"""
import re

# Real codes are NOT always numeric — M001 and NA4192 book fine.
CODE_PATTERN = re.compile(r"^[A-Za-z0-9/-]{1,12}$")
TECHNICIAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9+/]{22}==$")

# Every real code carries at least one digit. Verified against 28,089 successful
# bookings across both channels (22 unions, 2,032 societies, 2,548 farmer codes)
# and 110 health calls over 90 days: zero exceptions. Without this the shape
# check accepts MISSING / UNKNOWN / not_provided, which is why it only ever
# worked for create_ai_call, where the technician-id check did the real work.
_HAS_DIGIT = re.compile(r"[0-9]")


def invalid_code_field(
    union_code: str,
    society_code: str,
    farmer_code: str,
) -> str | None:
    """Name of the first code that cannot be real, else None."""
    for field, value in (
        ("union_code", union_code),
        ("society_code", society_code),
        ("farmer_code", farmer_code),
    ):
        cleaned = (value or "").strip()
        if not CODE_PATTERN.match(cleaned) or not _HAS_DIGIT.search(cleaned):
            return field
    return None


def invalid_technician_id(user_id: str) -> bool:
    """True when the technician id cannot be a real one (24 base64 chars, '==')."""
    return not TECHNICIAN_ID_PATTERN.match((user_id or "").strip())
