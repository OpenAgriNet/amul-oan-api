"""Tests for normalize_phone_to_mobile (Inc 7.2) — how the voice path derives a
caller's 10-digit mobile from the request user_id (and thus farmer resolution /
signed-in state). None means "not a usable phone" (anon / UUID / name / <10 digits).
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.farmer import normalize_phone_to_mobile


def test_plain_10_digit():
    assert normalize_phone_to_mobile("9876543210") == "9876543210"


def test_country_code_and_formatting_stripped_to_last_10():
    assert normalize_phone_to_mobile("+91 98765 43210") == "9876543210"
    assert normalize_phone_to_mobile("919876543210") == "9876543210"


def test_anonymous_returns_none():
    assert normalize_phone_to_mobile("anonymous") is None
    assert normalize_phone_to_mobile("anon") is None
    assert normalize_phone_to_mobile("ANONYMOUS") is None


def test_uuid_returns_none():
    assert normalize_phone_to_mobile("550e8400-e29b-41d4-a716-446655440000") is None


def test_too_few_digits_returns_none():
    assert normalize_phone_to_mobile("12345") is None


def test_empty_or_none_returns_none():
    assert normalize_phone_to_mobile("") is None
    assert normalize_phone_to_mobile(None) is None
