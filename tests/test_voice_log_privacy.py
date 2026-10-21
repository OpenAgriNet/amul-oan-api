"""Voice's logs hold no caller's phone, words or farmer code: their length, the
session id or the error's class instead, as d8d2a22 and 9179f71 did for the route
and the agent input.

Every log call in the modules voice runs is read off the code, so a line added
later is held to it too. An error that can quote the phone it was given is
checked by running it.
"""
import ast
import asyncio
from pathlib import Path

import pytest

from agents.tools import farmer, onex_sms
from agents.voice.services import loan_eligibility as le
from tests.test_voice_loan_eligibility import _FakeSession, _wire

_ROOT = Path(__file__).resolve().parents[1]
_MODULES = [
    "app/voice/classifiers.py",
    "app/voice/pretranslation.py",
    "app/voice/moderation.py",
    "agents/voice/tools/search.py",
    "agents/voice/tools/ai_call.py",
    "agents/voice/tools/milk_collection.py",
    "agents/voice/tools/loan.py",
    "agents/voice/services/loan_eligibility.py",
    "agents/tools/onex_sms.py",
    "agents/tools/farmer.py",
]
# What holds the caller's words, phone or farmer code where these modules log.
_RAW_NAMES = {"query", "text", "expanded_query", "mobile", "phone", "farmer_code", "_to_msisdn"}
# As attributes too, except text, which is also an HTTP response's body.
_RAW_ATTRIBUTES = _RAW_NAMES - {"text"}
_MEASURED = {"len", "type", "bool"}
_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}
_MOBILE = "9876543210"


def _raw(node):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _MEASURED:
        return set()
    found = set()
    if isinstance(node, ast.Name) and node.id in _RAW_NAMES:
        found.add(node.id)
    elif isinstance(node, ast.Attribute) and node.attr in _RAW_ATTRIBUTES:
        found.add(node.attr)
    for child in ast.iter_child_nodes(node):
        found |= _raw(child)
    return found


def _logged_raw(module):
    tree = ast.parse((_ROOT / module).read_text(encoding="utf-8"))
    return [
        f"{module}:{call.lineno} {name}"
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _LEVELS
        and "log" in ast.unparse(call.func.value).lower()
        for arg in call.args
        for name in sorted(_raw(arg))
    ]


@pytest.mark.parametrize("module", _MODULES)
def test_voice_logs_hold_no_callers_phone_words_or_farmer_code(module):
    assert _logged_raw(module) == []


def _quotes_the_phone(*_a, **_k):
    raise RuntimeError(f"lookup failed for mobile={_MOBILE}")


def _farmer_lookup(monkeypatch):
    monkeypatch.setattr(farmer, "fetch_authenticated_farmers", _quotes_the_phone)
    asyncio.run(farmer.get_farmer_data_by_mobile(_MOBILE))


def _farmer_refresh(monkeypatch):
    monkeypatch.setattr(farmer, "fetch_authenticated_farmers", _quotes_the_phone)
    asyncio.run(farmer.fetch_farmer_info_with_outcome(_MOBILE))


def _loan_check(monkeypatch):
    _wire(monkeypatch, _FakeSession())
    monkeypatch.setattr(le, "_active_code_for_phone", _quotes_the_phone)
    asyncio.run(le.evaluate_and_issue(phone=_MOBILE, accounts=[], channel="voice", session_id="call-1"))


class _Gateway:
    def __init__(self, **_k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_e):
        return False

    async def get(self, url, params):
        raise RuntimeError(f"GET {url}?to={params['to']} failed")


def _loan_sms(monkeypatch):
    monkeypatch.setattr(onex_sms.settings, "onex_sms_key", "key")
    monkeypatch.setattr(onex_sms.httpx, "AsyncClient", _Gateway)
    asyncio.run(onex_sms.send_loan_approval_sms(_MOBILE, "Ramesh", 5000.0, "123456"))


@pytest.mark.parametrize("fails", [_farmer_lookup, _farmer_refresh, _loan_check, _loan_sms], ids=lambda f: f.__name__.strip("_"))
def test_an_error_that_quotes_the_phone_is_logged_by_its_class(monkeypatch, caplog, fails):
    caplog.set_level("DEBUG")

    fails(monkeypatch)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "RuntimeError" in logged
    assert _MOBILE not in logged
