"""Tests for the deterministic micro-loan eligibility service, SMS body, and the
tool's outcome→message mapping.

Style matches the repo: plain pytest with `monkeypatch` + `asyncio.run` (no
pytest-asyncio). The DB session and external calls (milk API, SMS) are patched,
so no Postgres or network is required.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.deps import FarmerAccount
from agents.services import loan_eligibility as le
from agents.tools import loan as loan_tool
from agents.tools.onex_sms import build_loan_sms_body, _to_msisdn, _format_amount


# ── fakes ────────────────────────────────────────────────────────────────────
class _FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        self.committed = True

    async def rollback(self):
        pass

    async def close(self):
        pass


class _FakeCM:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *a):
        return False


def _wire(monkeypatch, session, *, feature=True, bank=True, milk=True, allow_multiple=False, resend=False, sms=False):
    monkeypatch.setattr(le, "loan_db_configured", lambda: True)
    monkeypatch.setattr(le, "get_loan_session", lambda: _FakeCM(session))
    monkeypatch.setattr(le.settings, "loan_feature_enabled", feature)
    monkeypatch.setattr(le.settings, "loan_check_bank_list_enabled", bank)
    monkeypatch.setattr(le.settings, "loan_check_milk_enabled", milk)
    monkeypatch.setattr(le.settings, "loan_allow_multiple_codes", allow_multiple)
    monkeypatch.setattr(le.settings, "loan_resend_sms_on_request", resend)
    monkeypatch.setattr(le.settings, "loan_sms_enabled", sms)
    monkeypatch.setattr(le.settings, "loan_max_amount", 5000.0)
    monkeypatch.setattr(le.settings, "loan_milk_threshold", 3000.0)


def _run(**kw):
    defaults = dict(phone="917011854675", accounts=[], farmer_name="Ramesh", channel="voice")
    defaults.update(kw)
    return asyncio.run(le.evaluate_and_issue(**defaults))


# ── service branches ─────────────────────────────────────────────────────────
class TestEvaluateAndIssue:
    def test_disabled_when_feature_off(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(), feature=False)
        assert _run().outcome == le.DISABLED

    def test_no_phone(self, monkeypatch):
        _wire(monkeypatch, _FakeSession())
        assert _run(phone=None).outcome == le.NO_PHONE
        assert _run(phone="anonymous").outcome == le.NO_PHONE

    def test_existing_code_is_reshared(self, monkeypatch):
        # An existing active code is re-shared (ELIGIBLE), not rejected, when
        # multiple codes are not allowed.
        _wire(monkeypatch, _FakeSession(), allow_multiple=False)

        async def _existing(session, phone):
            return SimpleNamespace(code="999999", loan_amount=5000, farmer_name="Ramesh",
                                   sms_status="sent", issued_at="x", expires_at=None)

        monkeypatch.setattr(le, "_active_code_for_phone", _existing)
        res = _run()
        assert res.outcome == le.ELIGIBLE
        assert res.reshared is True
        assert res.code == "999999"

    def test_allow_multiple_mints_new_code(self, monkeypatch):
        # With multiple codes allowed, an existing code is ignored and a new one issued.
        _wire(monkeypatch, _FakeSession(), allow_multiple=True)
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _row)
        monkeypatch.setattr(le, "_compute_last_month_milk", _milk(5200.0))
        monkeypatch.setattr(le, "_generate_unique_code", _code("222333"))
        res = _run()
        assert res.outcome == le.ELIGIBLE and res.reshared is False and res.code == "222333"

    def test_not_in_bank_list(self, monkeypatch):
        _wire(monkeypatch, _FakeSession())
        monkeypatch.setattr(le, "_active_code_for_phone", _none)
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _none)
        assert _run().outcome == le.NOT_IN_BANK_LIST

    def test_milk_below_threshold(self, monkeypatch):
        _wire(monkeypatch, _FakeSession())
        monkeypatch.setattr(le, "_active_code_for_phone", _none)
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _row)
        monkeypatch.setattr(le, "_compute_last_month_milk", _milk(1000.0))
        res = _run()
        assert res.outcome == le.MILK_BELOW_THRESHOLD
        assert res.milk_amount_month == 1000.0

    def test_eligible_dry_run_issues_and_stores(self, monkeypatch):
        session = _FakeSession()
        _wire(monkeypatch, session, sms=False)
        monkeypatch.setattr(le, "_active_code_for_phone", _none)
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _row)
        monkeypatch.setattr(le, "_compute_last_month_milk", _milk(5200.0))
        monkeypatch.setattr(le, "_generate_unique_code", _code("123456"))
        res = _run()
        assert res.outcome == le.ELIGIBLE
        assert res.code == "123456"
        assert res.loan_amount == 5000.0
        assert res.sms_status == "dry_run"
        assert session.committed and len(session.added) == 1
        assert session.added[0].code == "123456" and session.added[0].status == "active"

    def test_eligible_sends_sms_when_enabled(self, monkeypatch):
        session = _FakeSession()
        _wire(monkeypatch, session, sms=True)
        monkeypatch.setattr(le, "_active_code_for_phone", _none)
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _row)
        monkeypatch.setattr(le, "_compute_last_month_milk", _milk(5200.0))
        monkeypatch.setattr(le, "_generate_unique_code", _code("654321"))

        async def _send(phone, name, amount, code):
            assert code == "654321" and int(amount) == 5000
            return SimpleNamespace(ok=True, status="sent", message_id="m1", error=None)

        monkeypatch.setattr(le, "send_loan_approval_sms", _send)
        res = _run()
        assert res.outcome == le.ELIGIBLE and res.sms_status == "sent"

    def test_all_checks_bypassed_for_testing(self, monkeypatch):
        """Product test mode: every check off -> eligible without bank row or milk call."""
        session = _FakeSession()
        _wire(monkeypatch, session, bank=False, milk=False, sms=False)
        monkeypatch.setattr(le, "_active_code_for_phone", _none)  # no existing code to re-share
        monkeypatch.setattr(le, "_eligibility_row_for_phone", _none)  # no row exists
        monkeypatch.setattr(le, "_generate_unique_code", _code("111222"))

        called = {"milk": False}

        async def _milk_should_not_run(accounts):
            called["milk"] = True
            return 0.0

        monkeypatch.setattr(le, "_compute_last_month_milk", _milk_should_not_run)
        res = _run()
        assert res.outcome == le.ELIGIBLE and res.code == "111222"
        assert called["milk"] is False  # milk check skipped when disabled


# ── SMS body ─────────────────────────────────────────────────────────────────
class TestSmsBody:
    def test_msisdn(self):
        assert _to_msisdn("7011854675") == "917011854675"
        assert _to_msisdn("+91 70118-54675") == "917011854675"

    def test_amount_format(self):
        assert _format_amount(5000) == "5,000"
        assert _format_amount(4999.6) == "5,000"

    def test_body_has_placeholders_filled(self):
        body = build_loan_sms_body("રમેશભાઈ", 5000, "134582")
        assert "134582" in body and "5,000" in body and "રમેશભાઈ" in body

    def test_body_name_fallback(self):
        body = build_loan_sms_body("", 5000, "134582")
        assert "134582" in body  # empty name falls back, still renders code


# ── tool message mapping ─────────────────────────────────────────────────────
class TestMessageMapping:
    def test_eligible_message_has_code_and_amount(self):
        r = le.LoanResult(outcome=le.ELIGIBLE, code="777888", loan_amount=5000)
        msg = loan_tool._message_for(r)
        assert "777888" in msg and "5,000" in msg and "ELIGIBLE" in msg

    def test_failure_messages_direct_to_bank(self):
        for oc in (le.NOT_IN_BANK_LIST, le.MILK_BELOW_THRESHOLD):
            msg = loan_tool._message_for(le.LoanResult(outcome=oc, milk_threshold=3000))
            assert "bank" in msg.lower()

    def test_no_phone_message_asks_for_number(self):
        msg = loan_tool._message_for(le.LoanResult(outcome=le.NO_PHONE))
        assert "mobile" in msg.lower()


# ── shared async stubs ───────────────────────────────────────────────────────
async def _none(*a, **k):
    return None


def _row(*a, **k):
    async def inner(*a, **k):
        return SimpleNamespace(
            farmer_code="1554", mandali_name="DANTALI", sabhsad_name="ALPESH PATEL",
        )
    return inner()


def _milk(value):
    async def inner(accounts):
        return value
    return inner


def _code(value):
    async def inner(session):
        return value
    return inner
