"""Two confirmations of one farmer's loan at the same time, on a real Postgres.

Needs a Postgres the test may create tables in, as LOAN_TEST_DB_URL, e.g.
postgresql+asyncpg://postgres@127.0.0.1:5432/loan_test. Skipped without one.
"""
import asyncio
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from agents.voice.services import loan_eligibility as le
from app.core import loan_db
from app.voice.models.loan import Base, LoanCode

URL = os.getenv("LOAN_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set LOAN_TEST_DB_URL to a throwaway Postgres")

PHONE = "7011854675"


@pytest.fixture
def sent(monkeypatch):
    """The service on the test database with every check but milk off; the SMSs it sends."""
    monkeypatch.setattr(loan_db, "_engine", None)
    monkeypatch.setattr(loan_db, "_sessionmaker", None)
    for name, value in {
        "loan_db_url": URL,
        "loan_feature_enabled": True,
        "loan_check_bank_list_enabled": False,
        "loan_check_milk_enabled": True,
        "loan_resend_sms_on_request": False,
        "loan_sms_enabled": True,
        "loan_max_amount": 5000.0,
        "loan_milk_threshold": 3000.0,
        "loan_code_expiry_days": 0,
    }.items():
        monkeypatch.setattr(le.settings, name, value)
    messages = []

    async def _send(mobile, name, amount, code):
        messages.append(code)
        return SimpleNamespace(status="sent", message_id=f"m{len(messages)}", error=None)

    monkeypatch.setattr(le, "send_loan_approval_sms", _send)
    return messages


def _confirm_twice(monkeypatch):
    """Two confirmations at once. The milk check sits between reading the phone's
    codes and inserting one, so it waits for the other confirmation to get there
    too: without a lock both do, and both see no code."""

    async def _go():
        arrived = []
        both = asyncio.Event()

        async def _milk(accounts):
            arrived.append(1)
            if len(arrived) == 2:
                both.set()
            try:
                await asyncio.wait_for(both.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # the other confirmation is waiting on the lock
            return 5200.0

        monkeypatch.setattr(le, "_compute_last_month_milk", _milk)
        engine = loan_db._get_sessionmaker() and loan_db._engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        try:
            results = await asyncio.gather(*(
                le.evaluate_and_issue(phone=PHONE, accounts=[], farmer_name="Ramesh", channel="voice", confirm=True)
                for _ in range(2)
            ))
            async with loan_db.get_loan_session() as session:
                codes = (await session.execute(select(LoanCode.code).where(LoanCode.phone == PHONE))).scalars().all()
        finally:
            await engine.dispose()
        return results, codes

    return asyncio.run(_go())


def test_two_confirmations_at_once_issue_one_code_and_one_sms(monkeypatch, sent):
    monkeypatch.setattr(le.settings, "loan_allow_multiple_codes", False)

    results, codes = _confirm_twice(monkeypatch)

    assert [r.outcome for r in results] == [le.ELIGIBLE, le.ELIGIBLE]
    assert len(codes) == 1
    assert {r.code for r in results} == set(codes)
    assert sorted(r.reshared for r in results) == [False, True]  # the second read the first's code
    assert sent == codes


def test_with_several_codes_allowed_each_confirmation_gets_its_own(monkeypatch, sent):
    monkeypatch.setattr(le.settings, "loan_allow_multiple_codes", True)

    results, codes = _confirm_twice(monkeypatch)

    assert len(codes) == 2 and {r.code for r in results} == set(codes)
    assert sorted(sent) == sorted(codes)
