"""SQLAlchemy ORM models for the micro-loan eligibility feature.

Two tables live in the pre-existing per-env Postgres:

- ``loan_eligibility_list`` — the cooperative-bank eligibility list (loaded from
  the SABHSAD export). A caller is "on the list" when a row matches their
  normalized 10-digit phone and ``is_active`` is true.
- ``loan_codes`` — every issued approval code. This is the authoritative store
  the bank-side verification portal will later read to confirm / redeem a code.

This is the first Postgres/ORM in this service — there was previously no
app-owned relational layer (persistence was Redis + external HTTP APIs).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LoanEligibilityRow(Base):
    """One cooperative-bank-eligible member (loaded from the SABHSAD export)."""

    __tablename__ = "loan_eligibility_list"
    __table_args__ = (
        UniqueConstraint("phone", "farmer_code", name="uq_loan_eligibility_phone_farmer"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Normalized last-10-digits phone — the match key for eligibility.
    phone: Mapped[str] = mapped_column(String(15), nullable=False, index=True)
    farmer_code: Mapped[str | None] = mapped_column(String(64))
    mandali_name: Mapped[str | None] = mapped_column(String(128))
    sabhsad_name: Mapped[str | None] = mapped_column(String(256))
    ac_no: Mapped[str | None] = mapped_column(String(32))
    # Snapshot of the milk payout from the source sheet (informational only —
    # the live milk check uses the milk API, not this column).
    milk_payment_amount: Mapped[float | None] = mapped_column(Numeric(12, 2))
    source_batch: Mapped[str | None] = mapped_column(String(64), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LoanCode(Base):
    """An issued micro-loan approval code + the eligibility snapshot behind it."""

    __tablename__ = "loan_codes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(12), nullable=False, unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(15), nullable=False, index=True)

    farmer_name: Mapped[str | None] = mapped_column(String(256))
    farmer_code: Mapped[str | None] = mapped_column(String(64))
    mandali_name: Mapped[str | None] = mapped_column(String(128))
    union_code: Mapped[str | None] = mapped_column(String(64))
    society_code: Mapped[str | None] = mapped_column(String(64))

    loan_amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    milk_amount_month: Mapped[float | None] = mapped_column(Numeric(12, 2))
    milk_threshold: Mapped[float | None] = mapped_column(Numeric(12, 2))

    channel: Mapped[str | None] = mapped_column(String(16))  # 'chat' | 'voice'
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="active")

    sms_status: Mapped[str | None] = mapped_column(String(16))  # sent|failed|skipped|dry_run
    sms_message_id: Mapped[str | None] = mapped_column(String(128))
    sms_error: Mapped[str | None] = mapped_column(Text)

    # Audit of which eligibility checks were active when this code was issued
    # (so a bypassed-check test issue is distinguishable from a real approval).
    checks_applied: Mapped[dict | None] = mapped_column(JSONB)
    session_id: Mapped[str | None] = mapped_column(String(128))

    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    redeemed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    redeemed_by: Mapped[str | None] = mapped_column(String(128))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
