"""SQLAlchemy ORM model for partner webhook payload storage.

This model is intentionally denormalized and short-lived (24h retention by
default). We keep key lookup columns plus the full payload for compatibility
with schema changes from the sender.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class WebhookBase(DeclarativeBase):
    pass


class HeatAlertWebhookEvent(WebhookBase):
    """One inbound partner heat-alert webhook payload."""

    __tablename__ = "heat_alert_webhook_events"
    __table_args__ = (
        Index("ix_heat_alert_events_received_at", "received_at"),
        Index("ix_heat_alert_events_farmer_contact", "farmer_contact"),
        Index("ix_heat_alert_events_device_id", "device_id"),
        UniqueConstraint(
            "alert_id",
            "farmer_contact",
            name="uq_heat_alert_events_alert_id_farmer_contact",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    alert_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    notification_type: Mapped[str | None] = mapped_column(String(64))
    alert_type: Mapped[str | None] = mapped_column(String(32))
    heat_score: Mapped[str | None] = mapped_column(String(32))

    tag_no: Mapped[str | None] = mapped_column(String(64))
    system_tag_no: Mapped[str | None] = mapped_column(String(64))
    pashuaadhar_no: Mapped[str | None] = mapped_column(String(64))

    farm_name: Mapped[str | None] = mapped_column(String(256))
    farm_id: Mapped[str | None] = mapped_column(String(64))
    union_code: Mapped[str | None] = mapped_column(String(64))
    society_code: Mapped[str | None] = mapped_column(String(64), index=True)
    farmer_code: Mapped[str | None] = mapped_column(String(64), index=True)
    farmer_contact: Mapped[str | None] = mapped_column(String(32))

    ai_window_start_time_raw: Mapped[str | None] = mapped_column(Text)
    ai_window_end_time_raw: Mapped[str | None] = mapped_column(Text)
    alert_time_raw: Mapped[str | None] = mapped_column(Text)
    timestamp_raw: Mapped[str | None] = mapped_column(Text)
    msg_title: Mapped[str | None] = mapped_column(Text)
    msg_body: Mapped[str | None] = mapped_column(Text)
    booking_url: Mapped[str | None] = mapped_column(Text)
    called: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    call_booked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    # The outbound call (app/services/heat_alert_call.py): when it is due, the
    # provider's id once placed, and why it was not.
    call_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outbound_call_uuid: Mapped[str | None] = mapped_column(String(64))
    call_error: Mapped[str | None] = mapped_column(String(64))

    # Parsed timestamps (if parseable) improve queryability.
    ai_window_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ai_window_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    partner_timestamp_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    partner_alert_epoch_s: Mapped[int | None] = mapped_column(BigInteger)

    payload_raw: Mapped[dict] = mapped_column(JSONB, nullable=False)
