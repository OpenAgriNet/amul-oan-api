"""Request/response contracts for the partner heat-alert webhook.

Partner payloads use irregular key names (spaces, mixed casing). We accept the
exact wire keys via Field aliases while exposing snake_case attributes in code.

NOTE: Do not enable ``from __future__ import annotations`` here — Pydantic needs
concrete field annotations at class-body time for aliases to bind correctly.
"""
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class NotificationType(str, Enum):
    """Partner notification kinds. Required on every webhook body."""

    HEAT = "heat"


def _coerce_optional_str(value: Any) -> Any:
    """Partners sometimes send numeric JSON values for id-like fields.

    Coerce int/float to str so validation does not reject otherwise-valid
    payloads; leave other types for Pydantic to validate/reject.
    """
    if isinstance(value, bool):
        # bool is a subclass of int — do not stringify True/False.
        return value
    if isinstance(value, (int, float)):
        # Preserve integer-looking floats as ints in the string form
        # (e.g. 89.0 -> "89") to match partner string conventions.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return value


class HeatAlertWebhookRequest(BaseModel):
    """Inbound heat-alert webhook body from the external partner system."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    alert_id: Annotated[
        str, Field(alias="alertID", description="Partner alert identifier", max_length=64)
    ]
    device_id: Annotated[
        str, Field(alias="deviceid", description="Sensor / device identifier", max_length=128)
    ]
    tag_no: Annotated[
        Optional[str], Field(alias="tag no", description="Animal tag number", max_length=64)
    ] = None
    pashuaadhar_no: Annotated[
        Optional[str], Field(alias="Pashuaadhar No", description="Pashuaadhar number", max_length=64)
    ] = None
    farm_name: Annotated[
        Optional[str], Field(alias="FarmName", description="Farm display name", max_length=256)
    ] = None
    farm_id: Annotated[
        Optional[str], Field(alias="farmid", description="Farm identifier", max_length=64)
    ] = None
    msg_title: str | None = Field(default=None, description="Notification title")
    msg_body: str | None = Field(default=None, description="Notification body")
    heat_score: str | None = Field(
        default=None, description="Heat score as sent by partner", max_length=32
    )
    alert_type: str = Field(description="Alert severity, e.g. Amber", max_length=32)
    notification_type: NotificationType = Field(
        description="Required notification kind. Today the only value is heat."
    )
    ai_window_start_time: Annotated[
        str, Field(alias="AI Window start_time", description="AI window start (ISO string)")
    ]
    ai_window_end_time: Annotated[
        str, Field(alias="AI Window end_time", description="AI window end (ISO string)")
    ]
    alert_time: str | None = Field(
        default=None, description="Partner alert time (often a unix epoch string)"
    )
    timestamp: str = Field(description="Partner event timestamp (ISO string)")
    society_code: str | None = Field(
        default=None, description="Society / mandali code", max_length=64
    )
    farmer_code: str | None = Field(default=None, description="Farmer code", max_length=64)
    farmer_contact: str = Field(description="Farmer phone / contact", max_length=32)
    system_tag_no: str | None = Field(default=None, description="System tag number", max_length=64)
    booking_url: str | None = Field(
        default=None, description="Partner-provided AI booking URL", max_length=2048
    )

    @field_validator(
        "alert_id",
        "device_id",
        "tag_no",
        "pashuaadhar_no",
        "farm_name",
        "farm_id",
        "msg_title",
        "msg_body",
        "heat_score",
        "alert_type",
        "notification_type",
        "ai_window_start_time",
        "ai_window_end_time",
        "alert_time",
        "timestamp",
        "society_code",
        "farmer_code",
        "farmer_contact",
        "system_tag_no",
        mode="before",
    )
    @classmethod
    def _stringify_numeric_fields(cls, value: Any, info: ValidationInfo) -> Any:
        value = _coerce_optional_str(value)
        if info.field_name == "notification_type" and isinstance(value, str):
            return value.strip().lower()
        return value


class HeatAlertWebhookResponse(BaseModel):
    """Acknowledgement returned after a webhook payload is accepted."""

    accepted: bool = True
    id: UUID
    received_at: datetime
