"""Version-one wire models for the development milk UI."""
from __future__ import annotations

from datetime import date
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_BODY_BYTES = 256 * 1024
MAX_TEXT_BYTES = 16 * 1024
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChoiceValue(WireModel):
    option_id: Identifier


class Period(WireModel):
    fromdate: str
    todate: str

    @field_validator("fromdate", "todate")
    @classmethod
    def iso_date(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("date must use YYYY-MM-DD")
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def date_range(self):
        difference = (date.fromisoformat(self.todate) - date.fromisoformat(self.fromdate)).days
        if not 0 <= difference <= 31:
            raise ValueError("end date must be between zero and 31 days after start")
        return self


class ConfirmValue(WireModel):
    confirmed: bool


class ActionBase(WireModel):
    component_id: Identifier
    revision: Annotated[int, Field(ge=1)]


class ChoiceAction(ActionBase):
    kind: Literal["choice"]
    value: ChoiceValue


class DateAction(ActionBase):
    kind: Literal["date_range"]
    value: Period


class ConfirmAction(ActionBase):
    kind: Literal["confirm"]
    value: ConfirmValue


UIAction = Annotated[ChoiceAction | DateAction | ConfirmAction, Field(discriminator="kind")]


class ChangeAccount(WireModel):
    kind: Literal["change_milk_account"]
    state_revision: Annotated[int, Field(ge=0, le=9007199254740991)]


class RequestProps(WireModel):
    version: Literal[1]
    channel: Literal["web", "whatsapp"] = "web"
    source_lang: Annotated[str, Field(min_length=1, max_length=16)] = "gu"
    target_lang: Annotated[str, Field(min_length=1, max_length=16)] = "gu"
    input_method: Annotated[str, Field(max_length=32)] | None = None
    user_id: Annotated[str, Field(max_length=128)] = "anonymous"
    persona: Literal["farmer", "doctor"] | None = None
    ui_action: UIAction | None = None
    command: ChangeAccount | None = None


class UserMessage(WireModel):
    id: Identifier
    role: Literal["user"]
    content: str

    @field_validator("content")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("message must contain 1 to 16384 UTF-8 bytes")
        return value


class ChatRunRequest(WireModel):
    threadId: Identifier
    runId: Identifier
    state: dict
    messages: Annotated[list[UserMessage], Field(max_length=1)]
    tools: list
    context: list
    forwardedProps: RequestProps

    @model_validator(mode="after")
    def one_operation(self):
        if self.state or self.tools or self.context:
            raise ValueError("state, tools, and context must be empty")
        props = self.forwardedProps
        if bool(self.messages) + (props.ui_action is not None) + (props.command is not None) != 1:
            raise ValueError("send exactly one message, ui_action, or command")
        return self


class Option(WireModel):
    id: str
    label: str
    description: str | None


class Account(WireModel):
    option_id: str
    label: str


class Summary(WireModel):
    days_with_milk: int
    total_quantity_litres: float
    total_amount_rupees: float
    weighted_fat: float | None
    weighted_snf: float | None


class MilkRow(WireModel):
    date: str | None
    shift: str | None
    qty: float | None
    fat: float | None
    snf: float | None
    amount: float | None


class Deduction(WireModel):
    date: str | None
    account_name: str | None
    amount: float | None


class DailyQuantity(WireModel):
    date: str
    quantity_litres: float | None


class MilkResult(WireModel):
    status: Literal["ok", "no_records"]
    account: Account
    period: Period
    summary: Summary | None
    milk: list[MilkRow]
    deductions: list[Deduction]
    daily_quantity: list[DailyQuantity]
    warnings: list[Literal["unplottable_dates", "missing_quantities"]]


class ComponentBase(WireModel):
    id: str
    workflow_id: str
    message_id: str
    revision: int
    title: str
    status: Literal["pending", "submitted", "expired"]


class ChoiceComponent(ComponentBase):
    kind: Literal["choice"]
    purpose: Literal["milk_account"]
    options: list[Option]
    selected: ChoiceValue | None


class DateConstraints(WireModel):
    date_format: Literal["YYYY-MM-DD"] = "YYYY-MM-DD"
    max_difference_days: Literal[31] = 31
    max_date: str | None = None


class DateComponent(ComponentBase):
    kind: Literal["date_range"]
    purpose: Literal["milk_period"]
    constraints: DateConstraints
    selected: Period | None


class ChartComponent(ComponentBase):
    kind: Literal["chart"]
    purpose: Literal["milk_records"]
    status: Literal["submitted"]
    data: MilkResult


Component = Annotated[ChoiceComponent | DateComponent | ChartComponent, Field(discriminator="kind")]


class UIState(WireModel):
    version: Literal[1] = 1
    state_revision: int
    selected_milk_account: Account | None
    components: list[Component]


class ErrorDetail(WireModel):
    code: str
    message: str
    retryable: bool = False
    component_id: str | None = None
    field: str | None = None


class ErrorResponse(WireModel):
    error: ErrorDetail
