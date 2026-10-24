"""Models for the PashuGPT GetFarmerTicketSummary endpoint."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FarmerTicketSummaryRequestModel(BaseModel):
    """Authenticated farmer account codes sent as provider query params."""

    model_config = ConfigDict(populate_by_name=True)

    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")

    def to_query_params(self) -> dict[str, str]:
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
        }


class FarmerTicketSummaryFarmerModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    farmer_name: str | None = Field(None, alias="farmerName")
    farmer_code: str | None = Field(None, alias="farmerCode")
    society_name: str | None = Field(None, alias="societyName")
    society_code: str | None = Field(None, alias="societyCode")


class PendingTicketModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticket_number: str | None = Field(None, alias="ticketNumber")
    book_date_time: str | None = Field(None, alias="bookDateTime")


class LastBreedingActivityModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    last_ai: str | None = Field(None, alias="lastAI")
    last_ai_ticket_number: str | None = Field(None, alias="lastAITicketNumber")
    ai_bull_id: int | None = Field(None, alias="aiBullId")
    # Live responses also expose bullId as a string; keep both provider forms.
    bull_id: str | None = Field(None, alias="bullId")
    semen_type: str | None = Field(None, alias="semenType")
    last_pd: str | None = Field(None, alias="lastPD")
    last_pd_ticket_number: str | None = Field(None, alias="lastPDTicketNumber")
    pd_status: str | None = Field(None, alias="pdStatus")
    last_calving: str | None = Field(None, alias="lastCalving")
    last_calving_ticket_number: str | None = Field(
        None, alias="lastCalvingTicketNumber"
    )

    def bull_identifier(self) -> str | None:
        return self.bull_id or (
            str(self.ai_bull_id) if self.ai_bull_id is not None else None
        )


class LastLactationDetailModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    animal_tag_number: str | None = Field(None, alias="animalTagNumber")
    last_breeding_activity: LastBreedingActivityModel | None = Field(
        None, alias="lastBreedingActivity"
    )


class FarmerTicketSummaryResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    farmer_details: FarmerTicketSummaryFarmerModel | None = Field(
        None, alias="farmerDetails"
    )
    pending_tickets: list[PendingTicketModel] = Field(
        default_factory=list, alias="pendingTickets"
    )
    last_lactation_detail: list[LastLactationDetailModel] = Field(
        default_factory=list, alias="lastLactationDetail"
    )

    @field_validator("pending_tickets", "last_lactation_detail", mode="before")
    @classmethod
    def _nullable_lists_are_empty(cls, value):
        return [] if value is None else value

    def has_data(self) -> bool:
        return bool(self.pending_tickets or self.last_lactation_detail)
