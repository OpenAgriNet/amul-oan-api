"""Farmer bonus amount models for GetFarmerBonusAmount."""
from pydantic import BaseModel, ConfigDict, Field


class FarmerBonusAmountRequestModel(BaseModel):
    """Query params for PashuGPT GetFarmerBonusAmount."""

    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")

    def to_query_params(self) -> dict[str, str]:
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
        }


class FarmerBonusAmountRecordModel(BaseModel):
    """One bonus period row from GetFarmerBonusAmount (plain JSON array item)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    society_code: str | None = Field(None, alias="societyCode")
    society_name: str | None = Field(None, alias="societyName")
    society_name_local: str | None = Field(None, alias="societyNameLocal")
    farmer_code: str | None = Field(None, alias="farmerCode")
    farmer_name: str | None = Field(None, alias="farmerName")
    farmer_local_name: str | None = Field(None, alias="farmerLocalName")
    bonus_amount: float | int | None = Field(None, alias="bonusAmount")
    # API returns ISO datetime strings (e.g. "2026-04-01T00:00:00"); keep as str.
    from_date: str | None = Field(None, alias="fromDate")
    to_date: str | None = Field(None, alias="toDate")
