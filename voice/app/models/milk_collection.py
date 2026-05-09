"""Milk collection and deduction detail models."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


DATE_FORMAT = "%Y-%m-%d"
MAX_DATE_RANGE_DAYS = 31


class FarmerMilkCollectionRequestModel(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")
    fromdate: str
    todate: str

    def to_query_params(self) -> dict[str, str]:
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
            "fromdate": self.fromdate,
            "todate": self.todate,
        }

    def validate_date_range(self) -> None:
        from_date = _parse_collection_date(self.fromdate, "fromdate")
        to_date = _parse_collection_date(self.todate, "todate")

        if to_date < from_date:
            raise ValueError("todate must be on or after fromdate")
        if (to_date - from_date).days > MAX_DATE_RANGE_DAYS:
            raise ValueError("date range cannot exceed 31 days")

    @field_validator("fromdate", "todate")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        _parse_collection_date(value, "date")
        return value


class MilkCollectionRecordModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    date: str | None = None
    shift: str | None = None
    qty: float | int | None = None
    fat: float | int | None = None
    snf: float | int | None = None
    amount: float | int | None = None


class DeductionRecordModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    date: str | None = None
    account_name: str | None = Field(None, alias="accountname")
    amount: float | int | None = None


class FarmerMilkCollectionResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    result: str | None = None
    milk: list[MilkCollectionRecordModel] = Field(default_factory=list)
    deduction: list[DeductionRecordModel] = Field(default_factory=list)


def _parse_collection_date(value: str, field_name: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format") from exc
