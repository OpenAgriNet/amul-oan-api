from datetime import datetime

from pydantic import BaseModel, Field, field_validator


DATE_FORMAT = "%Y-%m-%d"


class FarmerMilkCollectionRequestModel(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")
    from_date: str = Field(..., alias="fromdate")
    to_date: str = Field(..., alias="todate")

    @field_validator("from_date", "to_date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, DATE_FORMAT)
        except ValueError as exc:
            raise ValueError("Date must be in YYYY-MM-DD format.") from exc
        return value

    def to_query_params(self) -> dict[str, str]:
        # Dates are validated as YYYY-MM-DD; PashuGPT expects the same in query params.
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
            "fromdate": self.from_date,
            "todate": self.to_date,
        }

    def validate_date_range(self) -> None:
        start_date = datetime.strptime(self.from_date, DATE_FORMAT)
        end_date = datetime.strptime(self.to_date, DATE_FORMAT)

        if end_date < start_date:
            raise ValueError("todate must be on or after fromdate.")
        if (end_date - start_date).days > 31:
            raise ValueError("Date range cannot exceed 31 days.")


class MilkCollectionRecordModel(BaseModel):
    date: str
    shift: str
    qty: float
    fat: float
    snf: float
    amount: float


class DeductionRecordModel(BaseModel):
    date: str
    account_name: str = Field(..., alias="accountname")
    amount: float


class FarmerMilkCollectionResponseModel(BaseModel):
    result: str | None = None
    milk: list[MilkCollectionRecordModel] = Field(default_factory=list)
    deduction: list[DeductionRecordModel] = Field(default_factory=list)
