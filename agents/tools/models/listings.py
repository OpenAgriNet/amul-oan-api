"""Existing listing filters shared by agent requests and AG-UI actions."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Text = Annotated[str, Field(max_length=256)]
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator('*', mode='before')
    @classmethod
    def empty_to_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


class CattleFilters(Filters):
    species: Text | None = None
    breed: Text | None = None
    district: Text | None = None
    village: Text | None = None
    max_price: Amount | None = None
    min_milk_per_day: Amount | None = None
    milking_status: Text | None = None
    pregnancy_status: Text | None = None


class FodderFilters(Filters):
    q: Text | None = None
    form: Literal["GREEN", "DRY", "HAY", "SILAGE", "CONCENTRATE"] | None = None
    district: Text | None = None
    subdistrict: Text | None = None
    village: Text | None = None
    comparison_group: Literal["green", "dry", "hay", "silage", "concentrate"] | None = None
    delivery: Literal["delivery", "pickup"] | None = None
    max_rate_per_kg: Amount | None = None
    min_qty_kg: Amount | None = None
    sort: Literal["recent", "rate", "qty"] | None = None
    limit: Annotated[int, Field(ge=1, le=20, strict=True)] = 5

    @field_validator("limit", mode="before")
    @classmethod
    def reset_limit(cls, value):
        return 5 if value is None or isinstance(value, str) and not value.strip() else value
