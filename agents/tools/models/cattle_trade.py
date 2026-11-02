"""Models for the PashuGPT cattle trade average-rate endpoint."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class BreedAverageRateModel(BaseModel):
    """One named-breed average-rate row."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    breed_name: str = Field(..., alias="breedName", min_length=1)
    average_rate: Decimal = Field(..., alias="averageRate", ge=0)


class CattleTradeRatesModel(BaseModel):
    """Species averages plus the provider's named-breed rows."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    female_cow: Decimal = Field(..., alias="femaleCow", ge=0)
    female_buffalo: Decimal = Field(..., alias="femaleBuffalo", ge=0)
    breeds: list[BreedAverageRateModel]


class VerifiedAnimalModel(BaseModel):
    """One verified cattle-trade listing."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    district_name: str | None = Field(None, alias="districtName")
    village_name: str | None = Field(None, alias="villageName")
    farmer_name: str | None = Field(None, alias="farmerName")
    farmer_mobile_number: str | None = Field(None, alias="farmerMobileNumber")
    cattle_type: str = Field(..., alias="cattleType", min_length=1)
    breed: str | None = None
    lactation: str | None = None
    age: str | None = None
    milking_status: str | None = Field(None, alias="milkingStatus")
    pregnancy_status: str | None = Field(None, alias="pregnancyStatus")
    milk_per_day: Decimal = Field(..., alias="milkPerDay", ge=0)
    cattle_price: Decimal = Field(..., alias="cattlePrice", ge=0)


class VerifiedAnimalBreedGroupModel(BaseModel):
    """Verified listings grouped by provider breed."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    breed_name: str = Field(..., alias="breedName", min_length=1)
    count: int = Field(..., ge=0)
    animals: list[VerifiedAnimalModel]


class VerifiedAnimalDetailsModel(BaseModel):
    """Verified cow and buffalo listings for the provider's rolling window."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    from_date: datetime = Field(..., alias="fromDate")
    to_date: datetime = Field(..., alias="toDate")
    total_cows: int = Field(..., alias="totalCows", ge=0)
    total_buffaloes: int = Field(..., alias="totalBuffaloes", ge=0)
    cows: list[VerifiedAnimalBreedGroupModel]
    buffaloes: list[VerifiedAnimalBreedGroupModel]
