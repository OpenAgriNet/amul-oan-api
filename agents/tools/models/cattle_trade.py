"""Models for the PashuGPT cattle trade average-rate endpoint."""

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
