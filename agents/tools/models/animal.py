from typing import Any
from pydantic import AliasChoices, BaseModel, Field, field_validator


class AnimalModel(BaseModel):
    tag_number: str | None = Field(
        None, validation_alias=AliasChoices("tagNumber", "tagNo")
    )
    animal_type: str | None = Field(None, alias="animalType")
    animal_name: str | None = Field(None, alias="animalName")
    breed: str | None = None
    milking_stage: str | None = Field(None, alias="milkingStage")
    pregnancy_stage: str | None = Field(None, alias="pregnancyStage")
    date_of_birth: str | None = Field(None, alias="dateOfBirth")
    lactation_no: int | None = Field(None, alias="lactationNo")
    last_breeding_activity: dict[str, Any] | None = Field(
        None, alias="lastBreedingActivity"
    )
    last_health_activity: dict[str, Any] | None = Field(
        None, alias="lastHealthActivity"
    )

    @field_validator(
        "animal_type",
        "animal_name",
        "breed",
        "milking_stage",
        "pregnancy_stage",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower()
