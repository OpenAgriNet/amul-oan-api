import re

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AISpecies(str, Enum):
    COW = "cow"
    BUFFALO = "buffalo"

    @property
    def encrypted_species_id(self) -> str:
        if self is AISpecies.COW:
            return "/cT4TzbfxFOo+L+ZN9x1ZQ=="
        return "M/3Ahr/kOi5ks+Bb5w2uoA=="


class AICallRequestModel(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")
    species: AISpecies

    def to_query_params(self) -> dict[str, str]:
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
            "speciesId": self.species.encrypted_species_id,
        }


class AICallResponseModel(BaseModel):
    ait_name: str | None = Field(None, alias="aitName")
    ticket_number: str | None = Field(None, alias="ticketNumber")

    @staticmethod
    def _normalize_phone(value: str) -> str:
        digits = re.sub(r"\D", "", value)
        if digits.startswith("91") and len(digits) > 10:
            digits = digits[2:].lstrip("0") or digits
        return digits.lstrip("0") or value

    @classmethod
    def _normalize_ait_phone(cls, value: str) -> str:
        if "(" not in value:
            return value
        phone_part, rest = value.split("(", 1)
        normalized_phone = cls._normalize_phone(phone_part.strip())
        return f"{normalized_phone}({rest}"

    @field_validator("ait_name", mode="before")
    @classmethod
    def normalize_ait_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._normalize_ait_phone(value)
