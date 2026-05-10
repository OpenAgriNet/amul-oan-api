"""Health call booking models."""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from voice.agents.models.ai_call import AISpecies


class HealthCaseType(str, Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"

    @property
    def case_type_id(self) -> str:
        if self is HealthCaseType.EMERGENCY:
            return "M/3Ahr/kOi5ks+Bb5w2uoA=="
        return "/cT4TzbfxFOo+L+ZN9x1ZQ=="


class HealthCallRequestModel(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")
    farmer_code: str = Field(..., alias="farmerCode")
    species: AISpecies
    case_type: HealthCaseType = Field(..., alias="caseType")
    remark: str | None = Field(None, alias="remark")

    def to_query_params(self) -> dict[str, str]:
        params = {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
            "farmerCode": self.farmer_code,
            "speciesId": self.species.encrypted_species_id,
            "caseTypeId": self.case_type.case_type_id,
        }
        if self.remark:
            params["remark"] = self.remark
        return params


class HealthCallResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticket_number: str | None = Field(None, alias="ticketNumber")
