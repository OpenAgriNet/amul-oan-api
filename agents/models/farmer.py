"""
Typed models for farmer and animal data from PashuGPT APIs.
Used throughout both chat and voice backends for consistent data handling.
"""
from datetime import datetime, timezone
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict


class AnimalRecord(BaseModel):
    """Canonical animal record — fields normalized from amulpashudhan and herdman APIs."""
    model_config = ConfigDict(extra="allow")

    tagNumber: Optional[str] = None
    animalType: Optional[str] = None
    breed: Optional[str] = None
    milkingStage: Optional[str] = None
    pregnancyStage: Optional[str] = None
    dateOfBirth: Optional[str] = None
    lactationNo: Optional[Union[int, str]] = None
    lastBreedingActivity: Optional[str] = None
    lastHealthActivity: Optional[str] = None
    lastPD: Optional[str] = None
    lastCalvingDate: Optional[str] = None
    farmerComplaint: Optional[str] = None
    diagnosis: Optional[str] = None
    medicineGiven: Optional[str] = None


class FarmerRecord(BaseModel):
    """Single farmer record from PashuGPT APIs."""
    model_config = ConfigDict(extra="allow")

    farmerName: Optional[str] = None
    societyName: Optional[str] = None
    farmerCode: Optional[str] = None
    totalAnimals: Optional[int] = None
    tagNo: Optional[str] = None
    tagNumbers: Optional[str] = None


class FarmerSummary(BaseModel):
    """Lightweight farmer summary — safe to embed in JWT payload."""
    farmerName: Optional[str] = None
    societyName: Optional[str] = None
    farmerCode: Optional[str] = None
    totalAnimals: Optional[int] = None
    recordCount: int = 0


class FarmerDataEnvelope(BaseModel):
    """Consistent wrapper used by both chat and voice backends."""
    farmers: List[FarmerRecord] = []
    fetchedAt: Optional[str] = None
    source: Optional[str] = None  # "cache" | "api"

    @classmethod
    def from_records(cls, records: list, source: str = "api") -> "FarmerDataEnvelope":
        """Create envelope from raw record dicts or Pydantic models."""
        farmers = [
            FarmerRecord.model_validate(r if isinstance(r, dict) else r.model_dump())
            for r in records
        ]
        return cls(
            farmers=farmers,
            fetchedAt=datetime.now(timezone.utc).isoformat(),
            source=source,
        )

    def to_summary(self) -> FarmerSummary:
        """Extract lightweight summary for JWT embedding."""
        first = self.farmers[0] if self.farmers else None
        return FarmerSummary(
            farmerName=first.farmerName if first else None,
            societyName=first.societyName if first else None,
            farmerCode=first.farmerCode if first else None,
            totalAnimals=first.totalAnimals if first else None,
            recordCount=len(self.farmers),
        )
