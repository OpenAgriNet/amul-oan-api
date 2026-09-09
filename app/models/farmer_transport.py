"""Transport and cache models for farmer data from PashuGPT APIs.

These camelCase records deliberately remain separate from
``app.models.farmer.FarmerModel``, the normalized snake_case domain model. They
live beside that model so the two representations and their conversion boundary
are explicit instead of being split between ``app.models`` and ``agents.models``.
"""
from datetime import datetime, timezone
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict


class AnimalRecord(BaseModel):
    """Canonical animal record normalized from amulpashudhan and herdman APIs."""

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
    """Single farmer transport record accepting camelCase and snake_case keys."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    farmerName: Optional[str] = None
    societyName: Optional[str] = None
    farmerCode: Optional[str] = None
    totalAnimals: Optional[int] = None
    tagNo: Optional[str] = None
    tagNumbers: Optional[str] = None

    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Map normalized ``FarmerModel`` dumps back to the transport shape."""
        if isinstance(obj, dict):
            mapped = dict(obj)
            snake_to_camel = {
                "farmer_name": "farmerName",
                "society_name": "societyName",
                "farmer_code": "farmerCode",
                "total_animals": "totalAnimals",
            }
            for snake, camel in snake_to_camel.items():
                if snake in mapped and camel not in mapped:
                    mapped[camel] = mapped.pop(snake)
            if "animal_tags" in mapped and "tagNo" not in mapped:
                tags = mapped.pop("animal_tags")
                mapped["tagNo"] = ",".join(tags) if isinstance(tags, list) else tags
            obj = mapped
        return super().model_validate(obj, **kwargs)


class FarmerSummary(BaseModel):
    """Lightweight farmer summary safe to embed in a JWT payload."""

    farmerName: Optional[str] = None
    societyName: Optional[str] = None
    farmerCode: Optional[str] = None
    totalAnimals: Optional[int] = None
    recordCount: int = 0


class FarmerDataEnvelope(BaseModel):
    """Cache envelope around raw farmer transport records.

    ``FarmerModel`` and ``FarmerRecord`` describe different boundaries. The
    former is normalized for domain use; this envelope preserves the camelCase
    cache/API representation. ``from_records`` is the explicit bridge between
    them.
    """

    farmers: List[FarmerRecord] = []
    aiTechnicians: List[dict] = []
    fetchedAt: Optional[str] = None
    source: Optional[str] = None
    stale: bool = False
    staleReason: Optional[str] = None
    refreshAfter: Optional[str] = None
    lookupStatus: Optional[str] = None

    @classmethod
    def from_records(
        cls,
        records: list,
        source: str = "api",
        lookup_status: str = "found",
    ) -> "FarmerDataEnvelope":
        """Create an envelope from raw dictionaries or Pydantic models."""
        farmers = [
            FarmerRecord.model_validate(
                record if isinstance(record, dict) else record.model_dump()
            )
            for record in records
        ]
        return cls(
            farmers=farmers,
            fetchedAt=datetime.now(timezone.utc).isoformat(),
            source=source,
            lookupStatus=lookup_status,
        )

    @classmethod
    def not_found(cls, source: str = "api") -> "FarmerDataEnvelope":
        """Return an empty envelope tagged as a confirmed upstream miss."""
        return cls(
            farmers=[],
            fetchedAt=datetime.now(timezone.utc).isoformat(),
            source=source,
            lookupStatus="not_found",
        )

    @classmethod
    def unknown(cls, source: str = "api") -> "FarmerDataEnvelope":
        """Return an empty envelope for ambiguous/failed upstream lookups."""
        return cls(
            farmers=[],
            fetchedAt=datetime.now(timezone.utc).isoformat(),
            source=source,
            lookupStatus="unknown",
        )

    def to_summary(self) -> FarmerSummary:
        """Extract the first farmer's lightweight JWT summary."""
        first = self.farmers[0] if self.farmers else None
        return FarmerSummary(
            farmerName=first.farmerName if first else None,
            societyName=first.societyName if first else None,
            farmerCode=first.farmerCode if first else None,
            totalAnimals=first.totalAnimals if first else None,
            recordCount=len(self.farmers),
        )
