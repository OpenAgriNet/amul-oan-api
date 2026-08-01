"""Farmer-domain rendering over a fetched FarmerDataEnvelope.

Surface-agnostic: used by the chat loan tool and by the voice turn builder.
"""
from typing import Optional

from agents.deps import FarmerAccount
from agents.models.farmer import FarmerDataEnvelope


def collect_farmer_accounts(envelope: Optional[FarmerDataEnvelope]) -> list[FarmerAccount]:
    """Extract every (union, society, farmer) account on the caller's mobile.

    A mobile can map to multiple PashuGPT accounts (e.g. a cow account and a
    buffalo account). The milk-collection tool fans out over all of these so
    a farmer's data is never missed because the agent picked one account.
    Deduplicated on (union_code, society_code, farmer_code).
    """
    if envelope is None:
        return []

    seen: set[tuple] = set()
    accounts: list[FarmerAccount] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        union_code = record.get("unionCode") or record.get("union_code")
        society_code = record.get("societyCode") or record.get("society_code")
        farmer_code = record.get("farmerCode") or record.get("farmer_code")
        if not (union_code and society_code and farmer_code):
            continue
        key = (str(union_code), str(society_code), str(farmer_code))
        if key in seen:
            continue
        seen.add(key)
        accounts.append(
            FarmerAccount(
                union_code=str(union_code),
                society_code=str(society_code),
                farmer_code=str(farmer_code),
                farmer_name=record.get("farmerName") or record.get("farmer_name"),
                society_name=record.get("societyName") or record.get("society_name"),
            )
        )
    return accounts
