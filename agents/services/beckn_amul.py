"""Typed chat-side adapters for private data served by the single Amul BPP.

This module is deliberately the only agent-layer code that understands the
normalized Beckn tags emitted by the Amul provider mappers.  Callers pass only
identifiers resolved from the authenticated session; provider credentials and
legacy encrypted identifiers remain inside the BPP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from app.models.animal import AnimalModel
from app.models.banas_visit import BanasOperatedVisitModel
from app.models.cvcc import CvccHealthResponseModel
from app.models.farmer import FarmerModel
from app.models.milk_collection import FarmerMilkCollectionResponseModel
from app.services.beckn_operations import (
    BecknActionResult,
    OperationState,
    get_beckn_operation_client,
)
FARMER_DOMAIN = "data:amul-farmer-profile"
ANIMAL_DOMAIN = "data:amul-animal-profile"
BOOKING_DOMAIN = "services:amul-vet-booking"
MILK_DOMAIN = "services:amul-milk-collection"


class BecknProviderUnavailable(RuntimeError):
    """The BPP did not return a completed business response."""


@dataclass(frozen=True)
class AuthenticatedFarmerAccount:
    union_code: str
    society_code: str
    farmer_code: str
    farmer_name: Optional[str] = None
    society_name: Optional[str] = None


class AITechnicianRecord(BaseModel):
    """Normalized technician item returned by booking on_search."""

    model_config = ConfigDict(extra="ignore")

    userId: Optional[str] = None
    fullName: Optional[str] = None
    mobileNumber: Optional[str] = None


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _code(value: Any) -> str:
    node = _record(value)
    descriptor = _record(node.get("descriptor"))
    return str(descriptor.get("code") or node.get("code") or "")


def _groups(tags: Any, code: str) -> list[dict[str, Any]]:
    if not isinstance(tags, list):
        return []
    return [tag for tag in tags if isinstance(tag, dict) and _code(tag) == code]


def _values(group: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    entries = group.get("list")
    if not isinstance(entries, list):
        return values
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("value") is None:
            continue
        values[_code(entry)] = str(entry["value"])
    return values


def _nested_values(group: Mapping[str, Any], code: str) -> list[str]:
    """Read repeated scalar tags or a nested tag group with the same code."""
    entries = group.get("list")
    if not isinstance(entries, list):
        return []
    result: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or _code(entry) != code:
            continue
        if entry.get("value") is not None:
            result.append(str(entry["value"]).strip())
        nested = entry.get("list")
        if isinstance(nested, list):
            result.extend(
                str(child.get("value")).strip()
                for child in nested
                if isinstance(child, dict) and child.get("value") is not None
            )
    return [value for value in result if value]


def _completed_payload(result: BecknActionResult, label: str) -> dict[str, Any]:
    operation = result.operation
    if operation.state in {OperationState.NACKED, OperationState.BUSINESS_FAILED}:
        error = _record(result.payload).get("error")
        message = _record(error).get("message") if isinstance(error, dict) else None
        raise BecknProviderUnavailable(str(message or f"{label} provider rejected the request"))
    if operation.state is OperationState.TIMED_OUT_PENDING:
        raise BecknProviderUnavailable(f"{label} callback is still pending")
    if operation.state is not OperationState.SUCCEEDED or not isinstance(result.payload, dict):
        raise BecknProviderUnavailable(f"{label} callback was not completed")
    return result.payload


def _order(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _record(_record(payload.get("message")).get("order"))


def _farmer_models_from_payload(
    payload: Mapping[str, Any], *, authenticated_mobile: str
) -> list[FarmerModel]:
    order = _order(payload)
    if order.get("state") == "NOT_FOUND":
        return []
    fulfillments = order.get("fulfillments")
    fulfillment = _record(fulfillments[0]) if isinstance(fulfillments, list) and fulfillments else {}
    customer = _record(fulfillment.get("customer"))
    person = _record(customer.get("person"))
    tags = person.get("tags")
    account_groups = _groups(tags, "farmer_accounts")
    global_tags = [
        value
        for group in _groups(tags, "animal_tags")
        for value in _nested_values(group, "tag_id")
    ]

    aliases = {
        "sub_district": "subDistrict",
        "union_name": "unionName",
        "union_code": "unionCode",
        "society_name": "societyName",
        "society_code": "societyCode",
        "farmer_name": "farmerName",
        "farmer_code": "farmerCode",
        "average_milk_cow": "avgMilkPerDayCow",
        "average_milk_buffalo": "avgMilkPerDayBuff",
        "cow_snf": "cowSnf",
        "cow_fat": "cowFat",
        "buffalo_snf": "buffSnf",
        "buffalo_fat": "buffFat",
        "total_animals": "totalAnimals",
        "total_cows": "cow",
        "total_buffaloes": "buffalo",
        "total_milking_animals": "totalMilkingAnimals",
    }
    models: list[FarmerModel] = []
    for index, group in enumerate(account_groups):
        raw = _values(group)
        mapped = {aliases.get(key, key): value for key, value in raw.items()}
        owned_tags = _nested_values(group, "tag_id") + _nested_values(group, "animal_tags")
        # Compatibility with the first mapper revision, which emitted global
        # tags. It is safe only for a single account because ownership would be
        # ambiguous across multiple union accounts.
        if not owned_tags and len(account_groups) == 1:
            owned_tags = global_tags
        if owned_tags:
            mapped["tagNo"] = ",".join(dict.fromkeys(owned_tags))
        mapped["mobileNumber"] = authenticated_mobile
        try:
            models.append(FarmerModel.model_validate(mapped))
        except Exception as exc:
            raise BecknProviderUnavailable(
                f"farmer profile mapper returned an invalid account at index {index}"
            ) from exc
    return models


async def fetch_authenticated_farmers(
    mobile: str,
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> list[FarmerModel]:
    """Fetch PashuGPT and applicable Herdman accounts via init/on_init."""
    client = get_beckn_operation_client()
    primary = _completed_payload(
        await client.init_farmer_profile(
            provider_id="amulpashudhan",
            mobile=mobile,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "farmer profile",
    )
    farmers = _farmer_models_from_payload(primary, authenticated_mobile=mobile)
    if not any((farmer.union_name or "").casefold() == "mehsana" for farmer in farmers):
        return farmers

    herdman = _completed_payload(
        await client.init_farmer_profile(
            provider_id="herdman",
            mobile=mobile,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "farmer profile",
    )
    from agents.tools.farmer_animal_backends import merge_farmer_data

    return merge_farmer_data(
        farmers + _farmer_models_from_payload(herdman, authenticated_mobile=mobile)
    )


def authenticated_accounts(farmers: Iterable[FarmerModel]) -> list[AuthenticatedFarmerAccount]:
    seen: set[tuple[str, str, str]] = set()
    accounts: list[AuthenticatedFarmerAccount] = []
    for farmer in farmers:
        key = (
            str(farmer.union_code or "").strip(),
            str(farmer.society_code or "").strip(),
            str(farmer.farmer_code or "").strip(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        accounts.append(
            AuthenticatedFarmerAccount(
                *key,
                farmer_name=farmer.farmer_name,
                society_name=farmer.society_name,
            )
        )
    return accounts


async def resolve_authenticated_account(
    mobile: str,
    *,
    union_code: str,
    society_code: str,
    farmer_code: str,
    session_id: Optional[str],
    tool_call_id: Optional[str],
) -> Optional[AuthenticatedFarmerAccount]:
    """Match a proposed booking account against a fresh signed-session profile.

    The model-facing booking schema still carries the three codes for selecting
    one of a farmer's possible accounts, but they are never forwarded until this
    exact ownership check succeeds. The returned canonical values are the ones
    used in the Beckn confirm.
    """
    requested = tuple(str(value).strip() for value in (union_code, society_code, farmer_code))
    farmers = await fetch_authenticated_farmers(
        mobile,
        session_id=session_id,
        tool_call_id=tool_call_id,
    )
    for account in authenticated_accounts(farmers):
        if (account.union_code, account.society_code, account.farmer_code) == requested:
            return account
    return None


def _activity(value: Optional[str]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {"summary": value}
    return parsed if isinstance(parsed, dict) else {"summary": value}


def _animal_groups(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    order = _order(payload)
    if order.get("state") == "NOT_FOUND":
        return []
    items = order.get("items")
    item = _record(items[0]) if isinstance(items, list) and items else {}
    return item.get("tags") if isinstance(item.get("tags"), list) else []


def _animal_model(tags: list[dict[str, Any]]) -> Optional[AnimalModel]:
    groups = _groups(tags, "animal_profile")
    if not groups:
        return None
    raw = _values(groups[0])
    mapped: dict[str, Any] = {
        "tagNumber": raw.get("tag_id"),
        "animalType": raw.get("animal_type"),
        "animalName": raw.get("animal_name"),
        "breed": raw.get("breed"),
        "milkingStage": raw.get("milking_stage"),
        "pregnancyStage": raw.get("pregnancy_stage"),
        "dateOfBirth": raw.get("date_of_birth"),
        "lastBreedingActivity": _activity(raw.get("last_breeding_activity")),
        "lastHealthActivity": _activity(raw.get("last_health_activity")),
    }
    if raw.get("lactation_number"):
        mapped["lactationNo"] = int(float(raw["lactation_number"]))
    return AnimalModel.model_validate(mapped)


def _merge_animals(primary: Optional[AnimalModel], fallback: Optional[AnimalModel]) -> Optional[AnimalModel]:
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    data = primary.model_dump()
    for key, value in fallback.model_dump().items():
        if data.get(key) in (None, "", [], {}):
            data[key] = value
    return AnimalModel.model_validate(data)


async def fetch_animal_profile(
    tag_id: str,
    *,
    union_name: Optional[str],
    union_code: Optional[str],
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Optional[AnimalModel]:
    client = get_beckn_operation_client()
    primary = _completed_payload(
        await client.init_animal_profile(
            provider_id="amulpashudhan",
            tag_id=tag_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "animal profile",
    )
    animal = _animal_model(_animal_groups(primary))
    if (union_name or "").casefold() != "mehsana":
        return animal
    herdman = _completed_payload(
        await client.init_animal_profile(
            provider_id="herdman",
            tag_id=tag_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "animal profile",
    )
    return _merge_animals(animal, _animal_model(_animal_groups(herdman)))


async def fetch_cvcc_health(
    tag_id: str,
    *,
    union_code: str,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Optional[CvccHealthResponseModel]:
    result = _completed_payload(
        await get_beckn_operation_client().init_animal_profile(
            provider_id="amuldairy",
            tag_id=tag_id,
            union_code=union_code,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "animal health history",
    )
    tags = _animal_groups(result)
    profiles = _groups(tags, "animal_profile")
    if not profiles and not _groups(tags, "treatment"):
        return None
    profile = _values(profiles[0]) if profiles else {}
    medicines_by_treatment: dict[int, list[dict[str, Any]]] = {}
    for values in map(_values, _groups(tags, "treatment_medicine")):
        index = int(values.get("treatment_index", "0"))
        medicines_by_treatment.setdefault(index, []).append({
            "Medicine Name": values.get("medicine_name"),
            "Medicine Dose": values.get("medicine_dose"),
            "Medicine Route": values.get("medicine_route"),
        })
    fodder_by_treatment: dict[int, list[dict[str, Any]]] = {}
    for values in map(_values, _groups(tags, "fodder_detail")):
        index = int(values.get("treatment_index", "0"))
        fodder_by_treatment.setdefault(index, []).append({
            "Fodder Group": values.get("fodder_group"),
            "Fodder Name": values.get("fodder_name"),
            "Fodder QTY (Kg.)": values.get("quantity_kg"),
        })
    treatments = []
    for fallback_index, values in enumerate(map(_values, _groups(tags, "treatment"))):
        index = int(values.get("treatment_index", fallback_index))
        treatments.append({
            "symptom": values.get("symptom"),
            "Treatment Date": values.get("treatment_date"),
            "treatment": values.get("treatment"),
            "medicine": medicines_by_treatment.get(index, []),
            "Fodder Detail": fodder_by_treatment.get(index, []),
        })
    vaccinations = [
        {
            "vaccine Name": values.get("vaccine_name"),
            "vaccination Type": values.get("vaccination_type"),
            "vaccination Date": values.get("vaccination_date"),
            "vaccine For Disease": values.get("disease"),
        }
        for values in map(_values, _groups(tags, "vaccination"))
    ]
    deworming = [
        {
            "Deworming Date": values.get("deworming_date"),
            "Dewormer Name": values.get("dewormer_name"),
            "Dewormer Content": values.get("dewormer_content"),
            "Dewormer Dose": values.get("dewormer_dose"),
        }
        for values in map(_values, _groups(tags, "deworming"))
    ]
    return CvccHealthResponseModel.model_validate({
        "msg": "success",
        "data": {
            "Tag": profile.get("tag_id"),
            "Animal Type": profile.get("animal_type"),
            "breed": profile.get("breed"),
            "Milking Stage": profile.get("milking_stage"),
            "Pregnancy Stage ": profile.get("pregnancy_stage"),
            "Lactation": profile.get("lactation_number"),
            "Milk Yield": profile.get("milk_yield"),
            "Treatment": treatments,
            "Vaccination": vaccinations,
            "Deworming": deworming,
        },
    })


async def fetch_banas_visits(
    tag_id: str,
    *,
    union_code: str,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> list[BanasOperatedVisitModel]:
    result = _completed_payload(
        await get_beckn_operation_client().init_animal_profile(
            provider_id="banasmobileapi",
            tag_id=tag_id,
            union_code=union_code,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "operated visit history",
    )
    tags = _animal_groups(result)
    medicines_by_visit: dict[int, list[dict[str, Any]]] = {}
    for values in map(_values, _groups(tags, "visit_medicine")):
        index = int(values.get("visit_index", "0"))
        medicines_by_visit.setdefault(index, []).append({
            "medicinename": values.get("medicine_name"),
            "stock": values.get("stock"),
            "remarks": values.get("remarks"),
            "uomdoctor": values.get("unit"),
        })
    reports_by_visit: dict[int, list[dict[str, Any]]] = {}
    for values in map(_values, _groups(tags, "lab_report")):
        index = int(values.get("visit_index", "0"))
        reports_by_visit.setdefault(index, []).append({
            "srno": values.get("sequence"),
            "sampledate": values.get("sample_date"),
            "samplename": values.get("sample_name"),
            "remarks": values.get("remarks"),
        })
    visits: list[BanasOperatedVisitModel] = []
    for fallback_index, values in enumerate(map(_values, _groups(tags, "operated_visit"))):
        index = int(values.get("visit_index", fallback_index))
        visits.append(BanasOperatedVisitModel.model_validate({
            "VisitCode": values.get("visit_code"),
            "VisitNoteDate": values.get("visit_date"),
            "VisitScheduleDate": values.get("scheduled_date"),
            "speciesname": values.get("species"),
            "gendername": values.get("gender"),
            "pregnancystatus": values.get("pregnancy_status"),
            "breed": values.get("breed"),
            "milkstatus": values.get("milk_status"),
            "Ailment1": values.get("ailment_1"),
            "Ailment2": values.get("ailment_2"),
            "Ailment3": values.get("ailment_3"),
            "DiseaseName": values.get("disease"),
            "diseasegroup": values.get("disease_group"),
            "medicineremarks": values.get("medicine_remarks"),
            "prognosisdetails": values.get("prognosis"),
            "VisitStatus": values.get("visit_status"),
            "MedicinesJson": medicines_by_visit.get(index, []),
            "LabReportsJson": reports_by_visit.get(index, []),
        }))
    return visits


async def search_ai_technicians(
    *,
    union_code: str,
    society_code: str,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> list[AITechnicianRecord]:
    payload = _completed_payload(
        await get_beckn_operation_client().search_ai_technicians(
            union_code=union_code,
            society_code=society_code,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "AI technician discovery",
    )
    catalog = _record(_record(payload.get("message")).get("catalog"))
    providers = catalog.get("providers") or catalog.get("bpp/providers") or []
    technicians: list[AITechnicianRecord] = []
    for provider in providers if isinstance(providers, list) else []:
        for item in _record(provider).get("items") or []:
            item = _record(item)
            details = _groups(item.get("tags"), "technician-details")
            values = _values(details[0]) if details else {}
            item_id = str(item.get("id") or "")
            technician_id = values.get("technician_id") or (
                item_id[4:] if item_id.startswith("ait:") else None
            )
            name = _record(item.get("descriptor")).get("name")
            if isinstance(name, str) and name.endswith(" (AI technician)"):
                name = name.removesuffix(" (AI technician)")
            if technician_id:
                technicians.append(AITechnicianRecord(
                    userId=technician_id,
                    fullName=str(name or ""),
                    mobileNumber=values.get("mobile"),
                ))
    return technicians


async def fetch_milk_collection(
    account: AuthenticatedFarmerAccount,
    *,
    fromdate: str,
    todate: str,
    session_id: Optional[str],
    tool_call_id: Optional[str],
) -> FarmerMilkCollectionResponseModel:
    payload = _completed_payload(
        await get_beckn_operation_client().init_milk_collection(
            union_code=account.union_code,
            society_code=account.society_code,
            farmer_code=account.farmer_code,
            fromdate=fromdate,
            todate=todate,
            session_id=session_id,
            tool_call_id=tool_call_id,
        ),
        "milk collection",
    )
    order = _order(payload)
    items = order.get("items")
    item = _record(items[0]) if isinstance(items, list) and items else {}
    tags = item.get("tags")
    result = None
    query_groups = _groups(tags, "query-period")
    if query_groups:
        result = _values(query_groups[0]).get("result")
    milk = [_values(group) for group in _groups(tags, "milk-record")]
    deduction = []
    for group in _groups(tags, "deduction-record"):
        values = _values(group)
        if "account_name" in values and "accountname" not in values:
            values["accountname"] = values.pop("account_name")
        deduction.append(values)
    return FarmerMilkCollectionResponseModel.model_validate({
        "result": result,
        "milk": milk,
        "deduction": deduction,
    })
