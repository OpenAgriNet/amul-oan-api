"""
Internal backends for farmer and animal data from multiple APIs.
- amulpashudhan.com (PASHUGPT_TOKEN): GetFarmerDetailsByMobile, GetAnimalDetailsByTagNo
- herdman.live (PASHUGPT_TOKEN_3): get-amul-farmer, get-amul-animal

Used by farmer.py and animal.py to provide cohesive tools with fallback and merged output.
"""
from beartype.typing import TypeVar
import json
import re
from typing import Any, Dict, Optional

import httpx
from pydantic import ValidationError

from app.models.farmer import FarmerModel, FarmerHerdmanModel
from helpers.utils import get_logger

logger = get_logger(__name__)

BASE_AMULPASHUDHAN = "https://api.amulpashudhan.com/configman/v1/PashuGPT"
BASE_HERDMAN = "https://herdman.live/apis/api"


def normalize_phone(mobile: str) -> str:
    """Strip non-digits; for Indian numbers optionally strip leading 91."""
    digits = re.sub(r"\D", "", mobile)
    if digits.startswith("91") and len(digits) > 10:
        digits = digits[2:].lstrip("0") or digits
    return digits.lstrip("0") or mobile


def normalize_tag(tag_no: str) -> str:
    """Strip whitespace from tag number."""
    return (tag_no or "").strip()


# --- Farmer ---


async def fetch_farmer_amulpashudhan(
    mobile: str, token: str
) -> list[FarmerModel] | None:
    """Returns list of farmer records or None on 204/error/empty."""
    url = f"{BASE_AMULPASHUDHAN}/GetFarmerDetailsByMobile?mobileNumber={mobile}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            response.raise_for_status()
            logger.info(f"[AmulPashudhan({mobile})] :: Response successfully recieved.")
            r_json = response.json()
            if not isinstance(r_json, list):
                raise Exception("Not a valid list provided in the response.")
            return [
                FarmerModel.model_validate(data, extra="ignore", by_alias=True)
                for data in r_json
            ]
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[AmulPashudhan({mobile})] :: Request failed with status code {e.response.status_code}, and message = {e.response.text}",
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"[AmulPashudhan({mobile})] :: Response didn't gave a valid json, failed due to decoding error {str(e)}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"[AmulPashudhan({mobile})] :: Request failed, due to error {str(e)}",
            exc_info=True,
        )


async def fetch_farmer_herdman(mobile: str, token: str) -> list[FarmerModel] | None:
    """Returns list of farmer records or None on error/empty."""
    url = f"{BASE_HERDMAN}/get-amul-farmer"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params={"mobileno": mobile},
                headers={"accept": "application/json", "api-token": f"Bearer {token}"},
            )
            response.raise_for_status()
            logger.info(f"[Herdman({mobile})] :: Response successfully recieved")
            data = FarmerHerdmanModel.model_validate_json(
                response.text, extra="ignore", by_alias=True
            )
            return data.farmers
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[Herdman({mobile})] :: Request failed with status code {e.response.status_code}, and message = {e.response.text}",
            exc_info=True,
        )
    except ValidationError as e:
        for error in e.errors():
            if error.get("type") == "model_type":
                logger.info(
                    f"[Herdman({mobile})] :: No information from herdman found."
                )
            else:
                logger.error(
                    f"[Herdman({mobile})] :: Failed to validated FarmerHerdmanModel, due to error {e}",
                    exc_info=True,
                )
    except Exception as e:
        logger.error(
            f"[Herdman({mobile})] :: Request failed, due to error {str(e)}",
            exc_info=True,
        )


# --- Animal ---


async def fetch_animal_amulpashudhan(tag_no: str, token: str) -> Optional[Dict[str, Any]]:
    """Returns single animal dict or None on 204/error/empty."""
    url = f"{BASE_AMULPASHUDHAN}/GetAnimalDetailsByTagNo?tagNo={tag_no}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url,
                headers={"accept": "application/json", "Authorization": f"Bearer {token}"},
            )
        if r.status_code == 204 or not (r.text or "").strip():
            return None
        if r.status_code != 200:
            return None
        data = json.loads(r.text)
        if isinstance(data, dict) and data.get("tagNumber"):
            return data
        if isinstance(data, dict) and data.get("tagNo"):
            data["tagNumber"] = data["tagNo"]
            return data
        return None
    except (json.JSONDecodeError, httpx.HTTPError, Exception):
        return None


def _normalize_herdman_animal(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map herdman Animal item to canonical keys."""
    # herdman: tagno, Animal Type, Breed, Milking Stage, DOB, Currant Lactation no, Last AI, Last PD, Last Calvingdate, etc.
    out: Dict[str, Any] = {}
    out["tagNumber"] = raw.get("tagno") or raw.get("tagNumber") or raw.get("TagID")
    out["animalType"] = raw.get("Animal Type") or raw.get("animalType")
    out["breed"] = raw.get("Breed") or raw.get("breed")
    out["milkingStage"] = raw.get("Milking Stage") or raw.get("milkingStage")
    out["pregnancyStage"] = raw.get("pregnancyStage")
    out["dateOfBirth"] = raw.get("DOB") or raw.get("dateOfBirth")
    out["lactationNo"] = raw.get("Currant Lactation no") if "Currant Lactation no" in raw else raw.get("lactationNo")
    out["lastBreedingActivity"] = raw.get("Last AI") or raw.get("lastBreedingActivity")
    out["lastHealthActivity"] = raw.get("lastHealthActivity")
    out["lastPD"] = raw.get("Last PD")
    out["lastCalvingDate"] = raw.get("Last Calvingdate")
    out["farmerComplaint"] = raw.get("Farmer complaint")
    out["diagnosis"] = raw.get("Diagnosis")
    out["medicineGiven"] = raw.get("Medicine Given")
    return {k: v for k, v in out.items() if v is not None}


async def fetch_animal_herdman(tag_no: str, token: str) -> Optional[Dict[str, Any]]:
    """Returns single animal dict (canonical keys) or None on error/empty."""
    url = f"{BASE_HERDMAN}/get-amul-animal"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url,
                params={"TagID": tag_no},
                headers={"accept": "application/json", "api-token": f"Bearer {token}"},
            )
        if r.status_code != 200 or not (r.text or "").strip():
            return None
        data = json.loads(r.text)
        if isinstance(data, dict) and data.get("Animal") and isinstance(data["Animal"], list) and len(data["Animal"]) > 0:
            return _normalize_herdman_animal(data["Animal"][0])
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return _normalize_herdman_animal(data[0])
        if isinstance(data, dict) and (data.get("tagno") or data.get("tagNumber")):
            return _normalize_herdman_animal(data)
        return None
    except (json.JSONDecodeError, httpx.HTTPError, Exception):
        return None


def merge_animal_data(primary: Optional[Dict], fallback: Optional[Dict]) -> Dict[str, Any]:
    """Merge primary (amulpashudhan) with fallback (herdman). Prefer primary; fill missing from fallback."""
    if primary and fallback:
        merged = dict(primary)
        for k, v in fallback.items():
            if v is not None and (merged.get(k) is None or merged.get(k) == ""):
                merged[k] = v
        return merged
    if primary:
        return primary
    if fallback:
        return fallback
    return {}

T = TypeVar("T", bound=FarmerModel)


def _merge_models(u1: T, u2: T, model: type[T]) -> T:
    return model.model_validate(
        {
            k: v2 if v2 is not None else v1
            for k, (v1, v2) in {
                k: (getattr(u1, k), getattr(u2, k)) for k in model.model_fields
            }.items()
        }
    )


def merge_farmer_data(data: list[FarmerModel]) -> list[FarmerModel]:
    seen = {}
    for farmer in data:
        key = f"{farmer.society_name}_{farmer.farmer_name}"
        if key in seen:
            farmer_1 = seen[key]
            seen[key] = _merge_models(farmer_1, farmer, FarmerModel)
        else:
            seen[key] = farmer
    return list(seen.values())
