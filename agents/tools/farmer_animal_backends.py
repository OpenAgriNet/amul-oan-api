"""
Internal backends for farmer and animal data from multiple APIs.
- amulpashudhan.com (PASHUGPT_TOKEN): GetFarmerDetailsByMobile, GetAnimalDetailsByTagNo
- herdman.live (PASHUGPT_TOKEN_3): get-amul-farmer, get-amul-animal

Used by farmer.py and animal.py to provide cohesive tools with fallback and merged output.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

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


async def fetch_farmer_amulpashudhan(mobile: str, token: str) -> Optional[List[Dict[str, Any]]]:
    """Returns list of farmer records or None on 204/error/empty."""
    url = f"{BASE_AMULPASHUDHAN}/GetFarmerDetailsByMobile?mobileNumber={mobile}"
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
        if isinstance(data, list) and len(data) > 0:
            return data
        if isinstance(data, dict) and data.get("data") and isinstance(data["data"], list):
            return data["data"]
        return None
    except (json.JSONDecodeError, httpx.HTTPError, Exception):
        return None


async def fetch_farmer_herdman(mobile: str, token: str) -> Optional[List[Dict[str, Any]]]:
    """Returns list of farmer records or None on error/empty."""
    url = f"{BASE_HERDMAN}/get-amul-farmer"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url,
                params={"mobileno": mobile},
                headers={"accept": "application/json", "api-token": f"Bearer {token}"},
            )
        if r.status_code != 200 or not (r.text or "").strip():
            return None
        data = json.loads(r.text)
        if isinstance(data, list) and len(data) > 0:
            return data
        if isinstance(data, dict) and data.get("data") and isinstance(data["data"], list):
            return data["data"]
        return None
    except (json.JSONDecodeError, httpx.HTTPError, Exception):
        return None


def _farmer_record_key(rec: Dict[str, Any]) -> tuple:
    """Key for deduplication: societyName + farmerCode."""
    return (str(rec.get("societyName") or ""), str(rec.get("farmerCode") or ""))


def merge_farmer_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by societyName+farmerCode; drop entries that are all nulls."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for rec in records:
        if not rec:
            continue
        key = _farmer_record_key(rec)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


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
