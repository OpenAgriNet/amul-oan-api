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

from app.core.cache import (
    build_api_cache_key,
    get_cached_api_response,
    set_cached_api_response,
)
from app.models.ai_call import AICallRequestModel, AICallResponseModel
from app.config import settings
from app.models.animal import AnimalModel
from app.models.banas_visit import BanasOperatedVisitModel
from app.models.cvcc import CvccHealthResponseModel
from app.models.farmer import FarmerModel, FarmerHerdmanModel
from helpers.utils import get_logger

logger = get_logger(__name__)

BASE_AMULPASHUDHAN = "https://api.amulpashudhan.com/configman/v1/PashuGPT"
BASE_HERDMAN = "https://herdman.live/apis/api"
BASE_BANAS_MOBILE = "https://banasmobileapi.amnex.com/api/FarmerVisitAPIKOS"
BASE_CVCC = "https://api.amuldairy.com/ai_cattle_dtl.php"


def normalize_phone(mobile: str) -> str:
    """Strip non-digits; for Indian numbers optionally strip leading 91."""
    digits = re.sub(r"\D", "", mobile)
    if digits.startswith("91") and len(digits) > 10:
        digits = digits[2:].lstrip("0") or digits
    return digits.lstrip("0") or mobile


def _load_json_lenient(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*,", ",", payload)
        cleaned = re.sub(r",\s*(?=[}\]])", "", cleaned)
        return json.loads(cleaned)


# --- Farmer ---


async def fetch_farmer_amulpashudhan(
    mobile: str, token: str
) -> list[FarmerModel] | None:
    """Returns list of farmer records or None on 204/error/empty."""
    cache_key = build_api_cache_key("amulpashudhan_farmer", mobile)
    cache_hit, cached_payload = await get_cached_api_response(cache_key)
    if cache_hit:
        if cached_payload is None:
            return None
        if not isinstance(cached_payload, list):
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid list, refetching.",
                cache_key,
            )
        else:
            try:
                return [
                    FarmerModel.model_validate(data, extra="ignore", by_alias=True)
                    for data in cached_payload
                ]
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached farmer payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

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
            if response.status_code == 204 or not (response.text or "").strip():
                await set_cached_api_response(cache_key, None)
                return None
            r_json = response.json()
            if not isinstance(r_json, list):
                raise Exception("Not a valid list provided in the response.")
            await set_cached_api_response(cache_key, r_json)
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
    cache_key = build_api_cache_key("herdman_farmer", mobile)
    cache_hit, cached_payload = await get_cached_api_response(cache_key)
    if cache_hit:
        if cached_payload is None:
            return None
        if not isinstance(cached_payload, dict):
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid dict, refetching.",
                cache_key,
            )
        else:
            try:
                data = FarmerHerdmanModel.model_validate(
                    cached_payload, extra="ignore", by_alias=True
                )
                return data.farmers
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached herdman payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

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
            if not (response.text or "").strip():
                await set_cached_api_response(cache_key, None)
                return None
            response_json = response.json()
            await set_cached_api_response(cache_key, response_json)
            data = FarmerHerdmanModel.model_validate(
                response_json, extra="ignore", by_alias=True
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


async def fetch_animal_amulpashudhan(tag_no: str, token: str) -> AnimalModel | None:
    """Returns a validated animal model or None on 204/error/empty."""
    cache_key = build_api_cache_key("amulpashudhan_animal", tag_no)
    cache_hit, cached_payload = await get_cached_api_response(cache_key)
    if cache_hit:
        if cached_payload is None:
            return None
        if not isinstance(cached_payload, dict):
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid dict, refetching.",
                cache_key,
            )
        else:
            try:
                cached_data = dict(cached_payload)
                if cached_data.get("tagNo") and not cached_data.get("tagNumber"):
                    cached_data["tagNumber"] = cached_data["tagNo"]
                if cached_data.get("tagNumber") or cached_data.get("tagNo"):
                    return AnimalModel.model_validate(
                        cached_data, extra="ignore", by_alias=True
                    )
                logger.warning(
                    "[Cache(%s)] :: Cached animal payload missing tag number, refetching.",
                    cache_key,
                )
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached animal payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

    url = f"{BASE_AMULPASHUDHAN}/GetAnimalDetailsByTagNo?tagNo={tag_no}"
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
            logger.info(
                f"[AmulPashudhan({tag_no})] :: Response successfully recieved."
            )
        if response.status_code == 204 or not (response.text or "").strip():
            await set_cached_api_response(cache_key, None)
            return None
        if response.status_code != 200:
            return None
        data = json.loads(response.text)
        if not isinstance(data, dict):
            raise Exception("Not a valid dict provided in the response.")
        if data.get("tagNo") and not data.get("tagNumber"):
            data["tagNumber"] = data["tagNo"]
        if data.get("tagNumber") or data.get("tagNo"):
            await set_cached_api_response(cache_key, data)
            return AnimalModel.model_validate(data, extra="ignore", by_alias=True)
        raise Exception("Animal response did not contain a tag number.")
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[AmulPashudhan({tag_no})] :: Request failed with status code {e.response.status_code}, and message = {e.response.text}",
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"[AmulPashudhan({tag_no})] :: Response didn't gave a valid json, failed due to decoding error {str(e)}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"[AmulPashudhan({tag_no})] :: Request failed, due to error {str(e)}",
            exc_info=True,
        )


async def fetch_banas_operated_visit(
    tag_no: str,
) -> list[BanasOperatedVisitModel] | None:
    """Returns operated visit list for a Banas animal tag or None on 204/error/empty."""
    api_key = settings.banas_mobile_api_key
    if not api_key:
        logger.warning("BANAS_MOBILE_API_KEY is not set")
        return None

    cache_key = build_api_cache_key("banas_operated_visit", tag_no)
    cache_hit, cached_payload = await get_cached_api_response(cache_key)
    if cache_hit:
        if cached_payload is None:
            return None
        if not isinstance(cached_payload, list):
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid list, refetching.",
                cache_key,
            )
        else:
            try:
                return [
                    BanasOperatedVisitModel.model_validate(
                        data, extra="ignore", by_alias=True
                    )
                    for data in cached_payload
                ]
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached banas visit payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

    url = f"{BASE_BANAS_MOBILE}/GetOperatedVisit"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"strApiKey": api_key, "tagId": tag_no},
            )
            response.raise_for_status()
            logger.info(f"[BanasOperatedVisit({tag_no})] :: Response successfully recieved.")
        if response.status_code == 204 or not (response.text or "").strip():
            await set_cached_api_response(cache_key, None)
            return None
        response_json = response.json()
        if not isinstance(response_json, list):
            raise Exception("Not a valid list provided in the response.")
        await set_cached_api_response(cache_key, response_json)
        return [
            BanasOperatedVisitModel.model_validate(
                data, extra="ignore", by_alias=True
            )
            for data in response_json
        ]
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[BanasOperatedVisit({tag_no})] :: Request failed with status code {e.response.status_code}, and message = {e.response.text}",
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"[BanasOperatedVisit({tag_no})] :: Response didn't gave a valid json, failed due to decoding error {str(e)}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"[BanasOperatedVisit({tag_no})] :: Request failed, due to error {str(e)}",
            exc_info=True,
        )


async def fetch_cvcc_health_details(
    tag_no: str,
    token: str,
    vendor_no: str = "9999999",
) -> CvccHealthResponseModel | None:
    """Returns validated CVCC health details or None on 204/error/empty."""
    cache_key = build_api_cache_key("cvcc_health", tag_no)
    cache_hit, cached_payload = await get_cached_api_response(cache_key)
    if cache_hit:
        if cached_payload is None:
            return None
        if not isinstance(cached_payload, dict):
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid dict, refetching.",
                cache_key,
            )
        else:
            try:
                return CvccHealthResponseModel.model_validate(
                    cached_payload, extra="ignore", by_alias=True
                )
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached cvcc payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                BASE_CVCC,
                headers={"Content-Type": "application/json"},
                json={
                    "token_no": token,
                    "vendor_no": vendor_no,
                    "tag_no": tag_no,
                },
            )
            response.raise_for_status()
            logger.info(f"[CVCC({tag_no})] :: Response successfully recieved.")
        if response.status_code == 204 or not (response.text or "").strip():
            await set_cached_api_response(cache_key, None)
            return None
        response_json = _load_json_lenient(response.text)
        if not isinstance(response_json, dict):
            raise Exception("Not a valid dict provided in the response.")
        await set_cached_api_response(cache_key, response_json)
        return CvccHealthResponseModel.model_validate(
            response_json, extra="ignore", by_alias=True
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[CVCC({tag_no})] :: Request failed with status code {e.response.status_code}, and message = {e.response.text}",
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"[CVCC({tag_no})] :: Response didn't gave a valid json, failed due to decoding error {str(e)}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"[CVCC({tag_no})] :: Request failed, due to error {str(e)}",
            exc_info=True,
        )


async def create_ai_call_api(
    request: AICallRequestModel, token: str
) -> AICallResponseModel | None:
    """Creates an artificial insemination call and returns the assigned technician."""
    api_url = f"{BASE_AMULPASHUDHAN}/CreateAICall"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                params=request.to_query_params(),
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            logger.info(
                "[CreateAICall(%s,%s,%s,%s)] :: Response successfully recieved.",
                request.union_code,
                request.society_code,
                request.farmer_code,
                request.species.value,
            )
        response_json = response.json()
        if not isinstance(response_json, dict):
            raise Exception("Not a valid dict provided in the response.")
        return AICallResponseModel.model_validate(
            response_json, extra="ignore", by_alias=True
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "[CreateAICall(%s,%s,%s,%s)] :: Request failed with status code %s, and message = %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            e.response.status_code,
            e.response.text,
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            "[CreateAICall(%s,%s,%s,%s)] :: Response didn't gave a valid json, failed due to decoding error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            str(e),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "[CreateAICall(%s,%s,%s,%s)] :: Request failed, due to error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            str(e),
            exc_info=True,
        )


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
            merged = _merge_models(farmer_1, farmer, FarmerModel)
            if farmer_1.union_name is not None:
                merged.union_name = farmer_1.union_name
            seen[key] = merged
        else:
            seen[key] = farmer
    return list(seen.values())
