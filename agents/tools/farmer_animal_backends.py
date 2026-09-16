"""
Internal backends for farmer and animal data from the PashuGPT APIs.
- amulpashudhan.com (PASHUGPT_TOKEN): GetFarmerDetailsByMobile, GetAnimalDetailsByTagNo,
  FarmerMilkCollectionDetails, GetFarmerBonusAmount, CreateAICall, CreateHealthCall

Used by farmer.py and animal.py to provide cohesive tools with merged output.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from beartype.typing import TypeVar
import json
import re
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.cache import (
    build_api_cache_key,
    cache,
    get_cached_api_response,
    set_cached_api_response,
)
from app.models.ai_call import AICallRequestModel, AICallResponseModel
from app.models.health_call import HealthCallRequestModel, HealthCallResponseModel
from app.models.bonus import (
    FarmerBonusAmountRecordModel,
    FarmerBonusAmountRequestModel,
)
from app.models.milk_collection import (
    FarmerMilkCollectionRequestModel,
    FarmerMilkCollectionResponseModel,
)
from app.config import settings
from app.models.animal import AnimalModel
from app.models.banas_visit import BanasOperatedVisitModel
from app.models.cvcc import CvccHealthResponseModel
from app.models.farmer import FarmerModel
from helpers.utils import get_logger
from app.observability import start_observation

logger = get_logger(__name__)

BASE_AMULPASHUDHAN = settings.amulpashudhan_base_url
BASE_BANAS_MOBILE = settings.banas_mobile_base_url
BASE_CVCC = settings.cvcc_base_url
FARMER_BACKEND_HTTP_TIMEOUT_SECONDS = settings.farmer_backend_http_timeout_seconds


def _use_farmer_mobile_api_cache() -> bool:
    return not settings.farmer_layer1_mobile_cache_bypass_enabled


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


async def _fetch_farmer_amulpashudhan_raw(
    mobile: str, token: str, *, skip_cache: bool = False
) -> list[dict] | None:
    """Raw amulpashudhan farmer fetch — returns the raw API dicts (camelCase) from
    cache or HTTP, WITHOUT model validation. Shared by fetch_farmer_amulpashudhan
    (→ FarmerModel domain path) and fetch_farmer_info_raw (→ FarmerRecord cache path,
    Option B: cache stays camelCase). Returns None on 204/empty/error. skip_cache
    forces a fresh HTTP fetch (used when cached data fails downstream validation)."""
    cache_key = build_api_cache_key("amulpashudhan_farmer", mobile)
    use_cache = _use_farmer_mobile_api_cache()
    if use_cache and not skip_cache:
        cache_hit, cached_payload = await get_cached_api_response(cache_key)
        if cache_hit:
            if cached_payload is None:
                return None
            if isinstance(cached_payload, list):
                return cached_payload
            logger.warning(
                "[Cache(%s)] :: Cached payload is not a valid list, refetching.",
                cache_key,
            )

    url = f"{BASE_AMULPASHUDHAN}/GetFarmerDetailsByMobile?mobileNumber={mobile}"
    try:
        with start_observation(
            "fetch_farmer_amulpashudhan",
            input={"mobile": mobile},
            metadata={"provider": "amulpashudhan", "url": url},
        ) as observation:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    url,
                    headers={
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=url)
                response.raise_for_status()
                logger.info(f"[AmulPashudhan({mobile})] :: Response successfully recieved.")
                if response.status_code == 204 or not (response.text or "").strip():
                    if use_cache:
                        await set_cached_api_response(cache_key, None)
                    return None
                r_json = response.json()
                if not isinstance(r_json, list):
                    raise Exception("Not a valid list provided in the response.")
                if use_cache:
                    await set_cached_api_response(cache_key, r_json)
                return r_json
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
    return None


async def fetch_farmer_amulpashudhan(
    mobile: str, token: str
) -> list[FarmerModel] | None:
    """Returns list of FarmerModel or None on 204/error/empty (domain path).

    Thin wrapper over _fetch_farmer_amulpashudhan_raw: validate raw dicts to
    FarmerModel; if a CACHED payload fails validation (e.g. an old schema), refetch
    fresh once (skip_cache) — preserving the previous cache-self-healing behavior.
    """
    raw = await _fetch_farmer_amulpashudhan_raw(mobile, token)
    if raw is None:
        return None
    try:
        return [
            FarmerModel.model_validate(data)
            for data in raw
        ]
    except Exception as e:
        logger.warning(
            "[AmulPashudhan(%s)] :: payload failed FarmerModel validation, refetching. error=%s",
            mobile,
            str(e),
        )
        raw = await _fetch_farmer_amulpashudhan_raw(mobile, token, skip_cache=True)
        if raw is None:
            return None
        try:
            return [
                FarmerModel.model_validate(data)
                for data in raw
            ]
        except Exception as e2:
            logger.warning(
                "[AmulPashudhan(%s)] :: refetch still failed FarmerModel validation: %s",
                mobile,
                str(e2),
            )
            return None


def _farmer_record_key(rec: dict) -> tuple:
    """Dedup key for raw farmer dicts: societyName + farmerCode."""
    return (str(rec.get("societyName") or ""), str(rec.get("farmerCode") or ""))


def merge_farmer_records(records: list[dict]) -> list[dict]:
    """Deduplicate raw farmer dicts by societyName+farmerCode; drop empties.
    Raw-dict analogue of merge_farmer_data (Option B). First occurrence wins."""
    seen: set = set()
    out: list[dict] = []
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
                        cached_data
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
        with start_observation(
            "fetch_animal_amulpashudhan",
            input={"tag_no": tag_no},
            metadata={"provider": "amulpashudhan", "url": url},
        ) as observation:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    url,
                    headers={
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=url)
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
            return AnimalModel.model_validate(data)
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
                        data
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
        async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
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
                data
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
                    cached_payload
                )
            except Exception as e:
                logger.warning(
                    "[Cache(%s)] :: Failed to validate cached cvcc payload, refetching. error=%s",
                    cache_key,
                    str(e),
                )

    try:
        async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
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
            response_json
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
    _ai_obs_input = {
        "union_code": request.union_code,
        "society_code": request.society_code,
        "farmer_code": request.farmer_code,
        "user_id": request.user_id,
        "species": request.species.value,
        "api_url": api_url,
    }

    try:
        with start_observation(
            "create_ai_call_api",
            as_type="generation",
            input=_ai_obs_input,
            metadata={"tool_backend": "amulpashudhan", "endpoint": "CreateAICall"},
        ) as ai_obs:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                # Trace the response (PII-safe) BEFORE raising, so failed bookings
                # (5xx) are captured in Langfuse, not just successes.
                _record_api_trace(ai_obs, response, provider="amulpashudhan", url=api_url)
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
            parsed = AICallResponseModel.model_validate(
                response_json
            )
            if ai_obs is not None:
                ai_obs.update(
                    output={
                        "success": True,
                        "status_code": response.status_code,
                        "ticket_number": parsed.ticket_number,
                        "ait_name": parsed.ait_name,
                    }
                )
            return parsed
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


async def create_health_call_api(
    request: HealthCallRequestModel, token: str
) -> HealthCallResponseModel | None:
    """Creates a health call and returns the ticket number details."""
    api_url = f"{BASE_AMULPASHUDHAN}/CreateHealthCall"
    _health_obs_input = {
        "union_code": request.union_code,
        "society_code": request.society_code,
        "farmer_code": request.farmer_code,
        "species": request.species.value,
        "case_type": request.case_type.value,
        "api_url": api_url,
    }

    try:
        with start_observation(
            "create_health_call_api",
            as_type="generation",
            input=_health_obs_input,
            metadata={"tool_backend": "amulpashudhan", "endpoint": "CreateHealthCall"},
        ) as health_obs:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                # Trace the response (PII-safe) BEFORE raising, so failed bookings
                # (5xx) are captured in Langfuse, not just successes.
                _record_api_trace(health_obs, response, provider="amulpashudhan", url=api_url)
                response.raise_for_status()
                logger.info(
                    "[CreateHealthCall(%s,%s,%s,%s,%s)] :: Response successfully recieved.",
                    request.union_code,
                    request.society_code,
                    request.farmer_code,
                    request.species.value,
                    request.case_type.value,
                )
        response_json = response.json()
        if not isinstance(response_json, dict):
            raise Exception("Not a valid dict provided in the response.")
        return HealthCallResponseModel.model_validate(
            response_json
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "[CreateHealthCall(%s,%s,%s,%s,%s)] :: Request failed with status code %s, and message = %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            request.case_type.value,
            e.response.status_code,
            e.response.text,
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            "[CreateHealthCall(%s,%s,%s,%s,%s)] :: Response didn't gave a valid json, failed due to decoding error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            request.case_type.value,
            str(e),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "[CreateHealthCall(%s,%s,%s,%s,%s)] :: Request failed, due to error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.species.value,
            request.case_type.value,
            str(e),
            exc_info=True,
        )

async def get_farmer_milk_collection_details_api(
    request: FarmerMilkCollectionRequestModel, token: str
) -> FarmerMilkCollectionResponseModel | None:
    """Fetches farmer milk collection and deduction details for a date range."""
    api_url = f"{BASE_AMULPASHUDHAN}/FarmerMilkCollectionDetails"

    try:
        with start_observation(
            "get_farmer_milk_collection_details_api",
            input=request.to_query_params(),
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=api_url)
                response.raise_for_status()
                logger.info(
                    "[FarmerMilkCollectionDetails(%s,%s,%s,%s,%s)] :: Response successfully recieved.",
                    request.union_code,
                    request.society_code,
                    request.farmer_code,
                    request.fromdate,
                    request.todate,
                )
        response_json = response.json()
        if not isinstance(response_json, dict):
            raise Exception("Not a valid dict provided in the response.")
        return FarmerMilkCollectionResponseModel.model_validate(
            response_json
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "[FarmerMilkCollectionDetails(%s,%s,%s,%s,%s)] :: Request failed with status code %s, and message = %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.fromdate,
            request.todate,
            e.response.status_code,
            e.response.text,
            exc_info=True,
        )
    except json.JSONDecodeError as e:
        logger.error(
            "[FarmerMilkCollectionDetails(%s,%s,%s,%s,%s)] :: Response didn't gave a valid json, failed due to decoding error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.fromdate,
            request.todate,
            str(e),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "[FarmerMilkCollectionDetails(%s,%s,%s,%s,%s)] :: Request failed, due to error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            request.fromdate,
            request.todate,
            str(e),
            exc_info=True,
        )


async def get_farmer_bonus_amount_api(
    request: FarmerBonusAmountRequestModel, token: str
) -> list[FarmerBonusAmountRecordModel] | None:
    """Fetches farmer bonus amount records (plain JSON array from GetFarmerBonusAmount).

    Returns an empty list when the API responds 200 with `[]`, or when the
    business body says farmer bonus data was not found. Returns None on
    unsupported-union / HTTP/parse/validation failure so callers can fan out
    across accounts.
    """
    api_url = f"{BASE_AMULPASHUDHAN}/GetFarmerBonusAmount"

    try:
        with start_observation(
            "get_farmer_bonus_amount_api",
            input=request.to_query_params(),
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=api_url)
                response.raise_for_status()
                logger.info(
                    "[GetFarmerBonusAmount(%s,%s,%s)] :: Response successfully received.",
                    request.union_code,
                    request.society_code,
                    request.farmer_code,
                )
        response_json = response.json()
        # Doc: success body is a plain JSON array — not an APIStatusCode envelope.
        if not isinstance(response_json, list):
            raise Exception("Not a valid list provided in the response.")
        return [
            FarmerBonusAmountRecordModel.model_validate(item)
            for item in response_json
        ]
    except httpx.HTTPStatusError as e:
        body = e.response.text or ""
        # Business messages from the API doc (validation / AMCS-only / not found).
        if "Bonus amount not supported" in body:
            logger.warning(
                "[GetFarmerBonusAmount(%s,%s,%s)] :: Union data source not supported "
                "(status=%s): %s",
                request.union_code,
                request.society_code,
                request.farmer_code,
                e.response.status_code,
                body,
            )
        elif "Farmer bonus data not found" in body:
            # No records for this account — treat as successful empty result so the
            # tool can show "no bonus records" instead of a temporary failure.
            logger.info(
                "[GetFarmerBonusAmount(%s,%s,%s)] :: No bonus data (status=%s): %s",
                request.union_code,
                request.society_code,
                request.farmer_code,
                e.response.status_code,
                body,
            )
            return []
        else:
            logger.error(
                "[GetFarmerBonusAmount(%s,%s,%s)] :: Request failed with status code %s, "
                "and message = %s",
                request.union_code,
                request.society_code,
                request.farmer_code,
                e.response.status_code,
                body,
                exc_info=True,
            )
    except json.JSONDecodeError as e:
        logger.error(
            "[GetFarmerBonusAmount(%s,%s,%s)] :: Response didn't give a valid json, "
            "failed due to decoding error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            str(e),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "[GetFarmerBonusAmount(%s,%s,%s)] :: Request failed, due to error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            str(e),
            exc_info=True,
        )

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


# ─────────────────────────────────────────────────────────────────────────────
# Voice-port (Inc 3.1) — AI-technician lookup + fetch tracing, canonical home.
# Reconciles chat's old agents/tools/get_ai_technicians_by_society.py into the
# backends so the farmer SWR cache (Inc 4) and farmer_context share ONE
# implementation: token-arg signature, graceful None-on-error (not raise),
# start_observation tracing, and a LENIENT camelCase record (Option B: cache
# stays camelCase). fetch_reason tags API calls so Langfuse can tell a cold/
# background refresh apart.
#
# Society-scoped technician list cache: one Redis entry per (union_code,
# society_code) pair; TTL from settings.ai_technician_cache_ttl_seconds.
# ─────────────────────────────────────────────────────────────────────────────

AI_TECHNICIAN_CACHE_NAMESPACE = "ai-technicians-by-society"
AI_TECHNICIAN_CACHE_TTL_SECONDS = settings.ai_technician_cache_ttl_seconds

_fetch_reason: ContextVar[str] = ContextVar("farmer_fetch_reason", default="request")


@contextmanager
def fetch_reason(reason: str):
    """Tag all Amul API calls made within this block with `reason`."""
    token = _fetch_reason.set(reason)
    try:
        yield
    finally:
        _fetch_reason.reset(token)


def current_fetch_reason() -> str:
    return _fetch_reason.get()


def _safe_response_summary(body: str) -> dict:
    """PII-safe shape of a response: record count + which keys are present/null,
    WITHOUT any values — enough to prove inconsistent returns without shipping PII."""
    out: dict[str, Any] = {"bytes": len(body)}
    if not body.strip():
        out["json"] = False
        return out
    try:
        data = json.loads(body)
    except Exception:
        out["json"] = False
        return out
    out["json"] = True
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if isinstance(data, list):
        out["records"] = len(data)
        first = data[0] if data and isinstance(data[0], dict) else None
    elif isinstance(data, dict):
        out["records"] = 1
        first = data
    else:
        out["records"] = 0
        first = None
    if isinstance(first, dict):
        out["keys"] = sorted(first.keys())
        out["null_keys"] = sorted(k for k, v in first.items() if v is None)
        array_lens = {k: len(v) for k, v in first.items() if isinstance(v, list)}
        if array_lens:
            out["array_lens"] = array_lens
        empty_str_keys = sorted(k for k, v in first.items() if v == "")
        if empty_str_keys:
            out["empty_str_keys"] = empty_str_keys
    return out


def _record_api_trace(observation, response, *, provider: str, url: str) -> None:
    """Attach status + a PII-safe response structure + source to a Langfuse
    observation. Raw bodies only when FARMER_API_TRACE_BODY is enabled. Tracing
    must never break a read."""
    if observation is None:
        return
    try:
        body = response.text or ""
    except Exception:
        body = ""
    try:
        output = {
            "status_code": response.status_code,
            "ok": 200 <= response.status_code < 300,
            "fetch_reason": _fetch_reason.get(),
            **_safe_response_summary(body),
        }
        if settings.farmer_api_trace_body and settings.farmer_api_trace_body_chars > 0:
            output["body"] = body[: settings.farmer_api_trace_body_chars]
        observation.update(output=output, metadata={"provider": provider, "url": url})
    except Exception:
        pass


class GetAITechniciansBySocietyQueryParams(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")

    def to_query_params(self) -> dict[str, str]:
        return {"unionCode": self.union_code, "societyCode": self.society_code}


class AITechnicianBySocietyRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    userId: Optional[str] = None
    fullName: Optional[str] = None
    mobileNumber: Optional[str] = None


def build_ai_technician_cache_key(union_code: str, society_code: str) -> str:
    """Normalized Redis key for a (union_code, society_code) technician list."""
    return f"{str(union_code).strip()}:{str(society_code).strip()}"


def _deserialize_ai_technicians(cached: Any) -> list[AITechnicianBySocietyRecord] | None:
    if not isinstance(cached, list):
        return None
    try:
        return [
            AITechnicianBySocietyRecord.model_validate(item)
            for item in cached
            if isinstance(item, dict)
        ]
    except Exception:
        return None


async def _get_cached_ai_technicians(
    union_code: str, society_code: str
) -> tuple[bool, list[AITechnicianBySocietyRecord] | None]:
    """Return (cache_hit, technicians). A miss is signaled by cache_hit=False."""
    key = build_ai_technician_cache_key(union_code, society_code)
    try:
        raw = await cache.get(key, namespace=AI_TECHNICIAN_CACHE_NAMESPACE)
    except Exception as e:
        logger.warning("AI technician cache read failed for %s: %s", key, e)
        return False, None
    if raw is None:
        return False, None
    technicians = _deserialize_ai_technicians(raw)
    if technicians is None:
        logger.warning("AI technician cache payload invalid for %s; treating as miss", key)
        return False, None
    logger.debug("AI technician cache hit for %s (%d records)", key, len(technicians))
    return True, technicians


async def _set_cached_ai_technicians(
    union_code: str,
    society_code: str,
    technicians: list[AITechnicianBySocietyRecord],
) -> None:
    key = build_ai_technician_cache_key(union_code, society_code)
    try:
        await cache.set(
            key,
            [technician.model_dump() for technician in technicians],
            ttl=AI_TECHNICIAN_CACHE_TTL_SECONDS,
            namespace=AI_TECHNICIAN_CACHE_NAMESPACE,
        )
        logger.debug("AI technician cache set for %s (%d records)", key, len(technicians))
    except Exception as e:
        logger.warning("AI technician cache write failed for %s: %s", key, e)


async def get_ai_technicians_by_society_cached(
    query: GetAITechniciansBySocietyQueryParams,
    token: str,
) -> list[AITechnicianBySocietyRecord] | None:
    """Cache-first AI technician lookup keyed by (union_code, society_code).

    Reads Redis first; calls ``get_ai_technicians_by_society_api`` only on miss.
    Successful responses (including an empty list) are cached for
    ``AI_TECHNICIAN_CACHE_TTL_SECONDS``. Failures (``None``) are not cached.
    """
    union_code = str(query.union_code).strip()
    society_code = str(query.society_code).strip()
    if not union_code or not society_code:
        return None

    hit, cached = await _get_cached_ai_technicians(union_code, society_code)
    if hit:
        return cached

    technicians = await get_ai_technicians_by_society_api(query, token)
    if technicians is not None:
        await _set_cached_ai_technicians(union_code, society_code, technicians)
    return technicians


async def get_ai_technicians_by_society_refresh(
    query: GetAITechniciansBySocietyQueryParams,
    token: str,
) -> list[AITechnicianBySocietyRecord] | None:
    """Bypass Redis read and refresh technician cache from upstream API.

    This is used by verification paths that must not trust a previously cached
    empty result. Successful API responses (including empty lists) replace cache.
    """
    union_code = str(query.union_code).strip()
    society_code = str(query.society_code).strip()
    if not union_code or not society_code:
        return None

    technicians = await get_ai_technicians_by_society_api(query, token)
    if technicians is not None:
        await _set_cached_ai_technicians(union_code, society_code, technicians)
    return technicians


async def get_ai_technicians_by_society_api(
    query: GetAITechniciansBySocietyQueryParams,
    token: str,
) -> list[AITechnicianBySocietyRecord] | None:
    """Fetch AI technicians mapped to a union and society.

    Returns None on any error (graceful — callers treat None as 'could not fetch'
    and an empty list as 'none found'), so a flaky lookup never breaks a read.
    """
    api_url = f"{BASE_AMULPASHUDHAN}/GetAITUserDetailsBySocietyCode"
    try:
        with start_observation(
            "get_ai_technicians_by_society_api",
            input=query.to_query_params(),
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(timeout=FARMER_BACKEND_HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    api_url,
                    params=query.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=api_url)
                response.raise_for_status()

        response_json = response.json()
        if isinstance(response_json, dict) and isinstance(response_json.get("data"), list):
            response_json = response_json["data"]
        if not isinstance(response_json, list):
            raise ValueError("Expected list response from GetAITechniciansBySociety")
        return [AITechnicianBySocietyRecord.model_validate(item) for item in response_json if isinstance(item, dict)]
    except httpx.HTTPStatusError as e:
        logger.error(
            "[GetAITechniciansBySociety(%s,%s)] :: HTTP %s: %s",
            query.union_code, query.society_code, e.response.status_code, e.response.text,
        )
    except Exception as e:
        logger.error(
            "[GetAITechniciansBySociety(%s,%s)] :: Error: %s",
            query.union_code, query.society_code, e,
        )
    return None
