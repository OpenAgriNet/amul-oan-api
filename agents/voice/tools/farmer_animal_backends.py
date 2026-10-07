"""
Voice's one direct call to amulpashudhan.com (PASHUGPT_TOKEN): GetFarmerBonusAmount,
which has no Beckn action yet. Bookings and milk go through chat's Beckn functions,
and farmer, animal and technician data come from the shared farmer cache
(agents/tools/farmer_cache.py).
"""
import json
import re
from typing import Any

import httpx

from app.voice.models.bonus import (
    FarmerBonusAmountRecordModel,
    FarmerBonusAmountRequestModel,
)
from app.config import settings
from app.observability import start_observation
from helpers.utils import get_logger

_logger = get_logger(__name__)

BASE_AMULPASHUDHAN = "https://api.amulpashudhan.com/configman/v1/PashuGPT"


def _safe_response_summary(body: str) -> dict:
    """PII-safe shape of a response: record count + which keys are present/null,
    WITHOUT any values. This is what proves an inconsistent return (e.g. a record
    that came back missing `totalAnimals`) without shipping farmer PII to Langfuse.
    """
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
        # Lengths of list-valued fields (no values) — surfaces empty/thin records
        # (e.g. an empty animals/visits/technicians array) which signal a
        # degraded "empty" response distinct from a populated one.
        array_lens = {k: len(v) for k, v in first.items() if isinstance(v, list)}
        if array_lens:
            out["array_lens"] = array_lens
        # Keys whose value is an empty string — another empty-response signal.
        empty_str_keys = sorted(k for k, v in first.items() if v == "")
        if empty_str_keys:
            out["empty_str_keys"] = empty_str_keys
    return out


# Upstream error bodies are short ("Farmer Record Not Found." is 36 bytes); this
# is a safety bound, not a real limit.
_ERROR_BODY_TRACE_CHARS = 500
_LONG_DIGIT_RUN = re.compile(r"\d{10,}")


def _redact_error_body(body: str) -> str:
    """An upstream error body, with anything phone-shaped removed.

    Error responses can echo the request, and the request carries the caller's
    mobile. _safe_response_summary exists precisely so we never ship farmer PII
    to Langfuse, and recording error text must not become the exception.
    """
    return _LONG_DIGIT_RUN.sub("[redacted]", body[:_ERROR_BODY_TRACE_CHARS])


def _record_api_trace(observation, response, *, provider: str, url: str) -> None:
    """Attach status + a PII-safe response structure + source to a Langfuse
    observation. This is how we prove inconsistent upstream returns. Span latency
    is recorded by Langfuse from the observation duration. Raw bodies are only
    included when FARMER_API_TRACE_BODY is enabled (deep-debug), except for error
    responses, whose (redacted) body is always kept — {status_code, bytes, keys}
    alone cannot distinguish the partner's "Farmer Record Not Found." 500 from a
    genuine fault, which is the question this trace exists to answer (#282 P2).
    Wrapped in try/except: tracing must never break a read.
    """
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
            **_safe_response_summary(body),
        }
        if settings.farmer_api_trace_body and settings.farmer_api_trace_body_chars > 0:
            output["body"] = body[: settings.farmer_api_trace_body_chars]
        elif not output["ok"] and body.strip():
            output["error_body"] = _redact_error_body(body)
        observation.update(output=output, metadata={"provider": provider, "url": url})
    except Exception:
        pass


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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                _record_api_trace(observation, response, provider="amulpashudhan", url=api_url)
                response.raise_for_status()
                _logger.info(
                    "[GetFarmerBonusAmount(%s,%s,%s)] :: Response successfully received.",
                    request.union_code,
                    request.society_code,
                    request.farmer_code,
                )
        response_json = response.json()
        # Doc: success body is a plain JSON array — not an APIStatusCode envelope.
        if not isinstance(response_json, list):
            raise ValueError("Expected list response from GetFarmerBonusAmount")
        return [
            FarmerBonusAmountRecordModel.model_validate(item)
            for item in response_json
        ]
    except httpx.HTTPStatusError as e:
        body = e.response.text or ""
        # Business messages from the API doc (validation / AMCS-only / not found).
        if "Bonus amount not supported" in body:
            _logger.warning(
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
            _logger.info(
                "[GetFarmerBonusAmount(%s,%s,%s)] :: No bonus data (status=%s): %s",
                request.union_code,
                request.society_code,
                request.farmer_code,
                e.response.status_code,
                body,
            )
            return []
        else:
            _logger.error(
                "[GetFarmerBonusAmount(%s,%s,%s)] :: Request failed with status code %s, "
                "and message = %s",
                request.union_code,
                request.society_code,
                request.farmer_code,
                e.response.status_code,
                body,
            )
    except json.JSONDecodeError as e:
        _logger.error(
            "[GetFarmerBonusAmount(%s,%s,%s)] :: Response didn't give a valid json, "
            "failed due to decoding error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            str(e),
        )
    except Exception as e:
        _logger.error(
            "[GetFarmerBonusAmount(%s,%s,%s)] :: Request failed, due to error %s",
            request.union_code,
            request.society_code,
            request.farmer_code,
            str(e),
        )
    return None
