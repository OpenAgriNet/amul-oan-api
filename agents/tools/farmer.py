"""
Tool for fetching farmer details by mobile number from PashuGPT-style APIs.
Uses amulpashudhan.com and returns a cohesive, deduplicated set of records.
"""
import json
import re
import uuid
from enum import Enum

from agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    _fetch_farmer_amulpashudhan_raw,
    normalize_phone, merge_farmer_data, merge_farmer_records,
)
from app.models.farmer import FarmerModel
from app.models.farmer_transport import FarmerRecord
from helpers.utils import get_logger
from app.config import get_config_value

logger = get_logger(__name__)


class FarmerFetchOutcome(str, Enum):
    """Fetch outcome for the SWR cache ingestion path."""

    FOUND = "found"
    ERROR = "error"


def normalize_phone_to_mobile(user_id: str) -> str | None:
    """
    Clean user_id as a phone number: strip non-digits, take the last 10 digits.
    Returns None when user_id isn't a usable phone (anon, UUID, name, <10 digits) —
    used by the voice path to derive the caller's mobile from the request user_id.
    """
    if not user_id or not str(user_id).strip():
        return None
    s = str(user_id).strip()
    # Skip anon / anonymous (case-insensitive)
    if s.lower() in ("anon", "anonymous"):
        return None
    # Skip if it's a valid UUID
    try:
        uuid.UUID(s)
        return None
    except (ValueError, AttributeError, TypeError):
        pass
    # Skip if fewer than 10 digits
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return None
    return digits[-10:]


async def get_farmer_data_by_mobile(mobile_number: str) -> list[FarmerModel] | None:
    """
    Fetch farmer records by mobile number (same backends as get_farmer_by_mobile).
    Returns structured list of farmer records for use by chat/service layer.

    Args:
        mobile_number: The mobile number of the farmer. Can include +91 or spaces.

    Returns:
        List of farmer record dicts, or None if invalid mobile, no tokens, or no data.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return None

    token1 = get_config_value("PASHUGPT_TOKEN")
    if not token1:
        logger.error("PASHUGPT_TOKEN is not set")
        return None

    records: list[FarmerModel] = []

    try:
        data = await fetch_farmer_amulpashudhan(mobile, token1)
        if data is not None:
            records.extend(data)
            logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from amulpashudhan")
    except Exception as e:
        logger.warning(f"amulpashudhan farmer API error for {mobile}: {e}")

    if len(records) == 0:
        logger.info(f"No farmer data found for mobile {mobile}")
        return None

    return merge_farmer_data(records)


def _record_has_content(rec: dict) -> bool:
    """A farmer row is worth keeping if it carries animal tags, a non-zero
    animal count, or at least an identity (farmer/society name). Mirrors voice's
    has_content gate so empty placeholder rows don't pollute the SWR cache."""
    if rec.get("tagNo") or rec.get("tagNumbers"):
        return True
    total = rec.get("totalAnimals")
    if total not in (None, 0, "0"):
        return True
    return bool(rec.get("farmerName") or rec.get("societyName"))


async def fetch_farmer_info_with_outcome(
    mobile_number: str,
) -> tuple[list[FarmerRecord] | None, FarmerFetchOutcome]:
    """Raw farmer fetch with an explicit provider outcome for the SWR cache.

    Returns RAW camelCase ``FarmerRecord`` objects when ``outcome`` is ``FOUND``.
    Empty payloads are treated as non-authoritative and map to ``ERROR`` until
    upstream exposes a reliable explicit miss/error split.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return None, FarmerFetchOutcome.ERROR

    token1 = get_config_value("PASHUGPT_TOKEN")
    if not token1:
        logger.error("PASHUGPT_TOKEN is not set")
        return None, FarmerFetchOutcome.ERROR

    rows: list[dict] = []
    raw = await _fetch_farmer_amulpashudhan_raw(mobile, token1)
    if raw is not None:
        if len(raw) > 0:
            rows.extend(r for r in raw if isinstance(r, dict))

    if not rows:
        return None, FarmerFetchOutcome.ERROR

    kept = [r for r in rows if _record_has_content(r)] or rows
    deduped = merge_farmer_records(kept)
    if not deduped:
        return None, FarmerFetchOutcome.ERROR

    records = [FarmerRecord.model_validate(r) for r in deduped]
    logger.info(f"Raw farmer info for {mobile}: {len(records)} record(s) merged")
    return records, FarmerFetchOutcome.FOUND


async def fetch_farmer_info_raw(mobile_number: str) -> list[FarmerRecord] | None:
    """Backward-compatible wrapper over ``fetch_farmer_info_with_outcome``.

    Returns records only on ``FOUND``; otherwise ``None``.
    """
    records, outcome = await fetch_farmer_info_with_outcome(mobile_number)
    if outcome == FarmerFetchOutcome.FOUND:
        return records
    return None


async def get_farmer_by_mobile(mobile_number: str) -> str:
    """
    Fetch farmer information by mobile number. Returns farmer details including
    farmer ID, name, location, society, and associated animal tag numbers.

    Args:
        mobile_number: The mobile number of the farmer (required). Can include +91 or spaces.

    Returns:
        str: Formatted JSON string with farmer details and associated tag numbers,
             or a clear message if no data found. Handles API failures and empty responses.
    """
    records = await get_farmer_data_by_mobile(mobile_number)
    if records is None:
        mobile = normalize_phone(mobile_number) or mobile_number
        return "Please provide a valid mobile number." if not mobile else f"Farmer details for mobile {mobile}:\n\nNo farmer data found for this mobile number."
    mobile = normalize_phone(mobile_number)
    formatted = json.dumps([record.model_dump() for record in records], indent=2, ensure_ascii=False)
    return f"Farmer details for mobile {mobile}:\n\n{formatted}"
