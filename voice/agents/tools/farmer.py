"""
Tool for fetching farmer details by mobile number from PashuGPT-style APIs.
Uses amulpashudhan.com first, then herdman.live if needed (cohesive output, fallback on failure/empty).
"""
import json
import os
import re
import uuid
from typing import Optional, Dict, Any, List

from voice.helpers.utils import get_logger

from voice.agents.models.farmer import FarmerRecord
from voice.agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    merge_farmer_records,
    normalize_phone,
)

logger = get_logger(__name__)


def normalize_phone_to_mobile(user_id: str) -> Optional[str]:
    """
    Clean user_id as phone number: remove spaces, special chars, take last 10 digits.
    Skip if user_id is not a number with at least 10 digits (e.g. uuid, names, anon).

    Args:
        user_id: Raw user identifier (expected to be phone number)

    Returns:
        Last 10 digits as string, or None if skip (uuid, John Doe, anon, <10 digits)
    """
    if not user_id or not str(user_id).strip():
        return None
    s = str(user_id).strip()
    # Skip anon, anonymous (case insensitive)
    if s.lower() in ('anon', 'anonymous'):
        return None
    # Skip if valid UUID
    try:
        uuid.UUID(s)
        return None
    except (ValueError, AttributeError, TypeError):
        pass
    # Skip if fewer than 10 digits
    digits = re.sub(r'\D', '', s)
    if len(digits) < 10:
        return None
    return digits[-10:]


async def _fetch_farmer_records_dual_backend(mobile: str) -> List[Dict[str, Any]]:
    """Fetch farmer records from both backends, merge and deduplicate."""
    records: List[Dict[str, Any]] = []
    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")

    if token1:
        try:
            data = await fetch_farmer_amulpashudhan(mobile, token1)
            if data:
                records = merge_farmer_records(records + data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from amulpashudhan")
        except Exception as e:
            logger.warning(f"amulpashudhan farmer API error for {mobile}: {e}")

    if token3:
        try:
            data = await fetch_farmer_herdman(mobile, token3)
            if data:
                records = merge_farmer_records(records + data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from herdman")
        except Exception as e:
            logger.warning(f"herdman farmer API error for {mobile}: {e}")

    return records


async def fetch_farmer_info_raw(mobile_number: str) -> Optional[List[FarmerRecord]]:
    """
    Fetch farmer details by mobile number and return raw data for programmatic use.
    Tries both backends (amulpashudhan + herdman), merges and deduplicates.

    Args:
        mobile_number: The mobile number (10 digits) to fetch details for

    Returns:
        Raw farmer data (list of records), or None if not found or on error
    """
    mobile = normalize_phone(str(mobile_number))
    if not mobile or len(mobile) < 10:
        return None

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.warning("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        return None

    records = await _fetch_farmer_records_dual_backend(mobile)
    if not records:
        return None

    # Filter empty records optionally
    def has_content(rec: Dict[str, Any]) -> bool:
        tag_no = rec.get("tagNo") or rec.get("tagNumbers")
        total = rec.get("totalAnimals")
        if tag_no or (total is not None and total != 0):
            return True
        return bool(rec.get("farmerName") or rec.get("societyName"))

    filtered = [r for r in records if has_content(r)]
    return [FarmerRecord.model_validate(r) for r in (filtered if filtered else records)]


async def get_farmer_by_mobile(mobile_number: str) -> str:
    """
    Fetch farmer information by mobile number. Returns farmer details including
    farmer ID, name, location, society, and associated animal tag numbers.
    Tries multiple backends (amulpashudhan, herdman) and merges results when both return data.

    Args:
        mobile_number: The mobile number of the farmer (required). Can include +91 or spaces.

    Returns:
        str: Formatted JSON string with farmer details and associated tag numbers,
             or a clear message if no data found. Handles API failures and empty responses.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return "Please provide a valid mobile number."

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.error("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        raise ValueError("PASHUGPT_TOKEN or PASHUGPT_TOKEN_3 environment variable must be set")

    records = await _fetch_farmer_records_dual_backend(mobile)
    if not records:
        logger.info(f"No farmer data found for mobile {mobile}")
        return f"Farmer details for mobile {mobile}:\n\nNo farmer data found for this mobile number."

    def has_content(rec: Dict[str, Any]) -> bool:
        tag_no = rec.get("tagNo") or rec.get("tagNumbers")
        total = rec.get("totalAnimals")
        if tag_no or (total is not None and total != 0):
            return True
        return bool(rec.get("farmerName") or rec.get("societyName"))

    filtered = [r for r in records if has_content(r)]
    if not filtered:
        filtered = records

    formatted = json.dumps(filtered, indent=2, ensure_ascii=False)
    return f"Farmer details for mobile {mobile}:\n\n{formatted}"
