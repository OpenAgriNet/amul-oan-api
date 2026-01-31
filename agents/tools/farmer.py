"""
Tool for fetching farmer details by mobile number from PashuGPT-style APIs.
Uses amulpashudhan.com first, then herdman.live if needed (cohesive output, fallback on failure/empty).
"""
import json
import os
from typing import Any, Dict, List

from agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    merge_farmer_records,
    normalize_phone,
)
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_farmer_by_mobile(mobile_number: str) -> str:
    """
    Fetch farmer information by mobile number. Returns farmer details including
    farmer ID, name, location, society, and associated animal tag numbers.
    Tries multiple backends and merges results when both return data.

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

    records: List[Dict[str, Any]] = []

    # Try amulpashudhan first
    if token1:
        try:
            data = await fetch_farmer_amulpashudhan(mobile, token1)
            if data:
                records = merge_farmer_records(records + data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from amulpashudhan")
        except Exception as e:
            logger.warning(f"amulpashudhan farmer API error for {mobile}: {e}")

    # If no data yet, or optionally merge, try herdman
    if token3:
        try:
            data = await fetch_farmer_herdman(mobile, token3)
            if data:
                records = merge_farmer_records(records + data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from herdman")
        except Exception as e:
            logger.warning(f"herdman farmer API error for {mobile}: {e}")

    if not records:
        logger.info(f"No farmer data found for mobile {mobile}")
        return f"Farmer details for mobile {mobile}:\n\nNo farmer data found for this mobile number."

    # Optional: filter out rows that are effectively empty (e.g. totalAnimals 0 and no tagNo)
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
