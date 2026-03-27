"""
Tool for fetching farmer details by mobile number from PashuGPT-style APIs.
Uses amulpashudhan.com first, then herdman.live if needed (cohesive output, fallback on failure/empty).
"""
import json
import os

from agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    normalize_phone, merge_farmer_data,
)
from app.models.farmer import FarmerModel
from app.models.union import UnionName
from helpers.utils import get_logger, is_from_union

logger = get_logger(__name__)


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

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.error("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        return None

    records: list[FarmerModel] = []

    if token1:
        try:
            data = await fetch_farmer_amulpashudhan(mobile, token1)
            if data is not None:
                records.extend(data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from amulpashudhan")
        except Exception as e:
            logger.warning(f"amulpashudhan farmer API error for {mobile}: {e}")

    if token3 and is_from_union(records, UnionName.MEHSANA):
        try:
            data = await fetch_farmer_herdman(mobile, token3)
            if data:
                records.extend(data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from herdman")
        except Exception as e:
            logger.warning(f"herdman farmer API error for {mobile}: {e}")

    if len(records) == 0:
        logger.info(f"No farmer data found for mobile {mobile}")
        return None

    return merge_farmer_data(records)



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
    records = await get_farmer_data_by_mobile(mobile_number)
    if records is None:
        mobile = normalize_phone(mobile_number) or mobile_number
        return "Please provide a valid mobile number." if not mobile else f"Farmer details for mobile {mobile}:\n\nNo farmer data found for this mobile number."
    mobile = normalize_phone(mobile_number)
    formatted = json.dumps([record.model_dump() for record in records], indent=2, ensure_ascii=False)
    return f"Farmer details for mobile {mobile}:\n\n{formatted}"
