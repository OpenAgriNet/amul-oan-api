"""
Tool for fetching animal details by tag number from PashuGPT-style APIs.
Uses amulpashudhan.com first, then herdman.live if needed (cohesive output, fallback on failure/empty).
"""
import json
import os
from typing import Any, Dict, Optional

from helpers.utils import get_logger

from agents.tools.farmer_animal_backends import (
    fetch_animal_amulpashudhan,
    fetch_animal_herdman,
    merge_animal_data,
    normalize_tag,
)

logger = get_logger(__name__)


async def get_animal_by_tag(tag_no: str) -> str:
    """
    Fetch animal information by tag number. Returns details including breed,
    milking stage, pregnancy stage, lactation, date of birth, and last
    breeding/health activities. Tries multiple backends and merges when both return data.

    Args:
        tag_no: The tag number of the animal (required).

    Returns:
        str: Formatted JSON string with animal details, or a clear message if no data found.
             Handles API failures, 204 No Content, and empty responses.
    """
    tag = normalize_tag(tag_no)
    if not tag:
        return "Please provide a valid tag number."

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.error("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        raise ValueError("PASHUGPT_TOKEN or PASHUGPT_TOKEN_3 environment variable must be set")

    primary: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = None

    if token1:
        try:
            primary = await fetch_animal_amulpashudhan(tag, token1)
            if primary:
                logger.info(f"Animal data for tag {tag}: got from amulpashudhan")
        except Exception as e:
            logger.warning(f"amulpashudhan animal API error for tag {tag}: {e}")

    if token3:
        try:
            fallback = await fetch_animal_herdman(tag, token3)
            if fallback:
                logger.info(f"Animal data for tag {tag}: got from herdman")
        except Exception as e:
            logger.warning(f"herdman animal API error for tag {tag}: {e}")

    merged = merge_animal_data(primary, fallback)
    if not merged:
        logger.info(f"No animal data found for tag {tag}")
        return f"Animal details for tag {tag}:\n\nNo animal data found for this tag number."

    formatted = json.dumps(merged, indent=2, ensure_ascii=False)
    return f"Animal details for tag {tag}:\n\n{formatted}"
