"""
Tool for fetching animal details by tag number from PashuGPT-style APIs.
Uses amulpashudhan.com and returns normalized snake_case animal data.
"""
from app.models.animal import AnimalModel
import json
import os
from typing import Optional

from helpers.utils import get_logger

from agents.tools.farmer_animal_backends import fetch_animal_amulpashudhan

# Herdman fallback is temporarily disabled. Keep the previous imports commented
# out so the old path is easy to restore.
# from agents.tools.farmer_animal_backends import (
#     fetch_animal_herdman,
#     merge_animal_data,
# )

logger = get_logger(__name__)


async def get_animal_data_by_tag(tag: str) -> AnimalModel | None:
    """
    Fetch structured animal data by tag number.

    Args:
        tag: The tag number of the animal.

    Returns:
        A normalized animal record dict, or None if no data is found.
    """
    if not tag:
        return None

    token1 = os.getenv("PASHUGPT_TOKEN")
    if not token1:
        logger.error("PASHUGPT_TOKEN is not set")
        return None

    try:
        return await fetch_animal_amulpashudhan(tag, token1)
    except Exception as e:
        logger.warning(f"amulpashudhan animal API error for tag {tag}: {e}")

    return None


async def get_animal_by_tag(tag: str, society_name: Optional[str] = None) -> str:
    """
    Fetch animal information by tag number. Returns details including breed,
    milking stage, pregnancy stage, lactation, date of birth, and last
    breeding/health activities from amulpashudhan.

    Args:
        tag: The tag number of the animal (required).
        society_name: Reserved for compatibility. Currently unused.

    Returns:
        str: Formatted JSON string with animal details, or a clear message if no data found.
             Handles API failures, 204 No Content, and empty responses.
    """
    if not tag:
        return "Please provide a valid tag number."

    _ = society_name
    animal = await get_animal_data_by_tag(tag)

    # Herdman fallback is temporarily disabled. Keep the old flow commented out
    # instead of removing it completely.
    # token3 = os.getenv("PASHUGPT_TOKEN_3")
    # fallback: Optional[Dict[str, Any]] = None
    # if token3 and society_name == "Mehsana":
    #     try:
    #         fallback = await fetch_animal_herdman(tag, token3)
    #         if fallback:
    #             logger.info(f"Animal data for tag {tag}: got from herdman")
    #     except Exception as e:
    #         logger.warning(f"herdman animal API error for tag {tag}: {e}")
    # merged = merge_animal_data(animal, fallback)

    if not animal:
        logger.info(f"No animal data found for tag {tag}")
        return f"Animal details for tag {tag}:\n\nNo animal data found for this tag number."

    formatted = json.dumps(animal, indent=2, ensure_ascii=False)
    return f"Animal details for tag {tag}:\n\n{formatted}"
