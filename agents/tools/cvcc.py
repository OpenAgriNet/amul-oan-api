"""
Tool for fetching CVCC health details by tag number from Amul Dairy API.
"""
import os
from typing import Optional
from pydantic_ai import ModelRetry
import json

from agents.tools.farmer_animal_backends import fetch_cvcc_health_details
from app.models.union import UnionName
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_cvcc_health_data_by_tag(
    tag_no: str,
    token_no: Optional[str] = None,
    vendor_no: str = "9999999",
    union_name: Optional[str] = None,
):
    if not tag_no:
        return None

    if not token_no:
        token_no = os.getenv("PASHUGPT_TOKEN_2")
        if not token_no:
            logger.error("PASHUGPT_TOKEN_2 is not set")
            return None

    normalized_union_name = (union_name or "").strip().lower()
    if normalized_union_name != UnionName.SABARKAIRA.value:
        return None

    try:
        return await fetch_cvcc_health_details(tag_no, token_no, vendor_no)
    except Exception as e:
        logger.warning(f"cvcc API error for tag {tag_no}: {e}")
        return None


async def get_cvcc_health_details(
    tag_no: str,
    token_no: Optional[str] = None,
    vendor_no: str = "9999999",
    union_name: Optional[str] = None
) -> str:
    """
    Fetch health-related information for an animal by tag number. This returns health-specific 
    details including treatments, vaccinations, deworming records, milk yield, farmer information, 
    and other health metrics. Use this tool when users ask about animal health, treatments, 
    vaccinations, or medical history. This tool is only available if the farmer belongs to SabarKaira union.
    
    Args:
        tag_no: The tag number of the animal to fetch health details for (required)
        token_no: Token number for CVCC API authentication (optional, defaults to PASHUGPT_TOKEN_2 env var)
        vendor_no: Vendor number for CVCC API (default: 9999999)
        union_name: The name of the union, the farmer belongs to (required, only if it is available)
        
    Returns:
        str: Raw text response from the CVCC API containing health details including Tag, 
             Animal Type, Breed, Milking Stage, Pregnancy Stage, Lactation, Milk Yield, 
             Farmer information, Treatment records, Vaccination records, and Deworming records.
             The response may be in JSON format (possibly malformed) or plain text.
    """
    try:
        normalized_union_name = (union_name or "").strip().lower()
        if normalized_union_name != UnionName.SABARKAIRA.value:
            return "The farmer doesn't belong to \"SabarKaira\" union."

        data = await get_cvcc_health_data_by_tag(
            tag_no=tag_no,
            token_no=token_no,
            vendor_no=vendor_no,
            union_name=normalized_union_name,
        )
        if data is None:
            return (
                f"CVCC Health Details for Tag {tag_no}:\n\n"
                "No CVCC health data found for this tag number."
            )

        formatted = json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
        return f"CVCC Health Details for Tag {tag_no}:\n\n{formatted}"

    except Exception as e:
        logger.error(f"Error fetching CVCC health details for tag {tag_no}: {e}")
        raise ModelRetry(f"Error fetching CVCC health details, please try again: {str(e)}")
