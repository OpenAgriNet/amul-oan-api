"""
Tool for fetching CVCC health details by tag number from Amul Dairy API.
"""
import os
import json
import re
import httpx
from typing import Optional
from pydantic_ai import ModelRetry
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_cvcc_health_details(
    tag_no: str,
    token_no: Optional[str] = None,
    vendor_no: str = "9999999"
) -> str:
    """
    Fetch health-related information for an animal by tag number. This returns health-specific 
    details including treatments, vaccinations, deworming records, milk yield, farmer information, 
    and other health metrics. Use this tool when users ask about animal health, treatments, 
    vaccinations, or medical history.
    
    Args:
        tag_no: The tag number of the animal to fetch health details for (required)
        token_no: Token number for CVCC API authentication (optional, defaults to PASHUGPT_TOKEN_2 env var)
        vendor_no: Vendor number for CVCC API (default: 9999999)
        
    Returns:
        str: Formatted JSON string with health details including Tag, Animal Type, Breed, 
             Milking Stage, Pregnancy Stage, Lactation, Milk Yield, Farmer information, 
             Treatment records, Vaccination records, and Deworming records
    """
    try:
        # Get token_no from parameter or environment variable
        if not token_no:
            token_no = os.getenv('PASHUGPT_TOKEN_2')
            if not token_no:
                raise ValueError("PASHUGPT_TOKEN_2 environment variable is not set and token_no not provided")
        
        api_url = "https://api.amuldairy.com/ai_cattle_dtl.php"
        
        logger.info(f"Fetching CVCC health details for tag: {tag_no}")
        
        payload = {
            "token_no": token_no,
            "vendor_no": vendor_no,
            "tag_no": tag_no
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    'Content-Type': 'application/json',
                },
                json=payload
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"CVCC API error for tag {tag_no}: {response.status_code} - {error_text}")
                raise ModelRetry(f"Failed to fetch CVCC health details: {response.status_code}")
            
            # The API may return malformed JSON with trailing commas, so we need to fix it
            text = response.text
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Try to fix trailing comma issue using Python regex
                fixed_text = re.sub(r',\s*}', '}', text)
                fixed_text = re.sub(r',\s*]', ']', fixed_text)
                data = json.loads(fixed_text)
        
        # Check if the response indicates success
        if data.get('msg') != 'Success':
            error_msg = data.get('msg', 'Unknown error')
            logger.warning(f"CVCC API returned non-success message for tag {tag_no}: {error_msg}")
            return f"CVCC Health Details for Tag {tag_no}:\n\nNo health data available. Message: {error_msg}"
        
        # Format the response as a readable string
        formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
        return f"CVCC Health Details for Tag {tag_no}:\n\n{formatted_data}"
        
    except Exception as e:
        logger.error(f"Error fetching CVCC health details for tag {tag_no}: {e}")
        raise ModelRetry(f"Error fetching CVCC health details, please try again: {str(e)}")
