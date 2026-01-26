"""
Tool for fetching farmer details by mobile number from PashuGPT API.
"""
import os
import json
import httpx
from pydantic_ai import ModelRetry
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_farmer_by_mobile(
    mobile_number: str
) -> str:
    """
    Fetch farmer information by mobile number. This returns farmer details including 
    farmer ID, name, location, and associated animal tag numbers.
    
    Args:
        mobile_number: The mobile number of the farmer to fetch details for (required)
        
    Returns:
        str: Formatted JSON string with farmer details including farmer information 
             and associated tag numbers
    """
    try:
        pashugpt_token = os.getenv('PASHUGPT_TOKEN')
        if not pashugpt_token:
            raise ValueError("PASHUGPT_TOKEN environment variable is not set")
        
        api_url = f"https://api.amulpashudhan.com/configman/v1/PashuGPT/GetFarmerDetailsByMobile?mobileNumber={mobile_number}"
        
        logger.info(f"Fetching farmer details for mobile number: {mobile_number}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                api_url,
                headers={
                    'accept': 'application/json',
                    'Authorization': f'Bearer {pashugpt_token}',
                }
            )
            
            # Handle different status codes
            if response.status_code == 204:
                # 204 No Content - valid response meaning no data found
                logger.info(f"No farmer data found for mobile number {mobile_number} (204 No Content)")
                return f"Farmer Details for Mobile Number {mobile_number}:\n\nNo farmer data found for this mobile number."
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"API error for mobile number {mobile_number}: {response.status_code} - {error_text}")
                raise ModelRetry(f"Failed to fetch farmer details: {response.status_code}")
            
            # Handle empty response body
            if not response.text or response.text.strip() == '':
                logger.info(f"Empty response for mobile number {mobile_number}")
                return f"Farmer Details for Mobile Number {mobile_number}:\n\nNo farmer data found for this mobile number."
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for mobile number {mobile_number}: {e}. Response text: {response.text[:500]}")
                raise ModelRetry(f"Failed to parse farmer details response")
        
        # Format the response as a readable string
        formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
        return f"Farmer Details for Mobile Number {mobile_number}:\n\n{formatted_data}"
        
    except Exception as e:
        logger.error(f"Error fetching farmer details for mobile number {mobile_number}: {e}")
        raise ModelRetry(f"Error fetching farmer details, please try again: {str(e)}")
