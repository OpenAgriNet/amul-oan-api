"""
Tool for fetching animal details by tag number from PashuGPT API.
"""
import os
import json
import httpx
from pydantic_ai import ModelRetry
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_animal_by_tag(
    tag_no: str
) -> str:
    """
    Fetch general animal information by tag number. This returns basic animal details 
    including breed, milking stage, pregnancy stage, lactation number, date of birth, 
    and last breeding/health activities.
    
    Args:
        tag_no: The tag number of the animal to fetch details for (required)
        
    Returns:
        str: Formatted JSON string with animal details including tagNumber, animalType, 
             breed, milkingStage, pregnancyStage, dateOfBirth, lactationNo, 
             lastBreedingActivity, and lastHealthActivity
    """
    try:
        pashugpt_token = os.getenv('PASHUGPT_TOKEN')
        if not pashugpt_token:
            raise ValueError("PASHUGPT_TOKEN environment variable is not set")
        
        api_url = f"https://api.amulpashudhan.com/configman/v1/PashuGPT/GetAnimalDetailsByTagNo?tagNo={tag_no}"
        
        logger.info(f"Fetching animal details for tag: {tag_no}")
        
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
                logger.info(f"No animal data found for tag {tag_no} (204 No Content)")
                return f"Animal Details for Tag {tag_no}:\n\nNo animal data found for this tag number."
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"API error for tag {tag_no}: {response.status_code} - {error_text}")
                raise ModelRetry(f"Failed to fetch animal details: {response.status_code}")
            
            # Handle empty response body
            if not response.text or response.text.strip() == '':
                logger.info(f"Empty response for tag {tag_no}")
                return f"Animal Details for Tag {tag_no}:\n\nNo animal data found for this tag number."
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for tag {tag_no}: {e}. Response text: {response.text[:500]}")
                raise ModelRetry(f"Failed to parse animal details response")
        
        # Format the response as a readable string
        formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
        return f"Animal Details for Tag {tag_no}:\n\n{formatted_data}"
        
    except Exception as e:
        logger.error(f"Error fetching animal details for tag {tag_no}: {e}")
        raise ModelRetry(f"Error fetching animal details, please try again: {str(e)}")
