"""
PashuGPT API tool for retrieving farmer details by mobile number.

This tool provides access to farmer registration data, animal counts, location information,
and productivity metrics from the Amul Pashudhan system.
"""
import os
from typing import List, Dict, Any, Optional
from pydantic_ai import ModelRetry
from helpers.utils import get_logger

logger = get_logger(__name__)

# API Configuration
PASHUGPT_BASE_URL = os.getenv(
    'PASHUGPT_BASE_URL',
    'https://api.amulpashudhan.com/configman/v1/PashuGPT'
)
PASHUGPT_TOKEN = os.getenv('PASHUGPT_TOKEN')


async def get_farmer_by_mobile(
    mobile_number: str,
) -> str:
    """
    Retrieve farmer details by mobile number from the PashuGPT API.
    
    Use this tool when the user asks about:
    - Their farmer profile or registration details
    - How many animals they have (cows, buffaloes, total)
    - Their location (state, district, village, society)
    - Their average milk production
    - Their farmer code or registration information
    
    This API does NOT require OTP authentication - it uses a bearer token.
    A farmer may have multiple registrations, so this returns an array.
    
    Args:
        mobile_number: The farmer's mobile number (10 digits, e.g., "9601335568")
        
    Returns:
        str: Formatted string containing farmer details including:
            - Farmer name and code
            - Location (state, district, sub-district, village)
            - Organization (union name, society name)
            - Animal counts (total, cows, buffaloes, milking animals)
            - Average milk production per day
            
    Example:
        User: "How many animals do I have?"
        Agent calls: get_farmer_by_mobile("9601335568")
        Returns: "Farmer: GEETABEN JASHWANATJI PARMAR (Code: 1165)
                 Location: FATEPUR, PALANPUR, BANASKANTHA, GUJARAT
                 Society: FATEPUR (VAD)M.P.C.S.LTD
                 Total Animals: 5 (3 Cows, 2 Buffaloes)
                 Milking Animals: 3
                 Avg Milk/Day: 12.5 liters"
    """
    if not PASHUGPT_TOKEN:
        error_msg = "PashuGPT API token not configured. Please set PASHUGPT_TOKEN environment variable."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not mobile_number or not mobile_number.strip():
        return "Error: Mobile number is required. Please provide a valid 10-digit mobile number."
    
    # Validate mobile number format (basic check)
    mobile_number = mobile_number.strip()
    if not mobile_number.isdigit() or len(mobile_number) != 10:
        return f"Error: Invalid mobile number format. Expected 10 digits, got: {mobile_number}"
    
    try:
        url = f"{PASHUGPT_BASE_URL}/GetFarmerDetailsByMobile?mobileNumber={mobile_number}"
        
        logger.info(f"Fetching farmer details for mobile: {mobile_number}")
        
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    'accept': 'application/json',
                    'Authorization': f'Bearer {PASHUGPT_TOKEN}',
                },
            )
            response.raise_for_status()
            data = response.json()
        
        # Handle array response (farmer may have multiple registrations)
        if not isinstance(data, list):
            data = [data] if data else []
        
        if len(data) == 0:
            return f"No farmer found with mobile number: {mobile_number}"
        
        # Format the response
        results = []
        for idx, farmer in enumerate(data):
            farmer_info = []
            
            # Basic Info
            farmer_info.append(f"**Farmer {idx + 1}:** {farmer.get('farmerName', 'N/A')}")
            farmer_info.append(f"**Farmer Code:** {farmer.get('farmerCode', 'N/A')}")
            farmer_info.append(f"**Mobile:** {farmer.get('mobileNumber', mobile_number)}")
            
            # Location
            location_parts = [
                farmer.get('village'),
                farmer.get('subDistrict'),
                farmer.get('district'),
                farmer.get('state'),
            ]
            location = ', '.join([p for p in location_parts if p])
            if location:
                farmer_info.append(f"**Location:** {location}")
            
            # Organization
            if farmer.get('societyName'):
                farmer_info.append(f"**Society:** {farmer.get('societyName')}")
            if farmer.get('unionName'):
                farmer_info.append(f"**Union:** {farmer.get('unionName')}")
            
            # Animal Counts
            total_animals = farmer.get('totalAnimals', 0)
            cows = farmer.get('cow', 0)
            buffaloes = farmer.get('buffalo', 0)
            milking = farmer.get('totalMilkingAnimals', 0)
            
            farmer_info.append(f"**Total Animals:** {total_animals}")
            if cows > 0 or buffaloes > 0:
                farmer_info.append(f"  - Cows: {cows}")
                farmer_info.append(f"  - Buffaloes: {buffaloes}")
            farmer_info.append(f"**Milking Animals:** {milking}")
            
            # Productivity
            avg_milk = farmer.get('avgMilkPerDayInLiter', 0.0)
            if avg_milk > 0:
                farmer_info.append(f"**Average Milk Production:** {avg_milk:.2f} liters/day")
            
            results.append('\n'.join(farmer_info))
        
        if len(results) == 1:
            return f"> Farmer Details\n\n{results[0]}"
        else:
            return f"> Farmer Details ({len(results)} registrations found)\n\n" + "\n\n---\n\n".join(results)
    
    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code} - {e.response.text}"
        logger.error(f"Error fetching farmer details: {error_msg}")
        if e.response.status_code == 404:
            return f"No farmer found with mobile number: {mobile_number}"
        elif e.response.status_code == 401:
            return "Error: Authentication failed. Please check PASHUGPT_TOKEN configuration."
        else:
            raise ModelRetry(f"Error fetching farmer details. Status: {e.response.status_code}")
    
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"Error fetching farmer details: {error_msg}")
        raise ModelRetry(f"Network error while fetching farmer details. Please try again.")
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Error fetching farmer details: {error_msg}")
        raise ModelRetry(f"Error fetching farmer details. Please try again.")
