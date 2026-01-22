"""
PashuGPT API tool for retrieving animal details by tag number.

This tool provides comprehensive animal information including breed, reproductive status,
breeding history, lactation data, and health information from the Amul Pashudhan system.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic_ai import ModelRetry
from helpers.utils import get_logger

logger = get_logger(__name__)

# API Configuration
PASHUGPT_BASE_URL = os.getenv(
    'PASHUGPT_BASE_URL',
    'https://api.amulpashudhan.com/configman/v1/PashuGPT'
)
PASHUGPT_TOKEN = os.getenv('PASHUGPT_TOKEN')


def _calculate_age(date_of_birth: Optional[str]) -> Optional[str]:
    """Calculate age in years and months from date of birth."""
    if not date_of_birth:
        return None
    
    try:
        dob = datetime.fromisoformat(date_of_birth.replace('Z', '+00:00'))
        now = datetime.now(dob.tzinfo) if dob.tzinfo else datetime.now()
        delta = now - dob
        
        years = delta.days // 365
        months = (delta.days % 365) // 30
        
        if years > 0:
            return f"{years} year{'s' if years != 1 else ''} {months} month{'s' if months != 1 else ''}"
        else:
            return f"{months} month{'s' if months != 1 else ''}"
    except Exception:
        return None


def _format_date(date_str: Optional[str]) -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return "Not recorded"
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return date_str


def _days_since(date_str: Optional[str]) -> Optional[int]:
    """Calculate days since a given date."""
    if not date_str:
        return None
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        return delta.days
    except Exception:
        return None


async def get_animal_by_tag(
    tag_number: str,
) -> str:
    """
    Retrieve comprehensive animal details by Pashu Aadhaar tag number from the PashuGPT API.
    
    Use this tool when the user asks about:
    - A specific animal's details (breed, age, type)
    - Animal's reproductive status (pregnant, milking, dry)
    - Breeding history (last AI, pregnancy detection, calving)
    - Lactation information (current lactation number, milking stage)
    - Animal's health status or health history
    - When was the last artificial insemination (AI)
    - Pregnancy status of an animal
    - Calving history
    
    This API does NOT require OTP authentication - it uses a bearer token.
    The tag number is the unique Pashu Aadhaar identifier for the animal.
    
    Args:
        tag_number: The animal's Pashu Aadhaar tag number (e.g., "106290093933")
        
    Returns:
        str: Formatted string containing animal details including:
            - Tag number and basic info (type, breed, age)
            - Reproductive status (pregnancy stage, milking stage)
            - Lactation information
            - Breeding activity history (last AI, PD, calving dates)
            - Health activity information
            
    Example:
        User: "Tell me about my cow with tag 106290093933"
        Agent calls: get_animal_by_tag("106290093933")
        Returns: "Animal: Buffalo (Mehsana breed)
                 Age: 4 years 3 months
                 Status: Milking, Non Pregnant
                 Lactation: #3
                 Last AI: January 21, 2026 (5 days ago)
                 Last PD: Not recorded
                 Last Calving: Not recorded"
    """
    if not PASHUGPT_TOKEN:
        error_msg = "PashuGPT API token not configured. Please set PASHUGPT_TOKEN environment variable."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not tag_number or not tag_number.strip():
        return "Error: Tag number is required. Please provide a valid Pashu Aadhaar tag number."
    
    tag_number = tag_number.strip()
    
    try:
        url = f"{PASHUGPT_BASE_URL}/GetAnimalDetailsByTagNo?tagNo={tag_number}"
        
        logger.info(f"Fetching animal details for tag: {tag_number}")
        
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
            animal = response.json()
        
        if not animal:
            return f"No animal found with tag number: {tag_number}"
        
        # Format the response
        info = []
        
        # Basic Information
        info.append(f"**Tag Number:** {animal.get('tagNumber', tag_number)}")
        info.append(f"**Animal Type:** {animal.get('animalType', 'N/A')}")
        info.append(f"**Breed:** {animal.get('breed', 'N/A')}")
        
        # Age
        dob = animal.get('dateOfBirth')
        age = _calculate_age(dob)
        if age:
            info.append(f"**Age:** {age}")
        if dob:
            info.append(f"**Date of Birth:** {_format_date(dob)}")
        
        # Reproductive Status
        info.append(f"**Milking Stage:** {animal.get('milkingStage', 'N/A')}")
        info.append(f"**Pregnancy Stage:** {animal.get('pregnancyStage', 'N/A')}")
        
        # Lactation
        lactation_no = animal.get('lactationNo')
        if lactation_no is not None:
            info.append(f"**Current Lactation:** #{lactation_no}")
        
        # Breeding Activity
        breeding = animal.get('lastBreedingActivity', {})
        if breeding:
            info.append("\n**Breeding History:**")
            
            last_ai = breeding.get('lastAI')
            if last_ai:
                days = _days_since(last_ai)
                days_str = f" ({days} days ago)" if days is not None else ""
                info.append(f"  - Last AI: {_format_date(last_ai)}{days_str}")
            else:
                info.append(f"  - Last AI: Not recorded")
            
            last_pd = breeding.get('lastPD')
            if last_pd:
                days = _days_since(last_pd)
                days_str = f" ({days} days ago)" if days is not None else ""
                info.append(f"  - Last Pregnancy Detection: {_format_date(last_pd)}{days_str}")
            else:
                info.append(f"  - Last Pregnancy Detection: Not recorded")
            
            last_calving = breeding.get('lastCalving')
            if last_calving:
                days = _days_since(last_calving)
                days_str = f" ({days} days ago)" if days is not None else ""
                info.append(f"  - Last Calving: {_format_date(last_calving)}{days_str}")
            else:
                info.append(f"  - Last Calving: Not recorded")
            
            calf_tag = breeding.get('calfTagNo')
            if calf_tag:
                info.append(f"  - Calf Tag Number: {calf_tag}")
            
            calf_male = breeding.get('calfMale', 0)
            calf_female = breeding.get('calfFemale', 0)
            if calf_male > 0 or calf_female > 0:
                info.append(f"  - Calves: {calf_male} male, {calf_female} female")
        
        # Health Activity
        health = animal.get('lastHealthActivity')
        if health:
            info.append("\n**Health Information:** Available (details in system)")
        else:
            info.append("\n**Health Information:** Not recorded")
        
        return f"> Animal Details\n\n" + "\n".join(info)
    
    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code} - {e.response.text}"
        logger.error(f"Error fetching animal details: {error_msg}")
        if e.response.status_code == 404:
            return f"No animal found with tag number: {tag_number}"
        elif e.response.status_code == 401:
            return "Error: Authentication failed. Please check PASHUGPT_TOKEN configuration."
        else:
            raise ModelRetry(f"Error fetching animal details. Status: {e.response.status_code}")
    
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"Error fetching animal details: {error_msg}")
        raise ModelRetry(f"Network error while fetching animal details. Please try again.")
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Error fetching animal details: {error_msg}")
        raise ModelRetry(f"Error fetching animal details. Please try again.")
