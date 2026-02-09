"""
Tasks for sending telemetry data.
Uses httpx for non-blocking HTTP so the event loop is not blocked.
"""
import os
from dotenv import load_dotenv
from typing import Dict
import httpx
from helpers.utils import get_logger

load_dotenv()

logger = get_logger(__name__)

# TODO: Make sure env has correct value
TELEMETRY_API_URL = os.getenv("TELEMETRY_API_URL", "https://vistaar.kenpath.ai/observability-service/action/data/v3/telemetry")
TELEMETRY_TIMEOUT = 60.0

async def send_telemetry(telemetry_data: Dict) -> Dict:
    """
    Background task to send telemetry events to the API.
    Non-blocking: uses httpx.AsyncClient so the event loop is not blocked.

    Args:
        telemetry_data: The telemetry data to send

    Returns:
        Dict containing status code and response
    """
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "dataType": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=TELEMETRY_TIMEOUT) as client:
            response = await client.post(
                TELEMETRY_API_URL,
                headers=headers,
                json=telemetry_data,
            )
        result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
        }
        logger.info(f"Telemetry sent successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error sending telemetry: {str(e)}")
        return {"error": str(e)} 