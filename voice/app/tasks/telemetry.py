"""
Tasks for sending telemetry data (async, non-blocking).
"""
import os
from dotenv import load_dotenv
from typing import Any, Dict
import httpx
from voice.helpers.utils import get_logger

load_dotenv()

logger = get_logger(__name__)

TELEMETRY_API_URL = os.getenv(
    "TELEMETRY_API_URL",
    "https://vistaar.kenpath.ai/observability-service/action/data/v3/telemetry",
)


async def send_telemetry(telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send telemetry events to the API (async, non-blocking).

    Args:
        telemetry_data: The telemetry data to send.

    Returns:
        Dict containing status_code and response.
    """
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "dataType": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                TELEMETRY_API_URL,
                headers=headers,
                json=telemetry_data,
            )
        result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
        }
        logger.info("Telemetry sent successfully: %s", result)
        return result
    except Exception as e:
        logger.error("Error sending telemetry: %s", e)
        return {"error": str(e)} 