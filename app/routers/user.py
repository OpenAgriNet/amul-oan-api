"""
User profile endpoint — returns farmer data for the authenticated user.
"""
from typing import Any

from fastapi import APIRouter, Depends

from app.auth.jwt_auth import get_current_user
from agents.services.farmer_cache import get_or_fetch_farmer_data
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/me", summary="Get current user's farmer profile")
async def get_user_profile(user_info: dict = Depends(get_current_user)) -> Any:
    """
    Returns farmer profile data for the authenticated user.

    Response status values:
    - `ok`: Farmer data found and returned
    - `not_found`: Authenticated user but no farmer records in PashuGPT
    - `error`: Failed to fetch farmer data (API or cache error)
    - `anonymous`: User has an anonymous token (no phone number)
    """
    if not user_info:
        return {"status": "anonymous", "farmer": None}

    phone = user_info.get("phone") or user_info.get("sub")
    is_anonymous = user_info.get("anonymous") or (
        isinstance(phone, str) and phone.startswith("anon-")
    )

    if not phone or is_anonymous:
        return {"status": "anonymous", "farmer": None}

    try:
        data = await get_or_fetch_farmer_data(phone)
        if data is None or data.lookupStatus == "not_found":
            return {"status": "not_found", "farmer": None}
        return {"status": "ok", "farmer": data.model_dump()}
    except Exception as e:
        logger.error(f"Error fetching farmer profile for phone: {e}")
        return {"status": "error", "farmer": None}
