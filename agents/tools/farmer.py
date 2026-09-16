"""Authenticated farmer profile access through the Amul Beckn BPP."""
import re
import uuid
from enum import Enum

from agents.tools.beckn.amul import fetch_authenticated_farmers
from agents.tools.models.farmer import FarmerModel
from agents.tools.models.farmer_transport import FarmerRecord
from helpers.utils import get_logger

logger = get_logger(__name__)


class FarmerFetchOutcome(str, Enum):
    """Fetch outcome for the SWR cache ingestion path."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"


def normalize_phone_to_mobile(user_id: str) -> str | None:
    """
    Clean user_id as a phone number: strip non-digits, take the last 10 digits.
    Returns None when user_id isn't a usable phone (anon, UUID, name, <10 digits) —
    used by the voice path to derive the caller's mobile from the request user_id.
    """
    if not user_id or not str(user_id).strip():
        return None
    s = str(user_id).strip()
    # Skip anon / anonymous (case-insensitive)
    if s.lower() in ("anon", "anonymous"):
        return None
    # Skip if it's a valid UUID
    try:
        uuid.UUID(s)
        return None
    except (ValueError, AttributeError, TypeError):
        pass
    # Skip if fewer than 10 digits
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return None
    return digits[-10:]


async def get_farmer_data_by_mobile(mobile_number: str) -> list[FarmerModel] | None:
    """Fetch normalized farmer accounts through Beckn init/on_init."""
    mobile = normalize_phone_to_mobile(mobile_number)
    if not mobile:
        return None
    try:
        records = await fetch_authenticated_farmers(mobile)
    except Exception as e:
        logger.warning("Beckn farmer profile lookup failed for %s: %s", mobile, e)
        return None
    return records or None


async def fetch_farmer_info_with_outcome(
    mobile_number: str,
) -> tuple[list[FarmerRecord] | None, FarmerFetchOutcome]:
    """Fetch cache records with an explicit provider outcome."""
    mobile = normalize_phone_to_mobile(mobile_number)
    if not mobile:
        return None, FarmerFetchOutcome.ERROR
    try:
        farmers = await fetch_authenticated_farmers(mobile)
    except Exception as exc:
        logger.warning("Beckn farmer profile lookup failed for %s: %s", mobile, exc)
        return None, FarmerFetchOutcome.ERROR
    if not farmers:
        return None, FarmerFetchOutcome.NOT_FOUND
    records = [FarmerRecord.model_validate(farmer.model_dump()) for farmer in farmers]
    return records, FarmerFetchOutcome.FOUND


async def fetch_farmer_info_raw(mobile_number: str) -> list[FarmerRecord] | None:
    """Backward-compatible wrapper over ``fetch_farmer_info_with_outcome``.

    Returns records only on ``FOUND``; otherwise ``None``.
    """
    records, outcome = await fetch_farmer_info_with_outcome(mobile_number)
    if outcome == FarmerFetchOutcome.FOUND:
        return records
    return None
