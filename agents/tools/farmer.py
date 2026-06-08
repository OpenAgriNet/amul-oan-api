"""
Tool for fetching farmer details by mobile number from PashuGPT-style APIs.
Uses amulpashudhan.com first, then herdman.live if needed (cohesive output, fallback on failure/empty).
"""
import json
import os

from agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    _fetch_farmer_amulpashudhan_raw,
    _fetch_farmer_herdman_raw,
    normalize_phone, merge_farmer_data, merge_farmer_records,
)
from agents.models.farmer import FarmerRecord
from app.models.farmer import FarmerModel
from app.models.union import UnionName
from helpers.utils import get_logger, is_from_union

logger = get_logger(__name__)


async def get_farmer_data_by_mobile(mobile_number: str) -> list[FarmerModel] | None:
    """
    Fetch farmer records by mobile number (same backends as get_farmer_by_mobile).
    Returns structured list of farmer records for use by chat/service layer.

    Args:
        mobile_number: The mobile number of the farmer. Can include +91 or spaces.

    Returns:
        List of farmer record dicts, or None if invalid mobile, no tokens, or no data.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return None

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.error("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        return None

    records: list[FarmerModel] = []

    if token1:
        try:
            data = await fetch_farmer_amulpashudhan(mobile, token1)
            if data is not None:
                records.extend(data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from amulpashudhan")
        except Exception as e:
            logger.warning(f"amulpashudhan farmer API error for {mobile}: {e}")

    if token3 and is_from_union(records, UnionName.MEHSANA):
        try:
            data = await fetch_farmer_herdman(mobile, token3)
            if data:
                records.extend(data)
                logger.info(f"Farmer data for {mobile}: got {len(data)} record(s) from herdman")
        except Exception as e:
            logger.warning(f"herdman farmer API error for {mobile}: {e}")

    if len(records) == 0:
        logger.info(f"No farmer data found for mobile {mobile}")
        return None

    return merge_farmer_data(records)


def _record_has_content(rec: dict) -> bool:
    """A farmer row is worth keeping if it carries animal tags, a non-zero
    animal count, or at least an identity (farmer/society name). Mirrors voice's
    has_content gate so empty placeholder rows don't pollute the SWR cache."""
    if rec.get("tagNo") or rec.get("tagNumbers"):
        return True
    total = rec.get("totalAnimals")
    if total not in (None, 0, "0"):
        return True
    return bool(rec.get("farmerName") or rec.get("societyName"))


def _rows_include_mehsana(rows: list[dict]) -> bool:
    """Raw-dict analogue of is_from_union(records, MEHSANA): does any amulpashudhan
    row carry unionName == 'mehsana'? Raw dicts keep original casing, so lowercase
    before comparing (FarmerModel lowercases via validator)."""
    return any(
        str(r.get("unionName") or "").strip().lower() == UnionName.MEHSANA.value
        for r in rows
    )


async def fetch_farmer_info_raw(mobile_number: str) -> list[FarmerRecord] | None:
    """Raw farmer fetch for the SWR farmer cache (Option B).

    Returns RAW camelCase ``FarmerRecord`` objects — NO ``FarmerModel`` snake_case
    normalization — so the shared Redis envelope stays camelCase end-to-end and the
    camelCase-bridge gotcha is avoided. This is the voice/SWR ingestion path; the
    chat path keeps using ``get_farmer_data_by_mobile`` (FarmerModel) unchanged.

    Backends mirror get_farmer_data_by_mobile exactly: amulpashudhan for every
    farmer, plus herdman ONLY for MEHSANA (team decision 2026-06-08 — chat's
    MEHSANA-gating is canonical; voice's herdman-for-everyone was unintended).
    Records from both are merged + de-duplicated (society+farmerCode) and empty
    placeholder rows dropped. Returns None on invalid mobile, missing tokens, or
    no usable data.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return None

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        logger.error("Neither PASHUGPT_TOKEN nor PASHUGPT_TOKEN_3 is set")
        return None

    rows: list[dict] = []

    if token1:
        raw = await _fetch_farmer_amulpashudhan_raw(mobile, token1)
        if raw:
            rows.extend(r for r in raw if isinstance(r, dict))

    # herdman only for MEHSANA — gated on the amulpashudhan rows, exactly as
    # get_farmer_data_by_mobile gates is_from_union(records, MEHSANA).
    if token3 and _rows_include_mehsana(rows):
        raw_h = await _fetch_farmer_herdman_raw(mobile, token3)
        if raw_h:
            rows.extend(r for r in raw_h if isinstance(r, dict))

    if not rows:
        return None

    kept = [r for r in rows if _record_has_content(r)] or rows
    deduped = merge_farmer_records(kept)
    if not deduped:
        return None

    records = [FarmerRecord.model_validate(r) for r in deduped]
    logger.info(f"Raw farmer info for {mobile}: {len(records)} record(s) merged")
    return records or None


async def get_farmer_by_mobile(mobile_number: str) -> str:
    """
    Fetch farmer information by mobile number. Returns farmer details including
    farmer ID, name, location, society, and associated animal tag numbers.
    Tries multiple backends and merges results when both return data.

    Args:
        mobile_number: The mobile number of the farmer (required). Can include +91 or spaces.

    Returns:
        str: Formatted JSON string with farmer details and associated tag numbers,
             or a clear message if no data found. Handles API failures and empty responses.
    """
    records = await get_farmer_data_by_mobile(mobile_number)
    if records is None:
        mobile = normalize_phone(mobile_number) or mobile_number
        return "Please provide a valid mobile number." if not mobile else f"Farmer details for mobile {mobile}:\n\nNo farmer data found for this mobile number."
    mobile = normalize_phone(mobile_number)
    formatted = json.dumps([record.model_dump() for record in records], indent=2, ensure_ascii=False)
    return f"Farmer details for mobile {mobile}:\n\n{formatted}"
