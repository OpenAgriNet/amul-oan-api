"""
Tool for fetching farmer milk collection and deduction details.
"""
import json
import os

from agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
from app.models.milk_collection import FarmerMilkCollectionRequestModel
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_farmer_milk_collection_details(
    union_code: str,
    society_code: str,
    farmer_code: str,
    fromdate: str,
    todate: str,
) -> str:
    """
    Retrieve farmer milk collection records and deduction entries for a date range.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        fromdate: Start date in YYYY-MM-DD format.
        todate: End date in YYYY-MM-DD format.

    Returns:
        str: Formatted JSON response with milk and deduction details, or a clear failure message.
    """
    logger.info(
        "Farmer milk collection tool invoked for union=%s society=%s farmer=%s from=%s to=%s",
        union_code,
        society_code,
        farmer_code,
        fromdate,
        todate,
    )

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Milk collection lookup failed.\n\nPASHUGPT_TOKEN is not configured."

    try:
        request = FarmerMilkCollectionRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            fromdate=fromdate,
            todate=todate,
        )
        request.validate_date_range()
    except ValueError as exc:
        logger.info(
            "Farmer milk collection validation failed for union=%s society=%s farmer=%s: %s",
            union_code,
            society_code,
            farmer_code,
            str(exc),
        )
        return f"Milk collection lookup failed.\n\n{str(exc)}"

    response = await get_farmer_milk_collection_details_api(request, token)
    if response is None:
        logger.info(
            "Farmer milk collection lookup failed for union=%s society=%s farmer=%s from=%s to=%s",
            union_code,
            society_code,
            farmer_code,
            fromdate,
            todate,
        )
        return (
            "Milk collection lookup failed.\n\n"
            "Unable to fetch milk collection details at the moment."
        )

    formatted = json.dumps(response.model_dump(by_alias=True), indent=2, ensure_ascii=False)
    logger.info(
        "Farmer milk collection lookup succeeded for union=%s society=%s farmer=%s from=%s to=%s milk_records=%s deductions=%s",
        union_code,
        society_code,
        farmer_code,
        fromdate,
        todate,
        len(response.milk),
        len(response.deduction),
    )
    return f"Farmer milk collection details fetched successfully:\n\n{formatted}"
