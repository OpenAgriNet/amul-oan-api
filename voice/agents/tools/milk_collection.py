"""Tool for fetching farmer milk collection and deduction details."""
import json
import os

from pydantic import ValidationError

from voice.app.models.milk_collection import FarmerMilkCollectionRequestModel
from voice.agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
from voice.helpers.utils import get_logger

logger = get_logger(__name__)


async def get_farmer_milk_collection_details(
    union_code: str,
    society_code: str,
    farmer_code: str,
    fromdate: str,
    todate: str,
) -> str:
    """
    Fetch milk collection and deduction details for a farmer.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        fromdate: Start date in DD-MM-YYYY format.
        todate: End date in DD-MM-YYYY format.

    Returns:
        str: Formatted milk collection and deduction details, or a clear failure message.
    """
    logger.info(
        "Milk collection tool invoked: union=%s society=%s farmer=%s fromdate=%s todate=%s",
        union_code,
        society_code,
        farmer_code,
        fromdate,
        todate,
    )

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Milk collection lookup failed. Service is not configured."

    try:
        request = FarmerMilkCollectionRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            fromdate=fromdate,
            todate=todate,
        )
        request.validate_date_range()
    except (ValidationError, ValueError) as e:
        logger.info(
            "Milk collection lookup validation failed: union=%s society=%s farmer=%s fromdate=%s todate=%s error=%s",
            union_code,
            society_code,
            farmer_code,
            fromdate,
            todate,
            e,
        )
        return f"Milk collection lookup failed. {e}"

    response = await get_farmer_milk_collection_details_api(request, token)
    if response is None:
        logger.info(
            "Milk collection API failed: union=%s society=%s farmer=%s fromdate=%s todate=%s",
            union_code,
            society_code,
            farmer_code,
            fromdate,
            todate,
        )
        return "Milk collection lookup failed. Unable to fetch details at the moment."

    formatted = json.dumps(response.model_dump(by_alias=True), indent=2, ensure_ascii=False)
    return f"Milk collection details fetched successfully:\n\n{formatted}"
