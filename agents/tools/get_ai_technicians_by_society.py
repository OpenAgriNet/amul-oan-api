"""
Tool for fetching AI technicians by union and society code from PashuGPT.
"""
import asyncio
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from helpers.utils import get_logger

logger = get_logger(__name__)

BASE_AMULPASHUDHAN = "https://api.amulpashudhan.com/configman/v1/PashuGPT"


class GetAITechniciansBySocietyQueryParams(BaseModel):
    union_code: str = Field(..., alias="unionCode")
    society_code: str = Field(..., alias="societyCode")

    def to_query_params(self) -> dict[str, str]:
        return {
            "unionCode": self.union_code,
            "societyCode": self.society_code,
        }


class AITechnicianBySocietyResponseModel(BaseModel):
    user_id: str = Field(..., alias="userId")
    full_name: str = Field(..., alias="fullName")
    mobile_number: str = Field(..., alias="mobileNumber")


async def get_ai_technicians_by_society(
    query: GetAITechniciansBySocietyQueryParams,
) -> list[AITechnicianBySocietyResponseModel]:
    """
    Fetch AI technicians for a given union code and society code.

    Args:
        query: Query parameters containing union and society code.

    Returns:
        List of AI technicians for the provided society.
    """
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        raise ValueError("PASHUGPT_TOKEN is not set")

    url = f"{BASE_AMULPASHUDHAN}/GetAITUserDetailsBySocietyCode"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=query.to_query_params(),
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            response.raise_for_status()

        response_json: Any = response.json()
        if not isinstance(response_json, list):
            raise ValueError("Expected list response from API")

        return [
            AITechnicianBySocietyResponseModel.model_validate(
                item,
                extra="ignore",
                by_alias=True,
            )
            for item in response_json
        ]
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[AmulPashudhan(%s, %s)] :: Request failed with status code %s, message=%s",
            query.union_code,
            query.society_code,
            exc.response.status_code,
            exc.response.text,
            exc_info=True,
        )
        raise
    except Exception:
        logger.exception(
            "[AmulPashudhan(%s, %s)] :: Failed to fetch AI technicians.",
            query.union_code,
            query.society_code,
        )
        raise


if __name__ == "__main__":
    async def main() -> None:
        query = GetAITechniciansBySocietyQueryParams(
            unionCode="2021",
            societyCode="1066",
        )
        technicians = await get_ai_technicians_by_society(query)
        for technician in technicians:
            print(technician.model_dump(mode="json", by_alias=True))

    asyncio.run(main())
