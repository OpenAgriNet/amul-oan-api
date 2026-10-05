"""Direct provider adapter for bonus data, which has no Beckn operation yet."""

import json

import httpx

from agents.tools.models.bonus import FarmerBonusAmountRecordModel, FarmerBonusAmountRequestModel
from app.config import settings
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_farmer_bonus_amount_api(
    request: FarmerBonusAmountRequestModel,
    token: str,
) -> list[FarmerBonusAmountRecordModel] | None:
    """Fetch bonus records from the sole provider integration."""
    api_url = f"{settings.amulpashudhan_base_url}/GetFarmerBonusAmount"
    try:
        with start_observation(
            "get_farmer_bonus_amount_api",
            input=request.to_query_params(),
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(
                timeout=settings.farmer_backend_http_timeout_seconds
            ) as client:
                response = await client.get(
                    api_url,
                    params=request.to_query_params(),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                if observation is not None:
                    observation.update(output={"status_code": response.status_code})
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("bonus provider response is not a list")
        return [FarmerBonusAmountRecordModel.model_validate(item) for item in payload]
    except httpx.HTTPStatusError as exc:
        body = exc.response.text or ""
        if "Farmer bonus data not found" in body:
            return []
        if "Bonus amount not supported" in body:
            logger.warning("Bonus provider does not support this union: %s", body)
        else:
            logger.error("Bonus provider HTTP failure: %s", body, exc_info=True)
    except (json.JSONDecodeError, ValueError):
        logger.error("Bonus provider returned an invalid response", exc_info=True)
    except Exception:
        logger.error("Bonus provider request failed", exc_info=True)
    return None
