"""Direct provider adapter for cattle trade average-rate data."""

import json
from decimal import Decimal

import httpx

from agents.tools.models.cattle_trade import CattleTradeRatesModel
from app.config import settings
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)


async def get_cattle_trade_rates_api(token: str) -> CattleTradeRatesModel | None:
    """Fetch the provider's species and named-breed average rates."""
    api_url = f"{settings.amulpashudhan_base_url}/GetBreedWiseAverageRate"
    try:
        with start_observation(
            "get_cattle_trade_rates_api",
            input={},
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            async with httpx.AsyncClient(
                timeout=settings.farmer_backend_http_timeout_seconds
            ) as client:
                response = await client.get(
                    api_url,
                    headers={
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
                response.raise_for_status()
                if observation is not None:
                    observation.update(output={"status_code": response.status_code})

        payload = json.loads(response.text, parse_float=Decimal)
        if not isinstance(payload, dict):
            raise ValueError("cattle trade provider response is not an object")
        return CattleTradeRatesModel.model_validate(payload)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Cattle trade rate provider HTTP failure (status=%s)",
            exc.response.status_code,
            exc_info=True,
        )
    except (json.JSONDecodeError, ValueError):
        logger.error("Cattle trade rate provider returned an invalid response", exc_info=True)
    except Exception:
        logger.error("Cattle trade rate provider request failed", exc_info=True)
    return None
