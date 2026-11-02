"""Direct provider adapter for cattle trade average-rate data."""

import json
from decimal import Decimal
from typing import TypeVar

import httpx
from pydantic import BaseModel

from agents.tools.models.cattle_trade import (
    CattleTradeRatesModel,
    VerifiedAnimalDetailsModel,
)
from agents.tools.telemetry import reason
from app.config import settings
from app.observability import record_api_result, start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


async def _get_provider_model(
    token: str, endpoint: str, observation_name: str, model: type[ModelT]
) -> ModelT | None:
    api_url = f"{settings.amulpashudhan_base_url}/{endpoint}"
    try:
        with start_observation(
            observation_name,
            input={},
            metadata={"provider": "amulpashudhan", "url": api_url},
        ) as observation:
            try:
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
            except Exception as exc:
                record_api_result(observation, ok=False, reason=reason(exc))
                raise
            record_api_result(
                observation,
                ok=response.is_success,
                status_code=response.status_code,
                reason=f"HTTP {response.status_code}",
            )
            response.raise_for_status()

        payload = json.loads(response.text, parse_float=Decimal)
        if not isinstance(payload, dict):
            raise ValueError("cattle trade provider response is not an object")
        return model.model_validate(payload)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "%s HTTP failure (status=%s)",
            observation_name,
            exc.response.status_code,
            exc_info=True,
        )
    except (json.JSONDecodeError, ValueError):
        logger.error("%s returned an invalid response", observation_name, exc_info=True)
    except Exception:
        logger.error("%s request failed", observation_name, exc_info=True)
    return None


async def get_cattle_trade_rates_api(token: str) -> CattleTradeRatesModel | None:
    """Fetch the provider's species and named-breed average rates."""
    return await _get_provider_model(
        token, "GetBreedWiseAverageRate", "get_cattle_trade_rates_api", CattleTradeRatesModel
    )


async def get_verified_animal_details_api(token: str) -> VerifiedAnimalDetailsModel | None:
    """Fetch verified cattle-trade listings for the provider's rolling window."""
    return await _get_provider_model(
        token,
        "GetVerifiedAnimalDetails",
        "get_verified_animal_details_api",
        VerifiedAnimalDetailsModel,
    )
