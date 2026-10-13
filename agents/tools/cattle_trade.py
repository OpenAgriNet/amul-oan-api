"""Tool for current cattle trade species and breed average rates."""

from decimal import Decimal

from agents.tools.cattle_trade_backend import get_cattle_trade_rates_api
from agents.tools.models.cattle_trade import (
    BreedAverageRateModel,
    CattleTradeRatesModel,
)
from app.config import get_config_value
from helpers.utils import get_logger

logger = get_logger(__name__)

_PLACEHOLDER_BREED_KEYS = {"other", "otherbuffalo"}
_COW_KEYS = {"cow", "cows", "femalecow", "femalecows"}
_BUFFALO_KEYS = {
    "buffalo",
    "buffalos",
    "buffaloes",
    "femalebuffalo",
    "femalebuffalos",
    "femalebuffaloes",
}


def _match_key(value: str) -> str:
    """Normalize only case, whitespace, and punctuation; never infer aliases."""
    return "".join(char for char in value.casefold() if char.isalnum())


def _named_breeds(rates: CattleTradeRatesModel) -> list[BreedAverageRateModel]:
    """Remove non-breed placeholder categories from every output and lookup."""
    return [
        row
        for row in rates.breeds
        if _match_key(row.breed_name) not in _PLACEHOLDER_BREED_KEYS
    ]


def _format_rate(value: Decimal) -> str:
    return f"{value:,.2f}"


def _unavailable_breed_message(
    requested_breed: str,
    available_breeds: list[BreedAverageRateModel],
) -> str:
    names = ", ".join(row.breed_name for row in available_breeds)
    suffix = f" Available named breeds: {names}." if names else ""
    return (
        f'No average price data is available for breed "{requested_breed}" '
        f"in the provider response.{suffix} Do not estimate or infer a price."
    )


async def get_cattle_trade_prices(
    species: str | None = None,
    breed: str | None = None,
) -> str:
    """
    Fetch current provider-reported average cattle trade prices.

    Always use this tool when a farmer says they want to buy a cow or buffalo,
    asks the average cow/buffalo price, asks for a named breed price, compares
    cattle prices, or asks which breed prices are available. Do not answer such
    price questions from memory or document search. The provider may not contain
    every breed; an unavailable breed must be reported as unavailable, never
    estimated or mapped to another breed.

    Args:
        species: Cow or buffalo for a species-level average. Leave empty when
            requesting the full catalogue or when breed is supplied.
        breed: Breed name exactly as the farmer said it. Matching ignores only
            case, spaces, and punctuation; it does not guess aliases.

    Returns:
        Current exact average-rate data, or a clear unavailable/failure message.
    """
    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return (
            "Cattle trade price lookup failed. Provider access is not configured. "
            "Do not estimate or invent a price."
        )

    rates = await get_cattle_trade_rates_api(token)
    if rates is None:
        return (
            "Cattle trade price lookup failed. Current provider price data is "
            "unavailable. Do not estimate or invent a price."
        )

    available_breeds = _named_breeds(rates)
    requested_breed = (breed or "").strip()
    if requested_breed:
        requested_key = _match_key(requested_breed)
        matched = next(
            (
                row
                for row in available_breeds
                if _match_key(row.breed_name) == requested_key
            ),
            None,
        )
        if matched is None:
            return _unavailable_breed_message(requested_breed, available_breeds)
        return (
            f"Provider-reported average price for {matched.breed_name}: "
            f"{_format_rate(matched.average_rate)}. Use this exact value; "
            "do not infer a price for any other breed."
        )

    requested_species = (species or "").strip()
    if requested_species:
        species_key = _match_key(requested_species)
        if species_key in _COW_KEYS:
            return (
                "Provider-reported average price for a cow: "
                f"{_format_rate(rates.female_cow)}."
            )
        if species_key in _BUFFALO_KEYS:
            return (
                "Provider-reported average price for a buffalo: "
                f"{_format_rate(rates.female_buffalo)}."
            )
        return (
            f'No species-level average price data is available for "{requested_species}". '
            "The provider supplies species averages only for cow and buffalo. "
            "Do not estimate or infer a price."
        )

    lines = [
        "Current provider-reported cattle trade average prices:",
        f"- Cow: {_format_rate(rates.female_cow)}",
        f"- Buffalo: {_format_rate(rates.female_buffalo)}",
        "Named breed average prices:",
    ]
    lines.extend(
        f"- {row.breed_name}: {_format_rate(row.average_rate)}"
        for row in available_breeds
    )
    lines.append(
        "Only the values listed above are available. Do not estimate prices for "
        "unlisted breeds."
    )
    return "\n".join(lines)
