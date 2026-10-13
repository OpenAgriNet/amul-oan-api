"""Tool for Amul Pashudhan's average cattle rates by species and breed."""

import re
from decimal import ROUND_HALF_UP, Decimal

from agents.tools.cattle_trade_backend import get_cattle_trade_rates_api
from agents.tools.models.cattle_trade import CattleTradeRatesModel
from app.config import get_config_value
from helpers.utils import get_logger

logger = get_logger(__name__)

# Provider breedName (as a match key) -> (farmer-facing name, species, other spellings).
# The provider has no species field and misspells some names (Jursy, Kakrej), and the
# pre-translation glossary emits others (Mehsani, Kankrej Cow, Banni Buffalo, crossbreed),
# so every spelling a farmer or the glossary uses is listed here.
_KNOWN_BREEDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gir": ("Gir", "cow", ("gyr",)),
    "kakrej": ("Kankrej", "cow", ("kankrej", "kankreji", "kakreji", "kankraj")),
    "sahiwal": ("Sahiwal", "cow", ("sahival",)),
    "jursy": ("Jersey", "cow", ("jersey", "jersy", "jarsi", "jarsey")),
    "jursyshankar": (
        "Jersey crossbred",
        "cow",
        ("jerseycross", "jerseycrossbred", "jerseycrossbreed", "jerseyshankar", "jarsisankar"),
    ),
    "shankar": ("Crossbred (Shankar)", "cow", ("sankar", "crossbred", "crossbreed", "cross")),
    "banni": ("Banni", "buffalo", ("bani", "kutchi", "kachchhi")),
    "jaffrabadi": ("Jafarabadi", "buffalo", ("jafarabadi", "jafrabadi", "jaffarabadi")),
    "mehsana": ("Mehsana", "buffalo", ("mehsani", "mahesani", "mahesana", "mehasana")),
    "murrah": ("Murrah", "buffalo", ("murra", "mura")),
    "surti": ("Surti", "buffalo", ("surati",)),
}
# Catch-all rows, not breeds: never quoted or matched.
_PLACEHOLDER_BREED_KEYS = {"other", "otherbuffalo"}

_SPECIES_WORDS = {
    "cow": {"cow", "cows", "gai", "gay", "gaay", "femalecow"},
    "buffalo": {"buffalo", "buffalos", "buffaloes", "bhens", "bhains", "bhes", "femalebuffalo"},
}
# A species argument that means "any": list every rate.
_ANY_SPECIES_WORDS = {"cattle", "animal", "animals", "livestock", "pashu", "dhor", "all", "both"}
# Words a breed name may carry that are not part of the breed itself.
_NOISE_WORDS = {"breed", "jaat", "jat", "olad", "nasl", "female"}


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _match_key(value: str) -> str:
    return "".join(_words(value))


def _species_of(value: str) -> str | None:
    key = _match_key(value)
    for species, words in _SPECIES_WORDS.items():
        if key in words:
            return species
    return None


def _breed_key(value: str) -> str:
    """Match key for a breed name, dropping species and filler words ("Mehsani buffalo")."""
    species_words = set().union(*_SPECIES_WORDS.values())
    return "".join(
        word for word in _words(value) if word not in species_words | _NOISE_WORDS
    )


def _rupees(value: Decimal) -> str:
    """Whole rupees rounded to the nearest 100, without digit grouping.

    The provider returns paise-level averages; that precision means nothing to a
    buyer, and decimals and grouped digits are both misread downstream (translation,
    and voice number-to-words splits on commas).
    """
    hundreds = (value / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return f"₹{int(hundreds) * 100}"


class _Breed:
    def __init__(self, provider_name: str, average_rate: Decimal):
        known = _KNOWN_BREEDS.get(_match_key(provider_name))
        self.name = known[0] if known else provider_name
        self.species = known[1] if known else None
        self.keys = {_match_key(provider_name), _breed_key(self.name)}
        if known:
            self.keys.update(known[2])
        self.rate = average_rate


def _breeds(rates: CattleTradeRatesModel) -> list[_Breed]:
    breeds = [
        _Breed(row.breed_name, row.average_rate)
        for row in rates.breeds
        if _match_key(row.breed_name) not in _PLACEHOLDER_BREED_KEYS
    ]
    return sorted(breeds, key=lambda breed: breed.name)


def _species_rate(rates: CattleTradeRatesModel, species: str) -> Decimal:
    return rates.female_cow if species == "cow" else rates.female_buffalo


def _species_line(rates: CattleTradeRatesModel, species: str) -> str:
    label = "Cows" if species == "cow" else "Buffaloes"
    return f"{label} (all breeds): about {_rupees(_species_rate(rates, species))}"


def _breed_list(breeds: list[_Breed], species: str | None) -> str:
    return ", ".join(
        f"{breed.name} {_rupees(breed.rate)}" for breed in breeds if breed.species == species
    )


_CAVEAT = (
    "These are averages, rounded to the nearest ₹100. An individual animal's price "
    "depends on its milk yield, age, lactation and health. Quote only these figures, "
    "as approximate averages."
)
_HEADER = "Amul Pashudhan average rates per animal:"


def _full_listing(rates: CattleTradeRatesModel, breeds: list[_Breed]) -> list[str]:
    lines = [_HEADER]
    for species in ("cow", "buffalo"):
        lines.append(f"- {_species_line(rates, species)}")
        by_breed = _breed_list(breeds, species)
        if by_breed:
            lines.append(f"  By breed: {by_breed}")
    unclassified = _breed_list(breeds, None)
    if unclassified:
        lines.append(f"- Other breeds (species not given by Amul Pashudhan): {unclassified}")
    return lines


async def get_cattle_trade_prices(
    species: str | None = None,
    breed: str | None = None,
) -> str:
    """
    Get Amul Pashudhan's current average rate, in rupees per animal, for cows, buffaloes and named breeds.

    Use it for any question about what a cow or buffalo costs or would sell for, including
    when the farmer wants to buy or sell one. Rates exist only for cows and buffaloes (no bulls,
    bullocks or other animals). It is not for milk or fat rates, and not for mandi crop prices.

    Args:
        species: "cow" or "buffalo" to get that species' average and its breed rates. Leave empty for all rates.
        breed: Breed name in English as the farmer said it (e.g. "Gir", "Mehsani", "Jersey cross"). Common spellings are matched; a breed with no rate is reported as unavailable.

    Returns:
        The matching average rates, or a message saying the rate is unavailable.
    """
    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return (
            "Cattle price lookup failed: Amul Pashudhan access is not configured. "
            "Say the rates are unavailable right now. Do not estimate or invent a price."
        )

    rates = await get_cattle_trade_rates_api(token)
    if rates is None:
        return (
            "Cattle price lookup failed: Amul Pashudhan rates are unavailable right now. "
            "Say so. Do not estimate or invent a price."
        )

    breeds = _breeds(rates)
    requested_breed = (breed or "").strip()
    requested_species = _species_of(species or "") or _species_of(requested_breed)

    if requested_breed and not _species_of(requested_breed):
        key = _breed_key(requested_breed) or _match_key(requested_breed)
        matched = next((b for b in breeds if key in b.keys), None)
        if matched is None:
            logger.info("Cattle price requested for unlisted breed %r", requested_breed)
            return "\n".join(
                [
                    f'Amul Pashudhan has no average rate for the breed "{requested_breed}". '
                    "Tell the farmer this breed's rate is not available. You may give the "
                    "overall cow or buffalo average below, labelled as such; never present it, "
                    "or another breed's rate, as this breed's price.",
                    *_full_listing(rates, breeds),
                    _CAVEAT,
                ]
            )
        lines = [
            f"Amul Pashudhan average rate for a {matched.name}"
            f"{' ' + matched.species if matched.species else ''}: "
            f"about {_rupees(matched.rate)} per animal."
        ]
        if matched.species:
            lines.append(_species_line(rates, matched.species) + ".")
        lines.append(_CAVEAT)
        return "\n".join(lines)

    if requested_species:
        lines = [_HEADER, f"- {_species_line(rates, requested_species)}"]
        by_breed = _breed_list(breeds, requested_species)
        if by_breed:
            lines.append(f"  By breed: {by_breed}")
        lines.append(_CAVEAT)
        return "\n".join(lines)

    if (species or "").strip() and _match_key(species) not in _ANY_SPECIES_WORDS:
        return (
            f'Amul Pashudhan average rates cover only cows and buffaloes; there is no rate for "{species.strip()}". '
            "Say so. Do not estimate or invent a price."
        )

    return "\n".join([*_full_listing(rates, breeds), _CAVEAT])
