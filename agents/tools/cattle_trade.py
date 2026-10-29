"""Tool for Amul Pashudhan's average cattle rates by species and breed."""

import re
from decimal import ROUND_HALF_UP, Decimal

from agents.tools.telemetry import report_tool
from agents.tools.cattle_trade_backend import (
    get_cattle_trade_rates_api,
    get_verified_animal_details_api,
)
from agents.tools.models.cattle_trade import (
    CattleTradeRatesModel,
    VerifiedAnimalDetailsModel,
)
from app.config import get_config_value
from helpers.utils import get_logger

logger = get_logger(__name__)

# Provider breedName (as a match key) -> (farmer-facing name, species, other spellings).
# The provider has no species field and misspells some names (Jursy, Kakrej), and the
# pre-translation glossary emits others (Mehsani, Kankrej Cow, Banni Buffalo, crossbreed),
# so every spelling a farmer or the glossary uses is listed here. Names are the forms the
# Gujarati translator renders cleanly: "Mehsana" came out as મેસાના and "Shankar" as શંકર.
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
    "shankar": ("Crossbred", "cow", ("sankar", "crossbred", "crossbreed", "cross")),
    "banni": ("Banni", "buffalo", ("bani", "kutchi", "kachchhi")),
    "jaffrabadi": ("Jafarabadi", "buffalo", ("jafarabadi", "jafrabadi", "jaffarabadi")),
    "mehsana": ("Mehsani", "buffalo", ("mehsani", "mahesani", "mahesana", "mehasana")),
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

    Use it for questions about average cow, buffalo or breed prices. Use the verified
    cattle-listings tool instead when the farmer asks which animals are currently for sale
    or requests seller contact details. Rates exist only for cows and buffaloes. It is not
    for milk or fat rates, and not for mandi crop prices.

    Args:
        species: "cow" or "buffalo" to get that species' average and its breed rates. Leave empty for all rates.
        breed: Breed name in English as the farmer said it (e.g. "Gir", "Mehsani", "Jersey cross"). Common spellings are matched; a breed with no rate is reported as unavailable.

    Returns:
        The matching average rates, or a message saying the rate is unavailable.
    """
    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        report_tool("unavailable", "PASHUGPT_TOKEN is not set")
        return (
            "Cattle price lookup failed: Amul Pashudhan access is not configured. "
            "Say the rates are unavailable right now. Do not estimate or invent a price."
        )

    rates = await get_cattle_trade_rates_api(token)
    if rates is None:
        report_tool("unavailable", "Amul Pashudhan rates request failed")
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
        report_tool("no_result", "rates exist only for cows and buffaloes")
        return (
            "Amul Pashudhan average rates are available only for cows and buffaloes. Say only "
            "that, without naming other animals, and do not estimate or invent a price."
        )

    return "\n".join([*_full_listing(rates, breeds), _CAVEAT])


def _text_key(value: str | None) -> str:
    return "".join(char for char in (value or "").casefold() if char.isalnum())


def _breed_matches(actual: str, requested: str) -> bool:
    actual_breed = _Breed(actual, Decimal(0))
    key = _breed_key(requested) or _match_key(requested)
    return key in actual_breed.keys


def _number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def _listed_price(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"₹{int(value)}"
    return f"₹{value:.2f}".rstrip("0").rstrip(".")


def _age(value: str | None) -> str:
    if not value:
        return "age not provided"
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if not match:
        return f"age {value}"
    return f"age {int(match.group(1))} years {int(match.group(2))} months"


def _date(value) -> str:
    return f"{value.day} {value.strftime('%B %Y')}"


def _verified_animals(details: VerifiedAnimalDetailsModel):
    for groups in (details.cows, details.buffaloes):
        for group in groups:
            for animal in group.animals:
                breed = animal.breed or group.breed_name
                yield animal, breed


def _filter_animals(
    details: VerifiedAnimalDetailsModel,
    species: str | None,
    breed: str | None,
    district: str | None,
    village: str | None,
    max_price: float | None,
    min_milk_per_day: float | None,
    milking_status: str | None,
    pregnancy_status: str | None,
):
    requested_species = _species_of(species or "")
    if species and not requested_species:
        return []
    maximum = Decimal(str(max_price)) if max_price is not None else None
    minimum_milk = Decimal(str(min_milk_per_day)) if min_milk_per_day is not None else None
    matches = []
    for animal, animal_breed in _verified_animals(details):
        if requested_species and _species_of(animal.cattle_type) != requested_species:
            continue
        if breed and not _breed_matches(animal_breed, breed):
            continue
        if district and _text_key(animal.district_name) != _text_key(district):
            continue
        if village and _text_key(animal.village_name) != _text_key(village):
            continue
        if maximum is not None and animal.cattle_price > maximum:
            continue
        if minimum_milk is not None and animal.milk_per_day < minimum_milk:
            continue
        if milking_status and _text_key(animal.milking_status) != _text_key(milking_status):
            continue
        if pregnancy_status and _text_key(animal.pregnancy_status) != _text_key(
            pregnancy_status
        ):
            continue
        matches.append((animal, animal_breed))
    return matches


def _animal_line(index, animal, breed: str, include_contact: bool) -> str:
    species = _species_of(animal.cattle_type) or animal.cattle_type.casefold()
    breed_label = (
        "breed not provided" if _match_key(breed) in _PLACEHOLDER_BREED_KEYS else breed
    )
    location = ", ".join(
        value for value in (animal.village_name, animal.district_name) if value
    ) or "location not provided"
    parts = [
        f"{breed_label} {species}",
        location,
        _listed_price(animal.cattle_price),
        _age(animal.age),
        f"lactation {animal.lactation or 'not applicable'}",
        animal.milking_status or "milking status not provided",
        animal.pregnancy_status or "pregnancy status not provided",
        (
            f"milk per day {_number(animal.milk_per_day)}"
            if animal.milk_per_day
            else "milk per day not applicable"
        ),
    ]
    if include_contact:
        parts.append(f"seller {animal.farmer_name or 'name not provided'}")
        parts.append(f"mobile {animal.farmer_mobile_number or 'not provided'}")
    return f"{index}. " + " — ".join(parts) + "."


async def get_verified_cattle_listings(
    species: str | None = None,
    breed: str | None = None,
    district: str | None = None,
    village: str | None = None,
    max_price: float | None = None,
    min_milk_per_day: float | None = None,
    milking_status: str | None = None,
    pregnancy_status: str | None = None,
    include_contact: bool = False,
) -> str:
    """
    Get every verified cow or buffalo currently listed for sale by Amul Pashudhan.

    Use this tool when a farmer asks which animals are available, listed or for sale,
    asks for individual animal details, or explicitly asks for a listed animal's seller
    contact. Use the cattle-price tool instead for average price questions. Return every
    matching listing; do not rank or truncate the provider results. Set include_contact
    only after the farmer explicitly asks for seller contact or says they want to contact
    the seller about a specific, unambiguous animal.

    Args:
        species: Optional "cow" or "buffalo" filter.
        breed: Optional breed filter, including common spellings such as Gir or Mehsani.
        district: Optional exact district-name filter.
        village: Optional exact village-name filter.
        max_price: Optional maximum listed price in rupees.
        min_milk_per_day: Optional minimum provider-reported milk-per-day value.
        milking_status: Optional exact status such as "Milking" or "Dry".
        pregnancy_status: Optional exact status such as "Pregnant" or "Non Pregnant".
        include_contact: Whether to include seller name and mobile after an explicit request.

    Returns:
        Every matching verified listing, a no-result message, or an unavailable message.
    """
    token = get_config_value("PASHUGPT_TOKEN")
    if not token:
        report_tool("unavailable", "PASHUGPT_TOKEN is not set")
        return "Current verified-animal listings are unavailable right now. Please try again later."

    details = await get_verified_animal_details_api(token)
    if details is None or details.from_date.year == 1 or details.to_date.year == 1:
        report_tool("unavailable", "Amul Pashudhan verified-animal request failed")
        return "Current verified-animal listings are unavailable right now. Please try again later."

    has_filters = any(
        value is not None and value != ""
        for value in (
            species,
            breed,
            district,
            village,
            max_price,
            min_milk_per_day,
            milking_status,
            pregnancy_status,
        )
    )
    matches = _filter_animals(
        details,
        species,
        breed,
        district,
        village,
        max_price,
        min_milk_per_day,
        milking_status,
        pregnancy_status,
    )
    window = f"{_date(details.from_date)} to {_date(details.to_date)}"
    if not matches:
        report_tool("no_result", "no verified cattle listings matched")
        if not has_filters:
            return (
                f"No verified cows or buffaloes are currently listed for the provider's "
                f"{window} window."
            )
        return (
            f"No verified animals match the request for the provider's {window} window. "
            "Try another breed, location or filter."
        )

    show_contact = include_contact and len(matches) == 1
    if not has_filters:
        header = (
            f"Found {len(matches)} verified animals for the provider window {window}: "
            f"{details.total_cows} cows and {details.total_buffaloes} buffaloes."
        )
    else:
        noun = "animal" if len(matches) == 1 else "animals"
        header = f"Found {len(matches)} verified {noun} matching the request for {window}."

    lines = [header]
    lines.extend(
        _animal_line(index, animal, animal_breed, show_contact)
        for index, (animal, animal_breed) in enumerate(matches, 1)
    )
    if include_contact and len(matches) > 1:
        lines.append(
            "More than one animal matches. Ask the farmer to identify one listing before "
            "sharing seller contact details."
        )
    elif not include_contact:
        lines.append("Seller contact is available on request for a specific animal.")
    return "\n".join(lines)
