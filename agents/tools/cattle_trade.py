"""Tool for Amul Pashudhan's average cattle rates by species and breed."""

import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from pydantic_ai import RunContext
from rapidfuzz import fuzz

from agents.deps import FarmerContext
from agents.tools.cattle_listings_cache import get_snapshot, snapshot_age_seconds
from agents.tools.districts import _TOWN_ALIASES, DISTRICTS, normalize_place, resolve_place
from agents.tools.telemetry import report_tool
from agents.tools.cattle_trade_backend import get_cattle_trade_rates_api
from agents.tools.models.cattle_trade import (
    CattleListingModel,
    CattleListingsSnapshotModel,
    CattleTradeRatesModel,
)
from app.config import get_config_value, settings
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


# ── Verified cattle listings ─────────────────────────────────────────────────
# Listings come from the Redis snapshot (agents/tools/cattle_listings_cache.py).
# Every listing carries a stable number: it keeps pointing at the same animal
# across refreshes and filters, so "number 12" on the next turn is the same animal.

_LIST_CAP = 10
_IST = timezone(timedelta(hours=5, minutes=30))
_UNAVAILABLE = (
    "Current verified-animal listings are unavailable right now. Say so and ask the "
    "farmer to try again later. Do not invent listings."
)
# Vav-Tharad was carved out of Banaskantha; farmer records and farmers still say
# Banaskantha for it, so a Banaskantha search covers both.
_DISTRICT_GROUPS = {"banaskantha": {"banaskantha", "vavtharad"}}
# Coordinates for districts the static table does not have (Tharad town).
_EXTRA_COORDS = {"vavtharad": (24.387, 71.625)}
_DISTRICT_DISPLAY = {"vavtharad": "Vav-Tharad"}
_PLACE_SUFFIXES = ("village", "gaam", "gam", "city", "town")
_MILKING = {
    "milking": "milking", "inmilk": "milking", "lactating": "milking", "yes": "milking",
    "dry": "dry", "notmilking": "dry", "nonmilking": "dry", "no": "dry",
}
_PREGNANCY = {
    "pregnant": "pregnant", "yes": "pregnant", "ingestation": "pregnant",
    "nonpregnant": "nonpregnant", "notpregnant": "nonpregnant", "notpregnancy": "nonpregnant",
    "open": "nonpregnant", "empty": "nonpregnant", "no": "nonpregnant",
}


def _status(value: str | None, aliases: dict[str, str]) -> str | None:
    key = _match_key(value or "")
    return aliases.get(key) if key else None


def _place_key(value: str | None) -> str:
    key = normalize_place(value)
    for suffix in _PLACE_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            key = key[: -len(suffix)]
    return key


def _district_key(value: str | None) -> str:
    """A district name as one key: "KHEDA"/"Kheda" agree, aliases and towns resolve."""
    key = _place_key(value)
    location = resolve_place(key)
    return location.key if location else key


def _district_name(listing: CattleListingModel) -> str:
    key = _district_key(listing.district_name)
    if key in _DISTRICT_DISPLAY:
        return _DISTRICT_DISPLAY[key]
    return DISTRICTS[key].display if key in DISTRICTS else (listing.district_name or "district not given")


def _coords(district_key: str) -> tuple[float, float] | None:
    if district_key in DISTRICTS:
        primary = DISTRICTS[district_key].primary
        return primary.lat, primary.lon
    return _EXTRA_COORDS.get(district_key)


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _fuzzy(requested: str, keys: set[str]) -> set[str]:
    """Contained or near-identical spellings ("Banaskanta", "Mota Mesara" for "Mesara")."""
    if requested in keys:
        return {requested}
    if len(requested) >= 3:
        contained = {key for key in keys if requested in key or (len(key) >= 3 and key in requested)}
        if contained:
            return contained
    ranked = sorted(((fuzz.WRatio(requested, key), key) for key in keys), reverse=True)
    if not ranked or ranked[0][0] < 85:
        return set()
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 5:
        return set()
    return {ranked[0][1]}


def _expand(district_keys: set[str]) -> set[str]:
    expanded = set(district_keys)
    for key in district_keys:
        expanded |= _DISTRICT_GROUPS.get(key, set())
    return expanded


def _place_filter(requested: str, listings: list[CattleListingModel], *, village_first: bool):
    """(listings, label) at a district or village, matched leniently; None when the place is unknown.

    An exact district comes first for a district request and a village first for a
    village request; then a near spelling of a district ("Vav" is Vav-Tharad), then
    a town that belongs to a district (Palanpur is Banaskantha).
    """
    key = _place_key(requested)
    if not key:
        return None
    district_keys = {_district_key(listing.district_name) for listing in listings} - {""}
    village_keys = {_place_key(listing.village_name) for listing in listings} - {""}
    resolved = resolve_place(key)

    def by_district(keys: set[str]):
        keys = _expand(keys)
        found = [listing for listing in listings if _district_key(listing.district_name) in keys]
        names = sorted(
            {_DISTRICT_DISPLAY.get(key) or (DISTRICTS[key].display if key in DISTRICTS else key) for key in keys}
        )
        if not found:
            names = [requested.strip()]
        return found, " and ".join(names) + (" districts" if len(names) > 1 else " district")

    def by_village(keys: set[str]):
        found = [listing for listing in listings if _place_key(listing.village_name) in keys]
        return found, " and ".join(sorted({listing.village_name for listing in found})) + " village"

    exact_district = None
    if key not in _TOWN_ALIASES:
        if resolved is not None:
            exact_district = {resolved.key}
        elif key in district_keys:
            exact_district = {key}
    village = _fuzzy(key, village_keys)
    steps = [
        ("district", exact_district),
        ("village", village or None),
    ]
    if village_first:
        steps.reverse()
    steps += [
        ("district", _fuzzy(key, district_keys) or None),
        ("district", {resolved.key} if resolved is not None else None),
    ]
    for kind, keys in steps:
        if keys:
            return by_district(keys) if kind == "district" else by_village(keys)
    return None


def _breed_name(listing: CattleListingModel) -> str:
    breed = listing.breed or listing.breed_group
    if _match_key(breed) in _PLACEHOLDER_BREED_KEYS:
        return "breed not given"
    known = _KNOWN_BREEDS.get(_match_key(breed))
    return known[0] if known else breed


def _species(listing: CattleListingModel) -> str:
    return _species_of(listing.cattle_type) or listing.cattle_type.casefold()


def _breed_matches(listing: CattleListingModel, requested: str) -> bool:
    breed = _Breed(listing.breed or listing.breed_group, Decimal(0))
    key = _breed_key(requested) or _match_key(requested)
    return bool(key) and key in breed.keys


def _number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def _age(value: str | None) -> str:
    """Provider ages are "years.months": "1.3" is 1 year 3 months, not 1.3 years."""
    match = re.fullmatch(r"(\d+)\.(\d+)", (value or "").strip())
    if not match:
        return f"age {value}" if value else "age not given"
    years, months = (int(part) for part in match.groups())
    parts = []
    if years:
        parts.append(f"{years} {'year' if years == 1 else 'years'}")
    if months:
        parts.append(f"{months} {'month' if months == 1 else 'months'}")
    return (" ".join(parts) or "under 1 month") + " old"


def _date(value: datetime) -> str:
    return f"{value.day} {value.strftime('%B %Y')}"


def _listing_line(listing: CattleListingModel) -> str:
    place = ", ".join(value for value in (listing.village_name, _district_name(listing)) if value)
    milking = _status(listing.milking_status, _MILKING)
    pregnancy = _status(listing.pregnancy_status, _PREGNANCY)
    breed = _breed_name(listing)
    parts = [
        f"{_species(listing)}, breed not given" if breed == "breed not given" else f"{breed} {_species(listing)}",
        place,
        f"₹{_number(listing.cattle_price)}",
        _age(listing.age),
        f"lactation {listing.lactation}" if listing.lactation else "lactation not given",
        {"milking": "giving milk", "dry": "not giving milk"}.get(milking or "", "milking status not given"),
        {"pregnant": "pregnant", "nonpregnant": "not pregnant"}.get(pregnancy or "", "pregnancy not given"),
    ]
    if listing.milk_per_day:
        parts.append(f"{_number(listing.milk_per_day)} litres milk a day")
    parts.append(
        f"seller {listing.farmer_name or 'name not given'}, "
        f"mobile {listing.farmer_mobile_number or 'not given'}"
    )
    # "Number", not "No.": the translator splits sentences at "No." and loses the number.
    return f"Number {listing.number}: " + ", ".join(parts) + "."


def _counts(values) -> str:
    return ", ".join(f"{name} {count}" for name, count in Counter(values).most_common())


def _breakdown(listings: list[CattleListingModel]) -> list[str]:
    """What the matches look like, per filter the farmer could narrow by."""
    lines = []
    species = Counter(_species(listing) for listing in listings)
    if len(species) > 1:
        lines.append(f"- species: {_counts(_species(listing) for listing in listings)}")
    breeds = [_breed_name(listing) for listing in listings]
    if len(set(breeds)) > 1:
        lines.append(f"- breed: {_counts(breeds)}")
    districts = [_district_name(listing) for listing in listings]
    if len(set(districts)) > 1:
        lines.append(f"- district: {_counts(districts)}")
    villages = [listing.village_name for listing in listings if listing.village_name]
    if 1 < len(set(villages)) <= 12:
        lines.append(f"- village: {_counts(villages)}")
    prices = [listing.cattle_price for listing in listings]
    if min(prices) != max(prices):
        lines.append(f"- price: ₹{_number(min(prices))} to ₹{_number(max(prices))}")
    milk = [listing.milk_per_day for listing in listings if listing.milk_per_day]
    if milk:
        lines.append(f"- milk a day: up to {_number(max(milk))} litres ({len(milk)} give milk now)")
    for label, field, aliases, names in (
        ("milking", "milking_status", _MILKING, {"milking": "giving milk", "dry": "not giving milk"}),
        ("pregnancy", "pregnancy_status", _PREGNANCY, {"pregnant": "pregnant", "nonpregnant": "not pregnant"}),
    ):
        values = [names.get(_status(getattr(listing, field), aliases) or "", "not given") for listing in listings]
        if len(set(values)) > 1:
            lines.append(f"- {label}: {_counts(values)}")
    return lines


def _nearest(place_key: str, listings: list[CattleListingModel], exclude: set[str]) -> str:
    """Districts that do have matching animals, nearest first when the place has coordinates."""
    by_district = Counter(
        _district_key(listing.district_name)
        for listing in listings
        if _district_key(listing.district_name) not in exclude
    )
    if not by_district:
        return ""
    names = {_district_key(listing.district_name): _district_name(listing) for listing in listings}
    origin = _coords(place_key)
    if origin:
        ranked = sorted(
            by_district,
            key=lambda key: _km(origin, _coords(key)) if _coords(key) else float("inf"),
        )
        parts = [
            f"{names[key]} {by_district[key]}"
            + (f" (about {round(_km(origin, _coords(key)) / 10) * 10} km)" if _coords(key) else "")
            for key in ranked[:4]
        ]
        return "Nearest districts with matching animals: " + ", ".join(parts) + "."
    parts = [f"{names[key]} {count}" for key, count in by_district.most_common(5)]
    return "Districts with matching animals: " + ", ".join(parts) + "."


def _picked(snapshot: CattleListingsSnapshotModel, number: int, as_of: str) -> str:
    listing = next((item for item in snapshot.listings if item.number == number), None)
    if listing is None:
        report_tool("no_result", "listing number not in the current listings")
        return (
            f"Listing number {number} is not among the current verified listings (it may have been "
            "sold or removed, or the number is wrong). Tell the farmer and offer to search again. "
            "Never give another listing's seller instead."
        )
    return "\n".join(
        [
            f"Verified listing picked by the farmer ({as_of}):",
            _listing_line(listing),
            "Give the farmer these details and the seller's name and mobile exactly as shown, "
            "without assuming the seller's gender, and suggest seeing the animal before paying.",
        ]
    )


async def get_verified_cattle_listings(
    ctx: RunContext[FarmerContext],
    species: str | None = None,
    breed: str | None = None,
    district: str | None = None,
    village: str | None = None,
    max_price: float | None = None,
    min_milk_per_day: float | None = None,
    milking_status: str | None = None,
    pregnancy_status: str | None = None,
    listing_number: int | None = None,
) -> str:
    """
    Find verified cows and buffaloes for sale on Amul Pashudhan, each with its seller's name and mobile.

    Use it when the farmer wants to buy an animal or asks which animals are available or for
    sale. Use get_cattle_trade_prices for average price questions. With no place, the search
    covers the signed-in farmer's district and counts the matches elsewhere. Every listing
    has a fixed number and its seller's name and mobile; when the farmer asks about one
    listing, call with only listing_number.

    Args:
        species: "cow" or "buffalo"; leave empty (or "cattle"/"animal") for both.
        breed: Breed in English as the farmer said it, e.g. "Gir", "Mehsani", "Jersey cross".
        district: District, or a town the farmer names (spelling variants are matched; Banaskantha includes Vav-Tharad).
        village: Village name; spelling variants are matched.
        max_price: Highest listed price in rupees the farmer will pay.
        min_milk_per_day: Lowest milk per day in litres.
        milking_status: "milking" or "dry".
        pregnancy_status: "pregnant" or "not pregnant".
        listing_number: The listing number the farmer picked; returns only that listing (other filters are ignored).

    Returns:
        A count and summary of the matches with numbered listings and seller contacts, one picked listing, or a reason nothing matched.
    """
    snapshot = await get_snapshot()
    if snapshot is None:
        report_tool("unavailable", "verified cattle listings unavailable")
        return _UNAVAILABLE
    fetched = datetime.fromtimestamp(snapshot.fetched_at, _IST)
    as_of = f"Amul Pashudhan data as of {fetched.strftime('%d %B %H:%M')} IST"
    age_hours = snapshot_age_seconds(snapshot) / 3600
    stale = ""
    if age_hours > 2 * settings.cattle_listings_refresh_seconds / 3600:
        stale = (
            f" Amul Pashudhan could not be reached for an update: these listings are about "
            f"{round(age_hours)} hours old, so tell the farmer some may have been sold."
        )
    if listing_number is not None:
        return _picked(snapshot, listing_number, as_of) + stale

    listings = snapshot.listings
    window = (
        f"verified listings from {_date(snapshot.from_date)} to {_date(snapshot.to_date)}, {as_of}"
    )
    if not listings:
        report_tool("no_result", "no verified cattle listed")
        return f"No cows or buffaloes are listed for sale on Amul Pashudhan right now ({window})." + stale

    asked: list[str] = []
    checks = []
    requested_species = _species_of(species or "") or _species_of(breed or "")
    if (species or "").strip() and not requested_species and _match_key(species) not in _ANY_SPECIES_WORDS:
        report_tool("no_result", "only cows and buffaloes are listed")
        return (
            "Amul Pashudhan lists only cows and buffaloes for sale. Say only that, without "
            "naming other animals."
        )
    if requested_species:
        asked.append(requested_species)
        checks.append(("species", lambda item: _species(item) == requested_species))
    if (breed or "").strip() and not _species_of(breed):
        if not any(_breed_matches(item, breed) for item in listings):
            report_tool("no_result", "breed not listed")
            names = _counts(_breed_name(item) for item in listings)
            return (
                f'No "{breed}" animals are listed right now ({window}). Breeds listed now: {names}. '
                "Tell the farmer and offer these instead; do not present another breed as this one."
                + stale
            )
        asked.append(f"breed {breed}")
        checks.append(("breed", lambda item: _breed_matches(item, breed)))
    if max_price is not None:
        maximum = Decimal(str(max_price))
        asked.append(f"up to ₹{_number(maximum)}")
        checks.append(("max price", lambda item: item.cattle_price <= maximum))
    if min_milk_per_day is not None:
        minimum = Decimal(str(min_milk_per_day))
        asked.append(f"at least {_number(minimum)} litres a day")
        checks.append(("milk per day", lambda item: item.milk_per_day >= minimum))
    milking = _status(milking_status, _MILKING)
    if milking:
        asked.append(milking)
        checks.append(("milking status", lambda item: _status(item.milking_status, _MILKING) == milking))
    pregnancy = _status(pregnancy_status, _PREGNANCY)
    if pregnancy:
        asked.append("pregnant" if pregnancy == "pregnant" else "not pregnant")
        checks.append(("pregnancy", lambda item: _status(item.pregnancy_status, _PREGNANCY) == pregnancy))

    def passing(items, skip=None):
        return [item for item in items if all(check(item) for name, check in checks if name != skip)]

    pool = passing(listings)

    place_text = (district or "").strip() or (village or "").strip()
    default_place = None if place_text else ctx.deps.get_farmer_district()
    place_key = _district_key(place_text or default_place)
    lines = []
    if place_text:
        at_place = _place_filter(place_text, listings, village_first=not (district or "").strip())
        if at_place is None:
            report_tool("no_result", "place has no verified listings")
            nearest = _nearest(place_key, pool, set())
            return (
                f'No verified animals are listed in "{place_text}" ({window}). {nearest} '
                "Tell the farmer honestly and offer these places."
                + stale
            ).replace("  ", " ")
        at_place, place_label = at_place
        asked.append(f"in {place_label}")
        place_ids = {item.number for item in at_place}
        matches = [item for item in pool if item.number in place_ids]
        elsewhere = len(pool) - len(matches)
    elif default_place:
        at_place = (_place_filter(default_place, listings, village_first=False) or ([], ""))[0]
        place_ids = {item.number for item in at_place}
        local = [item for item in pool if item.number in place_ids]
        elsewhere = len(pool) - len(local)
        district_label = default_place.title() + (
            " (includes Vav-Tharad)" if place_key == "banaskantha" else ""
        )
        if local:
            matches = local
            lines.append(f"Searched the farmer's district, {district_label}.")
        else:
            matches = sorted(
                pool,
                key=lambda item: _km(_coords(place_key), _coords(_district_key(item.district_name)))
                if _coords(place_key) and _coords(_district_key(item.district_name))
                else float("inf"),
            )
            lines.append(
                f"No matching animals in the farmer's district, {district_label}; these are "
                "from other districts, nearest first."
            )
            elsewhere = 0
    else:
        matches = pool
        elsewhere = 0

    description = ", ".join(asked) if asked else "any cow or buffalo"
    header = (
        f"Verified animals for sale on Amul Pashudhan ({window}). In all: {len(listings)} "
        f"({_counts(_species(item) for item in listings)})."
    )
    if not matches:
        report_tool("no_result", "no verified cattle listings matched")
        relaxed = []
        for name, _ in checks:
            count = len(
                [
                    item for item in passing(listings, skip=name)
                    if not place_text or item.number in place_ids
                ]
            )
            if count:
                relaxed.append(f"without the {name} filter: {count}")
        lines = [header, f"No verified animals match ({description})."]
        if relaxed:
            lines.append("Matches if one filter is dropped: " + "; ".join(relaxed) + ".")
        if place_text:
            nearest = _nearest(place_key, pool, {_district_key(item.district_name) for item in at_place})
            if nearest:
                lines.append(nearest)
        lines.append("Tell the farmer honestly and offer these options. Do not invent listings.")
        return "\n".join(lines) + stale

    lines = [header, *lines, f"Matches ({description}): {len(matches)}."]
    if elsewhere:
        lines.append(f"{elsewhere} more match elsewhere.")
    if len(matches) > 3:
        lines.append("Breakdown of the matches (use it to offer filters):")
        lines.extend(_breakdown(matches))
    shown = matches[:_LIST_CAP]
    lines.append(
        f"Listings{' (first ' + str(len(shown)) + ')' if len(shown) < len(matches) else ''}:"
    )
    lines.extend(_listing_line(item) for item in shown)
    if len(shown) < len(matches):
        lines.append(f"{len(matches) - len(shown)} more not shown; narrowing the filters shows them.")
    if not asked:
        lines.append(
            "The farmer has not narrowed the search yet: lead with the count and where the "
            "animals are, offer two or three filters from the breakdown, and show at most three "
            "listings."
        )
    lines.append(
        "Answer with the count and these listings. Start each listing with its own number as "
        "shown (\"Number 16: ...\"), never a 1, 2, 3 list: the farmer picks by that number. Keep "
        "each seller's name and mobile exactly as shown, without assuming the seller's gender. "
        "If there are many, offer two or three filters from the breakdown."
    )
    return "\n".join(lines) + stale
