"""find_animals: the signed-in farmer's own animals, for chat and voice.

The agent used to get the herd only as a list in its context, with nothing saying
how much of it was loaded or how old it was, and no way to refresh it or look an
animal up. So "Lakshmi is not in my list" came out as "you have no animal named
Lakshmi" when the list was a day old and the name was added since (AMUL-68). Few
animals have a name in the Amul records at all.

This tool says what it knows and how well: how many of the farmer's tags have a
record, how old the records are, how many have a name, and whether a refresh is
running and how far it has got. It starts the refresh itself when the records are
past their interval or the farmer says something changed, waits briefly for it,
and answers with what has landed so far.
"""
import asyncio
import difflib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.farmer_cache import (
    FARMER_ANIMAL_FORCE_MIN_AGE_SECONDS,
    AnimalRefresh,
    _record_tags,
    animals_age_seconds,
    animals_expired,
    get_or_fetch_farmer_data,
    read_animal_refresh,
    start_animal_refresh,
)
from agents.tools.models.farmer_transport import AnimalRecord, FarmerDataEnvelope
from helpers.gujarati_numbers import mask_tag_identifier
from helpers.utils import get_logger

logger = get_logger(__name__)

# How long one call waits for a running refresh before answering with what has
# landed. A voice turn is waiting on it; the agent can call again.
WAIT_SECONDS = 2.0
POLL_SECONDS = 0.25
MAX_MATCHES = 8

_NEGATION = re.compile(r"^(non|not|no)\b")


def _norm(value: Any) -> str:
    text = re.sub(r"[-_/]+", " ", str(value or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()


def _value_matches(want: str, have: Any) -> bool:
    """A filter value against a record's: equal, or every word of it in the record's
    with the same sense ("pregnant" is not "non pregnant")."""
    want, have = _norm(want), _norm(have)
    if not want or not have:
        return False
    if want == have:
        return True
    if bool(_NEGATION.match(want)) != bool(_NEGATION.match(have)):
        return False
    have_words = set(have.split())

    def _has(word: str) -> bool:
        return (
            word in have_words
            or (word.endswith("es") and word[:-2] in have_words)
            or (word.endswith("s") and word[:-1] in have_words)
        )

    return all(_has(word) for word in want.split())


def _name_matches(want: str, have: Optional[str]) -> bool:
    want, have = _norm(want), _norm(have)
    if not want or not have:
        return False
    return want == have or difflib.SequenceMatcher(None, want, have).ratio() >= 0.8


def _merged_animals(
    envelope: FarmerDataEnvelope, refresh: Optional[AnimalRefresh]
) -> tuple[list[str], dict[str, AnimalRecord]]:
    """Every tag, and the record per tag: a running refresh's over the cached one,
    none for a tag the refresh found no record for."""
    tags: list[str] = []
    for farmer in envelope.farmers:
        tags += [tag for tag in _record_tags(farmer) if tag not in tags]
    records = {
        animal.tagNumber: animal
        for farmer in envelope.farmers
        for animal in farmer.animals
        if animal.tagNumber
    }
    if refresh is not None:
        for tag, animal in refresh.fetched.items():
            if animal is None:
                records.pop(tag, None)
            else:
                records[tag] = animal
    return tags, records


def _tag_label(tag: str, spoken: bool) -> str:
    if spoken:
        return mask_tag_identifier(tag)
    return tag


def _animal_out(animal: AnimalRecord, spoken: bool) -> dict:
    out = {"tag": _tag_label(animal.tagNumber or "", spoken)}
    for key, value in (
        ("name", animal.animalName),
        ("type", animal.animalType),
        ("breed", animal.breed),
        ("milking_stage", animal.milkingStage),
        ("pregnancy_stage", animal.pregnancyStage),
        ("date_of_birth", animal.dateOfBirth),
        ("lactation_number", animal.lactationNo),
        ("last_breeding_activity", animal.lastBreedingActivity),
        ("last_health_activity", animal.lastHealthActivity),
    ):
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _counts(records: list[AnimalRecord]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for key, attr in (
        ("type", "animalType"),
        ("breed", "breed"),
        ("milking_stage", "milkingStage"),
        ("pregnancy_stage", "pregnancyStage"),
    ):
        tally: dict[str, int] = {}
        for animal in records:
            value = _norm(getattr(animal, attr))
            if value:
                tally[value] = tally.get(value, 0) + 1
        if tally:
            counts[key] = dict(sorted(tally.items(), key=lambda kv: -kv[1]))
    return counts


def _seconds_since(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - then.astimezone(timezone.utc)).total_seconds()


def _refresh_line(refresh: Optional[AnimalRefresh], refreshed: bool) -> str:
    if refresh is None:
        return "finished just now" if refreshed else "none running"
    if refresh.total is None:
        return "running: updating the farmer record, animals not fetched yet"
    started = _seconds_since(refresh.started_at)
    ago = f", started {int(started)} s ago" if started is not None else ""
    return f"running: {len(refresh.fetched)} of {refresh.total} tags fetched{ago}"


def _filters_given(filters: dict) -> bool:
    return any(value not in (None, "") for value in filters.values())


def _matching(records: list[AnimalRecord], filters: dict) -> list[AnimalRecord]:
    out = []
    for animal in records:
        if filters.get("name") and not _name_matches(filters["name"], animal.animalName):
            continue
        digits = re.sub(r"\D", "", str(filters.get("tag_last_digits") or ""))
        if digits and not (animal.tagNumber or "").endswith(digits):
            continue
        if filters.get("animal_type") and not _value_matches(filters["animal_type"], animal.animalType):
            continue
        if filters.get("breed") and not _value_matches(filters["breed"], animal.breed):
            continue
        if filters.get("milking_stage") and not _value_matches(filters["milking_stage"], animal.milkingStage):
            continue
        if filters.get("pregnancy_stage") and not _value_matches(filters["pregnancy_stage"], animal.pregnancyStage):
            continue
        out.append(animal)
    return out


def _note(
    *,
    filters: dict,
    matches: list[AnimalRecord],
    tags: list[str],
    records: dict[str, AnimalRecord],
    named: int,
    expired: bool,
    refresh: Optional[AnimalRefresh],
) -> str:
    """What the agent may conclude, said plainly, so it never turns missing or old
    data into "you have no such animal"."""
    if not tags:
        return "This farmer has no animal tags registered with Amul."
    if not _filters_given(filters):
        return (
            "Pass name, tag_last_digits, animal_type, breed, milking_stage or "
            "pregnancy_stage to get one animal's full record."
        )
    if matches:
        if refresh is not None:
            return "Matches found so far; the refresh is still fetching the rest of the herd."
        return "Matching records below."
    if refresh is not None:
        return (
            "No match yet, but the refresh is still fetching this farmer's animals. Do not say "
            "the animal is not registered. Call find_animals again with the same filters, or ask "
            "the farmer for the last four digits of the animal's tag."
        )
    if expired:
        return (
            "No match, but the records are older than their refresh interval and could not be "
            "refreshed now. Do not say the animal is not registered; ask the farmer for the last "
            "four digits of its tag."
        )
    missing = len([tag for tag in tags if tag not in records])
    caveat = (
        f" {missing} of {len(tags)} tags have no animal record in the Amul data, and the animal "
        "may be one of those."
        if missing
        else ""
    )
    if filters.get("name"):
        if named == 0:
            return (
                "None of this farmer's animals has a name in the Amul records, so a name cannot "
                f"identify the animal.{caveat} This does not mean the animal is not registered. Ask "
                "for the last four digits of its tag, or its type, breed or stage, and look again."
            )
        return (
            "No animal has that name in the Amul records; the names that are set are in "
            f"named_animals (check for another spelling or script).{caveat} Most animals have no "
            "name, so ask for the last four digits of the tag, or the type, breed or stage."
        )
    if missing:
        return (
            f"No match among the records.{caveat} Do not say the animal is not registered; ask "
            "the farmer for the last four digits of its tag."
        )
    return "No animal matches these filters in the farmer's current Amul records."


async def _find(
    ctx: RunContext[FarmerContext],
    *,
    spoken: bool,
    filters: dict,
    refresh: bool,
) -> str:
    mobile = (getattr(ctx.deps, "mobile", None) or "").strip()
    if not mobile:
        return "The farmer's animal records are not available in this session."
    envelope = await get_or_fetch_farmer_data(mobile)
    if envelope is None or envelope.lookupStatus != "found" or not envelope.farmers:
        return "The farmer's animal records are not available right now."

    has_tags = any(_record_tags(farmer) for farmer in envelope.farmers)
    age = animals_age_seconds(envelope)
    force = refresh and (age is None or age >= FARMER_ANIMAL_FORCE_MIN_AGE_SECONDS)

    # Join a refresh that is running, or start one when the records are past
    # their interval or the farmer says something changed; then wait a little for
    # it, answering as soon as it finishes or a requested animal shows up.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_SECONDS
    may_start, refreshed, started_now = has_tags, False, False
    while True:
        progress = await read_animal_refresh(mobile)
        if progress is None:
            if refreshed:
                envelope = await get_or_fetch_farmer_data(mobile) or envelope
            if may_start and (force or animals_expired(envelope)):
                may_start = False
                started_now = await start_animal_refresh(mobile, force=force)
                refreshed = refreshed or started_now
                if started_now:
                    continue
            break
        may_start, refreshed = False, True
        if _filters_given(filters):
            _, records = _merged_animals(envelope, progress)
            if _matching(list(records.values()), filters):
                break
        if loop.time() >= deadline:
            break
        await asyncio.sleep(POLL_SECONDS)

    tags, records = _merged_animals(envelope, progress)
    animals = [records[tag] for tag in tags if tag in records]
    named = [animal for animal in animals if animal.animalName]
    matches = _matching(animals, filters) if _filters_given(filters) else []
    age = animals_age_seconds(envelope)
    expired = progress is None and has_tags and animals_expired(envelope)

    result: dict[str, Any] = {
        "herd": {
            "tags": len(tags),
            "records_loaded": len(animals),
            "tags_without_record": len([tag for tag in tags if tag not in records]),
            "records_age_hours": round(age / 3600, 1) if age is not None else None,
            "records_current": bool(tags) and progress is None and not expired,
            "names_set": len(named),
            "refresh": _refresh_line(progress, refreshed),
        },
        "counts": _counts(animals),
        "named_animals": [
            {"tag": _tag_label(animal.tagNumber or "", spoken), "name": animal.animalName, "type": animal.animalType}
            for animal in named
        ],
    }
    if _filters_given(filters):
        result["matches"] = [_animal_out(animal, spoken) for animal in matches[:MAX_MATCHES]]
        result["matches_total"] = len(matches)
    result["note"] = _note(
        filters=filters,
        matches=matches,
        tags=tags,
        records=records,
        named=len(named),
        expired=expired,
        refresh=progress,
    )
    logger.info(
        "find_animals tags=%s loaded=%s named=%s matches=%s refresh=%s started=%s",
        len(tags), len(animals), len(named), len(matches), progress is not None, started_now,
    )
    return json.dumps(result, ensure_ascii=False, default=str)


async def find_animals(
    ctx: RunContext[FarmerContext],
    name: Optional[str] = None,
    tag_last_digits: Optional[str] = None,
    animal_type: Optional[str] = None,
    breed: Optional[str] = None,
    milking_stage: Optional[str] = None,
    pregnancy_stage: Optional[str] = None,
    refresh: bool = False,
) -> str:
    """Look up the signed-in farmer's own animals in the Amul records, and say how complete and current those records are.

    Call this whenever the farmer asks about one of their own animals: by its name, by
    its tag, or by a description ("my pregnant buffalo", "the Gir cow"). It returns
    how many of the farmer's tags have a record, how old the records are, how many
    animals have a name (few do), whether a refresh is running and how far it has
    got, the herd's counts by type, breed and stage, every named animal, and the
    full records that match the filters. Read its note before answering: never tell
    the farmer an animal is not registered unless the note says the records are
    current and complete.

    It refreshes the records itself when they are old. If a refresh is still running
    and nothing matched yet, call it again with the same filters.

    Args:
        name: The name the farmer calls the animal, as they said it (e.g. "Lakshmi").
        tag_last_digits: The last digits of the animal's tag number, digits only (e.g. "1234").
        animal_type: cow or buffalo.
        breed: Breed in English (e.g. "gir", "murrah", "jersey", "hf cb").
        milking_stage: e.g. "milking" or "dry".
        pregnancy_stage: e.g. "pregnant" or "non pregnant".
        refresh: True when the farmer says an animal's details changed recently (a new name, a new tag) or that the records are wrong: fetches every animal again now.
    """
    return await _find(
        ctx,
        spoken=False,
        filters={
            "name": name,
            "tag_last_digits": tag_last_digits,
            "animal_type": animal_type,
            "breed": breed,
            "milking_stage": milking_stage,
            "pregnancy_stage": pregnancy_stage,
        },
        refresh=refresh,
    )


async def find_animals_spoken(
    ctx: RunContext[FarmerContext],
    name: Optional[str] = None,
    tag_last_digits: Optional[str] = None,
    animal_type: Optional[str] = None,
    breed: Optional[str] = None,
    milking_stage: Optional[str] = None,
    pregnancy_stage: Optional[str] = None,
    refresh: bool = False,
) -> str:
    """Voice's find_animals: tags come back as their last four digits in words, the
    way the call's context names them, so a tag is never read out whole."""
    return await _find(
        ctx,
        spoken=True,
        filters={
            "name": name,
            "tag_last_digits": tag_last_digits,
            "animal_type": animal_type,
            "breed": breed,
            "milking_stage": milking_stage,
            "pregnancy_stage": pregnancy_stage,
        },
        refresh=refresh,
    )


# The model reads the chat docstring for both surfaces.
find_animals_spoken.__doc__ = find_animals.__doc__


async def prepare_find_animals(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> Optional[ToolDefinition]:
    """Only for a turn whose farmer record was resolved for the signed-in mobile:
    the lookup is keyed by that mobile alone."""
    if ctx.deps is not None and ctx.deps.has_farmer_profile():
        return tool_def
    return None
