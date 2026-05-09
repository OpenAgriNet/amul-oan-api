"""
Rich farmer context builder for the voice agent.
Fetches farmer records, animal details, and formats as markdown for the system prompt.
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from voice.agents.tools.farmer_animal_backends import (
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    fetch_animal_amulpashudhan,
    fetch_animal_herdman,
    merge_farmer_records,
    merge_animal_data,
    normalize_phone,
)
from voice.helpers.gujarati_numbers import mask_tag_identifier
from voice.helpers.utils import get_logger

logger = get_logger(__name__)

# Unions that have specific additional APIs
BANAS_UNIONS = {"banas"}
KAIRA_UNIONS = {"kaira"}
MEHSANA_UNIONS = {"mehsana", "dudhsagar"}


def _get_union_name(farmer: Dict[str, Any]) -> str:
    """Extract and normalize union name from farmer record."""
    name = farmer.get("unionName") or farmer.get("Union Name") or ""
    return name.strip().lower()


def _get_tags(farmer: Dict[str, Any]) -> List[str]:
    """Extract animal tags from farmer record."""
    tag_str = farmer.get("tagNo") or farmer.get("tagNumbers") or ""
    if not tag_str:
        return []
    return [t.strip() for t in tag_str.split(",") if t.strip()]


def _add_field(lines: list, label: str, value: Any) -> None:
    if value is None or value == "" or value == 0:
        return
    if isinstance(value, float):
        lines.append(f"- **{label}:** {value:g}")
    elif isinstance(value, int):
        lines.append(f"- **{label}:** {value}")
    else:
        lines.append(f"- **{label}:** {value}")


def _format_farmer(lines: list, farmer: Dict[str, Any], index: int) -> None:
    lines.append("")
    lines.append(f"## Farmer {index}")
    _add_field(lines, "Farmer name", farmer.get("farmerName"))
    _add_field(lines, "Mobile number", farmer.get("mobileNumber") or farmer.get("Mobile Number"))
    _add_field(lines, "Farmer code", farmer.get("farmerCode"))
    _add_field(lines, "Society name", farmer.get("societyName") or farmer.get("Society Name"))
    _add_field(lines, "Society code", farmer.get("societyCode"))
    _add_field(lines, "Union name", farmer.get("unionName") or farmer.get("Union Name"))
    _add_field(lines, "Union code", farmer.get("unionCode"))
    _add_field(lines, "Village", farmer.get("village"))
    _add_field(lines, "District", farmer.get("district"))
    _add_field(lines, "State", farmer.get("state"))

    # Herd summary
    total = farmer.get("totalAnimals") or farmer.get("Total Animal")
    if total:
        lines.append("")
        lines.append("### Herd summary")
        _add_field(lines, "Total animals", total)
        _add_field(lines, "Cows", farmer.get("cow") or farmer.get("Cow"))
        _add_field(lines, "Buffalo", farmer.get("buffalo") or farmer.get("Buffalo"))
        _add_field(lines, "Milking animals", farmer.get("totalMilkingAnimals") or farmer.get("Milking Animal"))

    # Milk metrics
    avg_cow = farmer.get("avgMilkPerDayCow")
    avg_buff = farmer.get("avgMilkPerDayBuff")
    if (avg_cow and avg_cow > 0) or (avg_buff and avg_buff > 0):
        lines.append("")
        lines.append("### Milk metrics")
        _add_field(lines, "Avg cow milk/day", avg_cow)
        _add_field(lines, "Avg buffalo milk/day", avg_buff)
        _add_field(lines, "Cow fat", farmer.get("cowFat"))
        _add_field(lines, "Cow SNF", farmer.get("cowSnf"))
        _add_field(lines, "Buffalo fat", farmer.get("buffFat"))
        _add_field(lines, "Buffalo SNF", farmer.get("buffSnf"))


def _format_animal(lines: list, tag: str, animal: Optional[Dict[str, Any]]) -> None:
    masked_tag = mask_tag_identifier(tag)
    lines.append("")
    lines.append(f"### Animal {masked_tag or tag}")
    if not animal:
        lines.append("- No animal data found for this tag.")
        return
    _add_field(lines, "Tag number", mask_tag_identifier(animal.get("tagNumber") or ""))
    _add_field(lines, "Animal type", animal.get("animalType"))
    _add_field(lines, "Breed", animal.get("breed"))
    _add_field(lines, "Milking stage", animal.get("milkingStage"))
    _add_field(lines, "Pregnancy stage", animal.get("pregnancyStage"))
    _add_field(lines, "Date of birth", animal.get("dateOfBirth"))
    _add_field(lines, "Lactation number", animal.get("lactationNo"))
    breeding = animal.get("lastBreedingActivity")
    if breeding:
        _add_field(lines, "Last breeding activity", json.dumps(breeding, ensure_ascii=False) if isinstance(breeding, (dict, list)) else breeding)
    health = animal.get("lastHealthActivity")
    if health:
        _add_field(lines, "Last health activity", json.dumps(health, ensure_ascii=False) if isinstance(health, (dict, list)) else health)


async def _fetch_animal_details(tag: str, token1: str, token3: str | None) -> tuple[str, Dict[str, Any] | None]:
    """Fetch animal details from both backends and merge."""
    primary = None
    fallback = None
    if token1:
        try:
            primary = await fetch_animal_amulpashudhan(tag, token1)
        except Exception:
            pass
    if token3:
        try:
            fallback = await fetch_animal_herdman(tag, token3)
        except Exception:
            pass
    merged = merge_animal_data(primary, fallback)
    return tag, merged if merged else None


async def get_farmer_full_context_string(mobile_number: str) -> str:
    """
    Build a rich markdown context string for the voice agent system prompt.
    Fetches farmer records and animal details, formats everything into structured markdown.
    """
    mobile = normalize_phone(mobile_number)
    if not mobile:
        return f"# Farmer Context\n\nNo farmer information found for mobile number `{mobile_number}`."

    token1 = os.getenv("PASHUGPT_TOKEN")
    token3 = os.getenv("PASHUGPT_TOKEN_3")
    if not token1 and not token3:
        return "# Farmer Context\n\nFarmer data service is not configured."

    # Fetch farmer records
    records: List[Dict[str, Any]] = []
    if token1:
        try:
            data = await fetch_farmer_amulpashudhan(mobile, token1)
            if data:
                records = merge_farmer_records(records + data)
                logger.info(f"Voice farmer context: {len(data)} record(s) from amulpashudhan for {mobile}")
        except Exception as e:
            logger.warning(f"amulpashudhan farmer error for {mobile}: {e}")

    if token3:
        # Check if Mehsana union (Herdman data available)
        union_names = {_get_union_name(r) for r in records}
        if union_names & MEHSANA_UNIONS:
            try:
                data = await fetch_farmer_herdman(mobile, token3)
                if data:
                    records = merge_farmer_records(records + data)
                    logger.info(f"Voice farmer context: {len(data)} record(s) from herdman for {mobile}")
            except Exception as e:
                logger.warning(f"herdman farmer error for {mobile}: {e}")

    if not records:
        return f"# Farmer Context\n\nNo farmer information found for mobile number `{mobile}`."

    lines = [
        "# Farmer Context",
        "",
        f"- **Requested mobile number:** `{mobile}`",
        f"- **Matched farmer records:** {len(records)}",
    ]

    for index, farmer in enumerate(records, start=1):
        _format_farmer(lines, farmer, index)

        tags = _get_tags(farmer)
        masked_tags = []
        for tag in tags:
            masked = mask_tag_identifier(tag)
            if masked:
                masked_tags.append(masked)
        lines.append("")
        lines.append("### Animal tags")
        if not masked_tags:
            lines.append("- No animal tags found for this farmer.")
            continue

        lines.append(f"- **Animal tags:** {', '.join(masked_tags)}")

        # Fetch animal details in parallel
        animal_tasks = [_fetch_animal_details(tag, token1 or "", token3) for tag in tags]
        animal_results = await asyncio.gather(*animal_tasks)
        for tag, animal_data in animal_results:
            _format_animal(lines, tag, animal_data)

    return "\n".join(lines)
