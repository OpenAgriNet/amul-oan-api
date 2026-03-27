from typing import Any
from app.models.animal import AnimalModel
import asyncio
import json

from agents.tools.animal import get_animal_data_by_tag
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_animal_backends import normalize_phone
from app.models.farmer import FarmerModel


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, str):
        return value.capitalize()
    return str(value)


def _add_field(lines: list[str], label: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    lines.append(f"- **{label.capitalize()}:** {_format_value(value)}")
    return True


def _append_section(lines: list[str], title: str, fields: list[tuple[str, Any]]) -> None:
    section_lines: list[str] = []
    for label, value in fields:
        _add_field(section_lines, label, value)
    if not section_lines:
        return
    lines.append("")
    lines.append(title)
    lines.extend(section_lines)


def _append_farmer_markdown(lines: list[str], farmer: FarmerModel, index: int) -> None:
    lines.append("")
    lines.append(f"## Farmer {index}")
    profile_fields = [
        ("Farmer name", farmer.farmer_name),
        ("Mobile number", farmer.mobile_number),
        ("Farmer code", farmer.farmer_code),
        ("Society name", farmer.society_name),
        ("Society code", farmer.society_code),
        ("Union name", farmer.union_name),
        ("Union code", farmer.union_code),
        ("Village", farmer.village),
        ("Sub-district", farmer.sub_district),
        ("District", farmer.district),
        ("State", farmer.state),
    ]
    herd_fields = [
        ("Total animals", farmer.total_animals),
        ("Total cows", farmer.total_cow),
        ("Total buffalo", farmer.total_buffalo),
        ("Total milking animals", farmer.total_milking_animals),
        ("Non-pregnant milking animals", farmer.non_pregnant_milking_animals),
        ("Pregnant milking animals", farmer.pregnant_milking_animals),
    ]
    milk_fields = [
        ("Average cow milk per day", farmer.avg_milk_per_day_cow),
        ("Average buffalo milk per day", farmer.avg_milk_per_day_buffalo),
        ("Cow SNF", farmer.cow_snf),
        ("Cow fat", farmer.cow_fat),
        ("Buffalo SNF", farmer.buff_snf),
        ("Buffalo fat", farmer.buff_fat),
    ]
    for label, value in profile_fields:
        _add_field(lines, label, value)
    _append_section(lines, "### Herd summary", herd_fields)
    _append_section(lines, "### Milk metrics", milk_fields)


def _append_animal_markdown(lines: list[str], tag: str, animal: AnimalModel | None) -> None:
    lines.append("")
    lines.append(f"### Animal {tag}")
    if animal is None:
        lines.append("- No animal data found for this tag.")
        return

    animal_fields = [
        ("Tag number", animal.tag_number),
        ("Animal type", animal.animal_type),
        ("Animal name", animal.animal_name),
        ("Breed", animal.breed),
        ("Milking stage", animal.milking_stage),
        ("Pregnancy stage", animal.pregnancy_stage),
        ("Date of birth", animal.date_of_birth),
        ("Lactation number", animal.lactation_no),
        (
            "Last breeding activity",
            json.dumps(animal.last_breeding_activity, ensure_ascii=False)
            if animal.last_breeding_activity is not None
            else None,
        ),
        (
            "Last health activity",
            json.dumps(animal.last_health_activity, ensure_ascii=False)
            if animal.last_health_activity is not None
            else None,
        ),
    ]
    for label, value in animal_fields:
        _add_field(lines, label, value)


async def get_farmer_full_data_by_mobile(mobile_number: str) -> str:
    farmers = await get_farmer_data_by_mobile(mobile_number)
    mobile = normalize_phone(mobile_number) or mobile_number

    if farmers is None:
        return (
            "# Farmer Context\n\n"
            f"No farmer information found for mobile number `{mobile}`."
        )

    lines = [
        "# Farmer Context",
        "",
        "This context is built from farmer records fetched by mobile number and animal records fetched by each farmer tag number.",
        "",
        f"- **Requested mobile number:** `{mobile}`",
        f"- **Matched farmer records:** {len(farmers)}",
    ]

    for index, farmer in enumerate(farmers, start=1):
        _append_farmer_markdown(lines, farmer, index)

        tags = farmer.animal_tags or []
        lines.append("")
        lines.append("### Animal tags")
        if not tags:
            lines.append("- No animal tags found for this farmer.")
            continue

        lines.append(f"- **Animal tags:** {', '.join(tags)}")
        animals = await asyncio.gather(
            *(get_animal_data_by_tag(tag) for tag in tags)
        )
        for tag, animal in zip(tags, animals):
            _append_animal_markdown(lines, tag, animal)

    return "\n".join(lines)
