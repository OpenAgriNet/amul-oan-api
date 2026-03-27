import asyncio
import json
from types import CoroutineType
from typing import Any

from agents.tools.animal import get_animal_data_by_tag
from agents.tools.cvcc import get_cvcc_health_data_by_tag
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_animal_backends import (
    fetch_banas_operated_visit,
    normalize_phone,
)
from app.models.animal import AnimalModel
from app.models.banas_visit import (
    BanasLabReportModel,
    BanasMedicineModel,
    BanasOperatedVisitModel,
)
from app.models.cvcc import (
    CvccDewormingModel,
    CvccHealthResponseModel,
    CvccTreatmentMedicineModel,
    CvccTreatmentModel,
    CvccVaccinationModel,
)
from app.models.farmer import FarmerModel
from app.models.union import UnionName
from helpers.utils import is_from_union


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _add_field(lines: list[str], label: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    lines.append(f"- **{label}:** {_format_value(value)}")
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

def _format_medicines(medicines: list[BanasMedicineModel] | None) -> str | None:
    if not medicines:
        return None
    parts = []
    for medicine in medicines:
        if medicine.medicine_name is None:
            continue
        detail = medicine.medicine_name
        if medicine.stock is not None and medicine.uom_doctor:
            detail = f"{detail} ({medicine.stock:g} {medicine.uom_doctor})"
        parts.append(detail)
    return "; ".join(parts) if parts else None


def _format_lab_reports(lab_reports: list[BanasLabReportModel] | None) -> str | None:
    if not lab_reports:
        return None
    parts = []
    for report in lab_reports:
        if report.sample_name is None and report.remarks is None:
            continue
        detail = report.sample_name or "lab report"
        if report.remarks:
            detail = f"{detail} ({report.remarks})"
        parts.append(detail)
    return "; ".join(parts) if parts else None


def _format_ailments(visit: BanasOperatedVisitModel) -> str | None:
    ailments = [
        ailment
        for ailment in [visit.ailment_1, visit.ailment_2, visit.ailment_3]
        if ailment and ailment != "-"
    ]
    return "; ".join(ailments) if ailments else None


def _format_cvcc_medicines(
    medicines: list[CvccTreatmentMedicineModel] | None,
) -> str | None:
    if not medicines:
        return None
    parts = []
    for medicine in medicines:
        if medicine.medicine_name is None:
            continue
        detail = medicine.medicine_name
        if medicine.medicine_dose and medicine.medicine_route:
            detail = f"{detail} ({medicine.medicine_dose}, {medicine.medicine_route})"
        elif medicine.medicine_dose:
            detail = f"{detail} ({medicine.medicine_dose})"
        parts.append(detail)
    return "; ".join(parts) if parts else None


def _format_cvcc_treatments(
    treatments: list[CvccTreatmentModel] | None,
) -> str | None:
    if not treatments:
        return None
    parts = []
    for treatment in treatments:
        detail_parts = [
            part
            for part in [
                treatment.treatment_date,
                treatment.symptom,
                treatment.treatment,
            ]
            if part
        ]
        medicines = _format_cvcc_medicines(treatment.medicine)
        if medicines:
            detail_parts.append(f"medicines: {medicines}")
        if detail_parts:
            parts.append(" | ".join(detail_parts))
    return " || ".join(parts) if parts else None


def _format_cvcc_vaccinations(
    vaccinations: list[CvccVaccinationModel] | None,
) -> str | None:
    if not vaccinations:
        return None
    parts = []
    for vaccination in vaccinations:
        detail_parts = [
            part
            for part in [
                vaccination.vaccination_date,
                vaccination.vaccine_name,
                vaccination.vaccine_for_disease,
            ]
            if part
        ]
        if detail_parts:
            parts.append(" | ".join(detail_parts))
    return " || ".join(parts) if parts else None


def _format_cvcc_deworming(
    deworming_records: list[CvccDewormingModel] | None,
) -> str | None:
    if not deworming_records:
        return None
    parts = []
    for deworming in deworming_records:
        detail_parts = [
            part
            for part in [
                deworming.deworming_date,
                deworming.dewormer_name,
                deworming.dewormer_dose,
            ]
            if part
        ]
        if detail_parts:
            parts.append(" | ".join(detail_parts))
    return " || ".join(parts) if parts else None


def _append_banas_visit_markdown(
    lines: list[str], visits: list[BanasOperatedVisitModel] | None
) -> None:
    if not visits:
        return

    lines.append("")
    lines.append("#### Operated visits")
    for index, visit in enumerate(visits, start=1):
        lines.append("")
        lines.append(f"##### Visit {index}")
        visit_fields = [
            ("Visit code", visit.visit_code),
            ("Visit status", visit.visit_status),
            ("Visit note date", visit.visit_note_date),
            ("Visit schedule date", visit.visit_schedule_date),
            ("Visit allocation date", visit.visit_allocation_date),
            ("Entry time", visit.entry_time),
            ("Visit response time", visit.visit_response_time),
            ("Disease", visit.disease_name or visit.disease),
            ("Disease group", visit.disease_group),
            ("Ailments", _format_ailments(visit)),
            ("Species", visit.species_name),
            ("Milk status", visit.milk_status),
            ("Primary doctor name", visit.primary_doctor_name),
            ("Doctor mobile", visit.doctor_mobile),
            ("Driver name", visit.driver_name),
            ("Payment mode", visit.payment_mode),
            ("Payment comment", visit.payment_comment),
            ("Vet centre name", visit.vet_centre_name),
            ("Prognosis details", visit.prognosis_details),
            ("Medicines", _format_medicines(visit.medicines)),
            ("Lab reports", _format_lab_reports(visit.lab_reports)),
            ("Report date", visit.report_date),
        ]
        for label, value in visit_fields:
            _add_field(lines, label, value)


def _append_cvcc_health_markdown(
    lines: list[str], cvcc_health: CvccHealthResponseModel | None
) -> None:
    if cvcc_health is None or cvcc_health.data is None:
        return

    data = cvcc_health.data
    lines.append("")
    lines.append("#### CVCC health details")
    cvcc_fields = [
        ("CVCC status", cvcc_health.msg),
        ("Tag", data.tag),
        ("Animal type", data.animal_type),
        ("Breed", data.breed),
        ("Milking stage", data.milking_stage),
        ("Pregnancy stage", data.pregnancy_stage),
        ("Lactation", data.lactation),
        ("Milk yield", data.milk_yield),
        ("Farmer mobile number", data.farmer_mobile_number),
        ("Farmer id", data.farmer_id),
        ("Collar belt", data.collar_belt),
        ("Treatments", _format_cvcc_treatments(data.treatment)),
        ("Vaccinations", _format_cvcc_vaccinations(data.vaccination)),
        ("Deworming", _format_cvcc_deworming(data.deworming)),
    ]
    for label, value in cvcc_fields:
        _add_field(lines, label, value)


def _append_animal_markdown(
    lines: list[str],
    tag: str,
    animal: AnimalModel | None,
    banas_visits: list[BanasOperatedVisitModel] | None = None,
    cvcc_health: CvccHealthResponseModel | None = None,
) -> None:
    lines.append("")
    lines.append(f"### Animal {tag}")
    if animal is None:
        lines.append("- No animal data found for this tag.")
    else:
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
    _append_banas_visit_markdown(lines, banas_visits)
    _append_cvcc_health_markdown(lines, cvcc_health)


async def _get_animal_context_bundle(
    tag: str,
    include_banas_visit: bool,
    include_cvcc_health: bool,
    union_name: str | None,
) -> tuple[
    str,
    AnimalModel | None,
    list[BanasOperatedVisitModel] | None,
    CvccHealthResponseModel | None,
]:
    tasks: list[CoroutineType[Any, Any, AnimalModel | list[BanasOperatedVisitModel] | CvccHealthResponseModel | None]] = [get_animal_data_by_tag(tag)]
    if include_banas_visit:
        tasks.append(fetch_banas_operated_visit(tag))
    if include_cvcc_health:
        tasks.append(get_cvcc_health_data_by_tag(tag, union_name=union_name))

    results = await asyncio.gather(*tasks)
    animal = results[0]
    result_index = 1
    banas_visits = None
    if include_banas_visit:
        banas_visits = results[result_index]
        result_index += 1
    cvcc_health = None
    if include_cvcc_health:
        cvcc_health = results[result_index]
    return tag, animal, banas_visits, cvcc_health  # ty: ignore


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
        include_banas_visit = is_from_union([farmer], UnionName.BANAS)
        include_cvcc_health = is_from_union([farmer], UnionName.SABARKAIRA)
        lines.append("")
        lines.append("### Animal tags")
        if not tags:
            lines.append("- No animal tags found for this farmer.")
            continue

        lines.append(f"- **Animal tags:** {', '.join(tags)}")
        animal_contexts = await asyncio.gather(
            *(
                _get_animal_context_bundle(
                    tag,
                    include_banas_visit,
                    include_cvcc_health,
                    farmer.union_name,
                )
                for tag in tags
            )
        )
        for tag, animal, banas_visits, cvcc_health in animal_contexts:
            _append_animal_markdown(lines, tag, animal, banas_visits, cvcc_health)

    return "\n".join(lines)
