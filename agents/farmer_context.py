import asyncio
import json
import os
from types import CoroutineType
from typing import Any

from agents.tools.animal import get_animal_data_by_tag
from agents.tools.cvcc import get_cvcc_health_data_by_tag
from agents.tools.farmer import get_farmer_data_by_mobile
from agents.tools.farmer_animal_backends import (
    GetAITechniciansBySocietyQueryParams,
    fetch_banas_operated_visit,
    get_ai_technicians_by_society_cached,
    get_ai_technicians_by_society_refresh,
    merge_farmer_data,
    normalize_phone,
)
from agents.services.beckn_amul import (
    fetch_authenticated_farmers,
    fetch_animal_profile,
    fetch_banas_visits,
    fetch_cvcc_health,
    search_ai_technicians,
)
from app.config import settings
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
from app.models.farmer_transport import FarmerRecord
from app.models.union import (
    UNION_BANNED_MESSAGE,
    UnionName,
    is_ai_call_banned_union,
    resolve_supported_unions,
)
from app.services.scheme_ingestion import (
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from app.config import settings
from helpers.utils import get_logger, is_from_union


logger = get_logger(__name__)
SUPPORTED_SCHEME_CONTEXT_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
    UnionName.SUMUL.value,
    UnionName.SURENDRANAGAR.value,
}


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


def _collect_farmer_unions(farmers: list[FarmerModel]) -> list[str]:
    seen: set[str] = set()
    unions: list[str] = []
    for farmer in farmers:
        normalized_union = (farmer.union_name or "").strip().lower()
        if not normalized_union or normalized_union in seen:
            continue
        seen.add(normalized_union)
        unions.append(normalized_union)
    return unions


def _collect_farmer_location(farmers: list[FarmerModel]) -> dict[str, str]:
    """Pick the structured location to expose to tools: {district, village, state}.

    First record that carries a district wins, and village/state are taken from
    that SAME record — mixing a district from one account with a village from
    another would invent a place. A mobile with several accounts (a cow account
    and a buffalo account, say) is normal and they share a location in practice.

    These fields are already rendered into the prompt markdown; this lifts them
    into structured deps so the mandi/weather tools can key off them instead of
    hardcoding Anand.
    """
    for farmer in farmers:
        district = (farmer.district or "").strip()
        if not district:
            continue
        return {
            "district": district,
            "village": (farmer.village or "").strip(),
            "state": (farmer.state or "").strip(),
        }
    return {}


async def _append_union_scheme_summary_markdown(lines: list[str], farmer_unions: list[str]) -> None:
    scheme_unions = resolve_supported_unions(farmer_unions, SUPPORTED_SCHEME_CONTEXT_UNIONS)
    if not scheme_unions:
        return

    lines.append("")
    lines.append("## Union schemes available")
    lines.append("- The following scheme titles are available from the union scheme cache. Use these titles and links for scheme-related questions. Retrieve full cached scheme details when the user asks about a specific scheme.")

    for union_name in scheme_unions:
        try:
            records = await get_cached_scheme_records_for_union(union_name)
        except SchemeDependencyError:
            logger.warning("Union scheme summary skipped because Redis dependency is unavailable union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache dependency is unavailable.")
            continue
        except SchemeCacheError:
            logger.warning("Union scheme summary skipped because scheme cache could not be read union=%s", union_name)
            lines.append(f"- **{union_name.title()}**: Scheme cache could not be read.")
            continue
        except Exception as exc:
            logger.warning("Union scheme summary skipped because of unexpected error union=%s error=%s", union_name, exc)
            lines.append(f"- **{union_name.title()}**: Scheme list is temporarily unavailable.")
            continue

        if not records:
            lines.append(f"- **{union_name.title()}**: No cached scheme list is available yet.")
            continue

        lines.append(f"- **{union_name.title()} union schemes:**")
        seen_links: set[tuple[str, str]] = set()
        for record in records:
            title = record.get("scheme_title")
            link = record.get("scheme_url")
            if not title or not link:
                continue
            dedupe_key = (str(title).casefold(), str(link))
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            lines.append(f"  - {title}: {link}")


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


async def _get_ai_technicians_for_farmer(
    farmer: FarmerModel,
    *,
    force_refresh: bool = False,
) -> tuple[list[str] | None, str | None]:
    if not farmer.union_code or not farmer.society_code:
        return None, "AI technician lookup skipped because union code or society code is missing."

    # In callback mode this is a directed search/on_search transaction to the
    # single Amul BPP. The union and society values came from the authenticated
    # farmer callback above, never from a model tool argument.
    if settings.enable_network and settings.beckn_callback_transactions_enabled:
        try:
            technicians = await search_ai_technicians(
                union_code=farmer.union_code,
                society_code=farmer.society_code,
            )
        except Exception as exc:
            logger.warning("AI technician Beckn lookup failed: %s", exc)
            technicians = None
    else:
        query = GetAITechniciansBySocietyQueryParams(
            unionCode=farmer.union_code,
            societyCode=farmer.society_code,
        )
        token = os.getenv("PASHUGPT_TOKEN")

        # force_refresh: always hit upstream. Otherwise use cache-first and
        # trust successful cached responses, including an empty list.
        if force_refresh:
            technicians = await get_ai_technicians_by_society_refresh(query, token)
        else:
            technicians = await get_ai_technicians_by_society_cached(query, token)
    if technicians is None:
        return None, "AI technician details could not be fetched right now."

    if not technicians:
        return [], None

    unique_technicians: dict[str, str] = {}
    for technician in technicians:
        key = technician.userId or f"{technician.fullName}|{technician.mobileNumber}"
        if key in unique_technicians:
            continue
        unique_technicians[key] = (
            f"- **Name:** {technician.fullName} | "
            f"**Mobile number:** {technician.mobileNumber} | "
            f"**user_id:** {technician.userId}"
        )

    return list(unique_technicians.values()), None


async def _append_ai_technicians_markdown(lines: list[str], farmer: FarmerModel) -> None:
    lines.append("")
    if is_ai_call_banned_union(farmer.union_name):
        logger.info(
            "Skipping AI technician lookup; union is banned from AI-call booking union=%s",
            farmer.union_name,
        )
        lines.append("### AI call booking")
        lines.append("- AI call booking is not allowed for this union.")
        lines.append(f"- Tell the farmer: `{UNION_BANNED_MESSAGE}`")
        lines.append("- Do not ask which technician they want. Do not call `create_ai_call`.")
        return

    lines.append("### Available AI technicians")

    technician_lines, error_message = await _get_ai_technicians_for_farmer(farmer)
    if error_message:
        lines.append(f"- {error_message}")
        return

    if technician_lines == []:
        lines.append("- No AI technicians were found for this society.")
        return

    if not technician_lines:
        lines.append("- AI technician details are unavailable.")
        return

    lines.append(
        "- Use these details when the user wants to book an AI call. Show only name and mobile number to the user, but use the mapped `user_id` when calling `create_ai_call`."
    )
    lines.extend(technician_lines)


def _technician_group_for_farmer(farmer: FarmerModel, ai_groups: list[dict]) -> dict | None:
    # Prefer exact farmer-code mapping if present; only fall back to
    # society/union when no exact match exists across all groups.
    def _same_union_society(group: dict) -> bool:
        return (
            farmer.society_code
            and farmer.union_code
            and str(farmer.society_code) == str(group.get("societyCode"))
            and str(farmer.union_code) == str(group.get("unionCode"))
        )

    for group in ai_groups:
        group_code = group.get("farmerCode")
        if (
            farmer.farmer_code
            and group_code
            and str(farmer.farmer_code) == str(group_code)
            and _same_union_society(group)
        ):
            return group

    for group in ai_groups:
        if _same_union_society(group):
            return group
    return None


def _format_cached_technician_lines(technicians: list[dict]) -> list[str]:
    unique_technicians: dict[str, str] = {}
    for technician in technicians:
        user_id = technician.get("userId")
        full_name = technician.get("fullName")
        mobile_number = technician.get("mobileNumber")
        key = user_id or f"{full_name}|{mobile_number}"
        if key in unique_technicians:
            continue
        unique_technicians[key] = (
            f"- **Name:** {full_name} | "
            f"**Mobile number:** {mobile_number} | "
            f"**user_id:** {user_id}"
        )
    return list(unique_technicians.values())


async def _append_ai_technicians_markdown_with_cache(
    lines: list[str],
    farmer: FarmerModel,
    ai_groups: list[dict] | None,
) -> None:
    lines.append("")
    if is_ai_call_banned_union(farmer.union_name):
        logger.info(
            "Skipping AI technician lookup; union is banned from AI-call booking union=%s",
            farmer.union_name,
        )
        lines.append("### AI call booking")
        lines.append("- AI call booking is not allowed for this union.")
        lines.append(f"- Tell the farmer: `{UNION_BANNED_MESSAGE}`")
        lines.append("- Do not ask which technician they want. Do not call `create_ai_call`.")
        return

    lines.append("### Available AI technicians")
    force_refresh = False

    if ai_groups:
        group = _technician_group_for_farmer(farmer, ai_groups)
        if group is not None:
            cached_failed = bool(group.get("techniciansLookupFailed"))
            cached_technicians = group.get("technicians")
            if not cached_failed and cached_technicians:
                technician_lines = _format_cached_technician_lines(cached_technicians)
                if technician_lines:
                    lines.append(
                        "- Use these details when the user wants to book an AI call. Show only name and mobile number to the user, but use the mapped `user_id` when calling `create_ai_call`."
                    )
                    lines.extend(technician_lines)
                    return

            # Retry live only for previously failed/unavailable cached lookups.
            # A cached empty list is a successful response and should be trusted.
            if cached_failed or cached_technicians is None:
                force_refresh = True
                logger.info(
                    "Cached AI technician lookup was unavailable; retrying live lookup union=%s society=%s",
                    farmer.union_code,
                    farmer.society_code,
                )

    technician_lines, error_message = await _get_ai_technicians_for_farmer(
        farmer,
        force_refresh=force_refresh,
    )
    if error_message:
        lines.append(f"- {error_message}")
        return

    if technician_lines == []:
        lines.append("- No AI technicians were found for this society.")
        return

    if not technician_lines:
        lines.append("- AI technician details are unavailable.")
        return

    lines.append(
        "- Use these details when the user wants to book an AI call. Show only name and mobile number to the user, but use the mapped `user_id` when calling `create_ai_call`."
    )
    lines.extend(technician_lines)


def _farmer_records_to_models(records: list[FarmerRecord]) -> list[FarmerModel]:
    farmers: list[FarmerModel] = []
    for record in records:
        try:
            farmers.append(
                FarmerModel.model_validate(record.model_dump(), extra="ignore", by_alias=True)
            )
        except Exception as exc:
            logger.warning("Skipping invalid farmer record during Layer 2 context build: %s", exc)
    # Preserve legacy chat semantics: consolidate duplicates with the same
    # merge strategy used by get_farmer_data_by_mobile().
    return merge_farmer_data(farmers)


def _not_found_context(mobile: str) -> tuple[str, list[str], dict[str, str]]:
    return (
        "# Farmer Context\n\n"
        f"No farmer information found for mobile number `{mobile}`.",
        [],
        {},
    )


async def _build_farmer_context_bundle_from_farmers(
    mobile: str,
    farmers: list[FarmerModel],
    *,
    ai_groups: list[dict] | None = None,
) -> tuple[str, list[str], dict[str, str]]:
    farmer_unions = _collect_farmer_unions(farmers)
    farmer_location = _collect_farmer_location(farmers)

    lines = [
        "# Farmer Context",
        "",
        "This context is built from farmer records fetched by mobile number and animal records fetched by each farmer tag number.",
        "",
        f"- **Requested mobile number:** `{mobile}`",
        f"- **Matched farmer records:** {len(farmers)}",
    ]
    await _append_union_scheme_summary_markdown(lines, farmer_unions)

    for index, farmer in enumerate(farmers, start=1):
        _append_farmer_markdown(lines, farmer, index)
        if ai_groups is not None:
            await _append_ai_technicians_markdown_with_cache(lines, farmer, ai_groups)
        else:
            await _append_ai_technicians_markdown(lines, farmer)

        tags = farmer.animal_tags or []
        include_banas_visit = is_from_union([farmer], UnionName.BANAS)
        include_cvcc_health = is_from_union([farmer], UnionName.KAIRA)
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
                    farmer.union_code,
                )
                for tag in tags
            )
        )
        for tag, animal, banas_visits, cvcc_health in animal_contexts:
            _append_animal_markdown(lines, tag, animal, banas_visits, cvcc_health)

    return "\n".join(lines), farmer_unions, farmer_location


async def _get_farmer_context_bundle_legacy(
    mobile_number: str,
) -> tuple[str, list[str], dict[str, str]]:
    farmers = await get_farmer_data_by_mobile(mobile_number)
    mobile = normalize_phone(mobile_number) or mobile_number

    if farmers is None:
        return _not_found_context(mobile)

    return await _build_farmer_context_bundle_from_farmers(mobile, farmers)


async def _get_farmer_context_bundle_layer2(
    mobile_number: str,
) -> tuple[str, list[str], dict[str, str]] | None:
    from agents.services.farmer_cache import get_or_fetch_farmer_data

    mobile = normalize_phone(mobile_number) or mobile_number
    envelope = await get_or_fetch_farmer_data(mobile)

    if envelope is None:
        return None

    if envelope.lookupStatus == "unknown":
        # Layer 2 could not establish an authoritative result; let caller
        # decide fallback behavior.
        return None

    if envelope.lookupStatus == "not_found":
        return _not_found_context(mobile)

    if not envelope.farmers:
        return None

    farmers = _farmer_records_to_models(envelope.farmers)
    if not farmers:
        return None

    return await _build_farmer_context_bundle_from_farmers(
        mobile,
        farmers,
        ai_groups=envelope.aiTechnicians or [],
    )


async def _get_farmer_context_bundle_beckn(
    mobile_number: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Build farmer context via Beckn callback transactions (network mode)."""
    mobile = normalize_phone(mobile_number) or mobile_number
    farmers = await fetch_authenticated_farmers(mobile)

    if farmers is None:
        return _not_found_context(mobile)

    farmer_unions = _collect_farmer_unions(farmers)
    farmer_location = _collect_farmer_location(farmers)

    lines = [
        "# Farmer Context",
        "",
        "This context is built from farmer records fetched by mobile number and animal records fetched by each farmer tag number.",
        "",
        f"- **Requested mobile number:** `{mobile}`",
        f"- **Matched farmer records:** {len(farmers)}",
    ]
    # Union scheme discovery is already an on_search tool in network mode.
    # Do not bypass the BPP by preloading the same Redis catalog directly into
    # context; the agent fetches it only when the farmer actually asks.

    for index, farmer in enumerate(farmers, start=1):
        _append_farmer_markdown(lines, farmer, index)
        await _append_ai_technicians_markdown(lines, farmer)

        tags = farmer.animal_tags or []
        include_banas_visit = is_from_union([farmer], UnionName.BANAS)
        include_cvcc_health = is_from_union([farmer], UnionName.KAIRA)
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
                    farmer.union_code,
                )
                for tag in tags
            )
        )
        for tag, animal, banas_visits, cvcc_health in animal_contexts:
            _append_animal_markdown(lines, tag, animal, banas_visits, cvcc_health)

    return "\n".join(lines), farmer_unions, farmer_location

async def get_farmer_context_bundle_by_mobile(
    mobile_number: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Return (prompt markdown, union names, structured location).

    The third element is {district, village, state} (possibly empty) and exists
    so tools can read the farmer's location. It is deliberately NOT parsed back
    out of the markdown: the markdown is a prompt, not an API.
    """
    if settings.enable_network and settings.beckn_callback_transactions_enabled:
        return await _get_farmer_context_bundle_beckn(mobile_number)

    if not settings.farmer_layer2_chat_context_enabled:
        return await _get_farmer_context_bundle_legacy(mobile_number)

    try:
        bundle = await _get_farmer_context_bundle_layer2(mobile_number)
    except Exception as exc:
        logger.warning(
            "Farmer context Layer 2 build failed for mobile=%s: %s",
            mobile_number,
            exc,
        )
        bundle = None

    if bundle is not None:
        return bundle

    if settings.farmer_layer2_fallback_to_legacy_enabled:
        logger.info(
            "Farmer cache read: fallback_legacy mobile=%s reason=layer2_unusable",
            normalize_phone(mobile_number) or mobile_number,
        )
        return await _get_farmer_context_bundle_legacy(mobile_number)

    mobile = normalize_phone(mobile_number) or mobile_number
    return _not_found_context(mobile)


async def get_farmer_full_data_by_mobile(mobile_number: str) -> str:
    farmer_context, _, _ = await get_farmer_context_bundle_by_mobile(mobile_number)
    return farmer_context


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
    union_code: str | None,
) -> tuple[
    str,
    AnimalModel | None,
    list[BanasOperatedVisitModel] | None,
    CvccHealthResponseModel | None,
]:
    use_amul_bpp = settings.enable_network and settings.beckn_callback_transactions_enabled
    if use_amul_bpp:
        tasks: list[CoroutineType[Any, Any, AnimalModel | list[BanasOperatedVisitModel] | CvccHealthResponseModel | None]] = [
            fetch_animal_profile(
                tag,
                union_name=union_name,
                union_code=union_code,
            )
        ]
        if include_banas_visit and union_code:
            tasks.append(fetch_banas_visits(tag, union_code=union_code))
        else:
            include_banas_visit = False
        if include_cvcc_health and union_code:
            tasks.append(fetch_cvcc_health(tag, union_code=union_code))
        else:
            include_cvcc_health = False
    else:
        tasks = [get_animal_data_by_tag(tag)]
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
