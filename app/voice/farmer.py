"""What voice reads off the farmer record, and how it reads it, from
voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``): the farmer
block, accounts, location, union schemes and AI technicians the agent is given.
"""
import json
from typing import Optional

import regex

from agents.deps import FarmerAccount
from agents.tools.farmer_cache import technician_lookup_failed
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from agents.voice.models.ai_call import strip_ait_name_codes
from agents.voice.services.farmer_identity import (
    identity_state_for_envelope,
    unavailable_capability_lines,
)
from app.voice.models.union import (
    UNION_BANNED_MESSAGE,
    is_ai_call_banned_union,
    resolve_supported_unions,
)
from app.voice.scheme_ingestion import (
    SUPPORTED_SCHEME_UNIONS,
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from helpers.gujarati_numbers import mask_tag_identifier
from helpers.utils import get_logger

logger = get_logger(__name__)


def _is_signed_in_session(user_info: Optional[dict], user_id: str) -> bool:
    if user_id and user_id != "anonymous":
        return True
    return bool(user_info)


def _extract_farmer_tags(records: list[FarmerRecord]) -> list[str]:
    tags: list[str] = []
    for record in records:
        raw = record.tagNumbers or record.tagNo or ""
        if not raw:
            continue
        for tag in str(raw).split(","):
            cleaned = tag.strip()
            masked = mask_tag_identifier(cleaned)
            if masked and masked not in tags:
                tags.append(masked)
    return tags


def _render_breeding_value(value) -> str:
    """Compact, model-readable form of lastBreedingActivity (an object with the AI
    date + bull id on most records, a flat string on some)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _append_animal_records(lines: list[str], envelope: FarmerDataEnvelope) -> None:
    """Per-animal records (incl. last AI date + bull id) for breeding/AI-history
    questions. Tags are masked (last 4 digits) so TTS never reads a full tag aloud.
    Populated by the background cache refresh — absent on a brand-new caller's
    first turn, present from the next turn on."""
    blocks: list[str] = []
    for record in envelope.farmers:
        for animal in getattr(record, "animals", None) or []:
            data = animal.model_dump()
            masked = mask_tag_identifier(data.get("tagNumber") or "")
            if not masked:
                continue
            parts = [f"tag ending {masked}"]
            if data.get("animalType"):
                parts.append(f"type={data['animalType']}")
            breeding = data.get("lastBreedingActivity")
            if breeding:
                parts.append(f"last AI/breeding={_render_breeding_value(breeding)}")
            else:
                parts.append("no AI records available")
            blocks.append("- " + ", ".join(parts))
    if blocks:
        lines.append("")
        lines.append("### Per-animal AI / breeding history")
        lines.extend(blocks)


def _normalize_farmer_name(value) -> str:
    """Names carry stray dots and double spaces (`PATEL..`, `A  B`)."""
    # \w excludes Unicode Mn/Mc, so a plain [^\w\s] strips Gujarati matras and
    # collapses distinct names and villages (કડી and કડા both became કડ).
    cleaned = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", "", str(value or ""))
    return regex.sub(r"\s+", " ", cleaned).strip().casefold()


def _build_compact_farmer_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    """Farmer block for the runtime context, or an explicit statement of what is
    unavailable and why.

    This used to return "" for both "no record exists" and "we have not resolved
    this caller yet", so the model received no farmer block at all and could not
    distinguish the two — or tell the caller anything useful about either. It
    then invented identifiers and called the tools anyway. The tools are now
    withheld on those turns (agents.voice.services.farmer_identity); these lines are
    what lets the model say something true rather than "that feature does not
    exist" or claiming a booking it never made. See issue #282.
    """
    state = identity_state_for_envelope(envelope)
    if state != "found":
        return "\n".join(unavailable_capability_lines(state))

    first = envelope.farmers[0]
    tags = _extract_farmer_tags(envelope.farmers)
    societies = sorted({r.societyName for r in envelope.farmers if r.societyName})

    lines = [
        f"- Farmer records matched: {len(envelope.farmers)}",
        f"- Farmer data source: {envelope.source or 'unknown'}",
        f"- Farmer cache state: {'stale' if envelope.stale else 'fresh'}",
    ]
    if envelope.refreshAfter:
        lines.append(f"- Farmer refresh after: {envelope.refreshAfter}")
    if first.farmerName:
        lines.append(f"- Farmer name: {first.farmerName}")
    if societies:
        lines.append(f"- Societies: {', '.join(societies[:3])}")
    if first.farmerCode:
        lines.append(f"- Farmer code available: yes")
    union_code = first.model_dump().get("unionCode") or first.model_dump().get("union_code")
    society_code = first.model_dump().get("societyCode") or first.model_dump().get("society_code")
    if union_code:
        lines.append(f"- Union code: {union_code}")
    if society_code:
        lines.append(f"- Society code: {society_code}")
    if first.farmerCode:
        lines.append(f"- Farmer code: {first.farmerCode}")
    # Herd counts: always surface what we have. The agent answers from this
    # context (the brittle get_herd_summary / list_animal_tags / get_farmer_profile
    # tools were dropped — they read the same cache and returned "not available"
    # when the upstream record omitted totalAnimals even though tags were present).
    first_data = first.model_dump()
    total_animals = first.totalAnimals
    if total_animals is None and tags:
        total_animals = len(tags)  # fallback when upstream omits the count
    if total_animals is not None:
        lines.append(f"- Total animals: {total_animals}")
    cow = first_data.get("cow") or first_data.get("Cow")
    if cow is not None:
        lines.append(f"- Cows: {cow}")
    buffalo = first_data.get("buffalo") or first_data.get("Buffalo")
    if buffalo is not None:
        lines.append(f"- Buffaloes: {buffalo}")
    milking = first_data.get("totalMilkingAnimals") or first_data.get("Milking Animal")
    if milking is not None:
        lines.append(f"- Milking animals: {milking}")
    if tags:
        # All tags inline — no truncation, since the list-tags tool was dropped.
        lines.append(f"- Known animal tags: {', '.join(tags)}")
    # Which farmer to book for is only a real question when the answer changes
    # the visit. Technicians come from (unionCode, societyCode) alone, so when
    # every record sits in one society they all yield the same technician, the
    # same village and the same visit — 90% of multi-account mobiles. Asking
    # anyway is what killed the call: the mobile is shared by a household and
    # the names are not separable by ear (of 364 real pairs the agent offered,
    # 250 share a name token and 19 are byte identical), so the caller answers,
    # the answer fits both, and the agent asks again. 71 of the 265 calls in
    # "AI booking not done" died in that loop — the largest single remaining
    # source of failed bookings. See issue #282.
    if len(envelope.farmers) > 1:
        # Option numbers here must match the "Farmer option N" lines emitted below.
        _numbered = list(enumerate((r.model_dump() for r in envelope.farmers), start=1))

        def _codes(d: dict) -> tuple:
            return (
                str(d.get("unionCode") or d.get("union_code") or "").strip(),
                str(d.get("societyCode") or d.get("society_code") or "").strip(),
            )

        def _village(d: dict) -> tuple:
            """The village a record belongs to, for grouping.

            (unionCode, societyCode) is exactly the key the technician lookup
            uses — it takes nothing else — so
            two records share a village iff they share this pair. A union-only
            key would merge two societies of one union.

            Only ever called on `bookable` records, which carry both codes by
            construction, so there is no missing-code case to handle here; that
            is decided once, above, and those records are excluded.
            """
            return _codes(d)

        def _village_label(d: dict) -> Optional[str]:
            """A village name the caller could say out loud, or None.

            Deliberately never a society code or a farmer name: a caller cannot
            read out "00731", and offering farmer names is the byte-similar-name
            question this block exists to remove.
            """
            return str(d.get("societyName") or "").strip() or None

        # _fetch_ai_technicians returns None unless BOTH codes are present, and
        # create_ai_call needs all three, so a record missing either has no
        # technician group and cannot be booked at all.
        bookable = [(i, d) for i, d in _numbered if all(_codes(d))]

        lines.append("- Multiple farmer records are registered on this mobile number.")
        if not bookable:
            lines.append(
                "- None of these records carry the society and union codes needed to book, "
                "so an AI booking cannot be made from them."
            )
            lines.append(
                "- Do NOT ask which farmer name for the AI booking. Tell the caller their "
                "details are not available right now."
            )
        else:
            if len(bookable) < len(_numbered):
                lines.append(
                    "- Only these options can be used for an AI booking (the rest are missing "
                    f"society or union codes): {', '.join(f'Farmer option {i}' for i, _ in bookable)}."
                )
            # One entry per village, carrying the first speakable label and the
            # lowest option number in it — the model should never have to infer
            # the mapping from the option lines' society names.
            by_village: dict = {}
            for i, d in bookable:
                key = _village(d)
                entry = by_village.setdefault(key, {"label": None, "option": i})
                if entry["label"] is None:
                    entry["label"] = _village_label(d)
                entry["option"] = min(entry["option"], i)
            first_option = min(i for i, _ in bookable)
            labels = [e["label"] for e in by_village.values()]
            spoken = {_normalize_farmer_name(l) for l in labels if l}
            if len(by_village) == 1:
                names = [_normalize_farmer_name(d.get("farmerName")) for _, d in bookable]
                all_named = len(set(names)) == 1 and all(names)
                lines.append(
                    "- For AI booking, they are all in the same village, served by the same "
                    "technicians, so the visit is identical whichever record is used."
                    + (" They are duplicate records of one farmer." if all_named else "")
                )
                lines.append(
                    f"- Do NOT ask which farmer name for the AI booking. Use Farmer option {first_option}."
                )
            elif all(labels) and len(spoken) == len(labels):
                # Distinctness judged as the caller hears it: "RAMOS" and
                # "Ramos " are two schema entries but one spoken choice.
                choices = ", ".join(
                    f"{e['label']} (Farmer option {e['option']})" for e in by_village.values()
                )
                lines.append(
                    "- For AI booking, they are in different villages, which means different technicians."
                )
                lines.append(
                    f"- Ask which village the animal is in, then use the option named here: {choices}. "
                    "Do NOT ask which farmer name for the AI booking."
                )
            else:
                lines.append(
                    "- For AI booking, these records may be in different villages, but they "
                    "cannot be told apart by village name."
                )
                lines.append(
                    f"- Do NOT ask which farmer name for the AI booking. Use Farmer option {first_option}."
                )

    # No cap: a farmer omitted here cannot be selected for booking, and its
    # technician group in _build_ai_technician_summary becomes unreachable.
    for index, record in enumerate(envelope.farmers, start=1):
        record_data = record.model_dump()
        farmer_name = record_data.get("farmerName") or "Unknown farmer"
        society_name = record_data.get("societyName") or "Unknown society"
        farmer_code = record_data.get("farmerCode")
        union_code = record_data.get("unionCode") or record_data.get("union_code")
        society_code = record_data.get("societyCode") or record_data.get("society_code")
        lines.append(
            f"- Farmer option {index}: name={farmer_name}, society_name={society_name}, "
            f"farmer_code={farmer_code}, union_code={union_code}, society_code={society_code}"
        )

    _append_animal_records(lines, envelope)

    return "\n".join(lines)


SUPPORTED_SCHEME_CONTEXT_UNIONS = SUPPORTED_SCHEME_UNIONS


def _collect_farmer_unions(envelope: Optional[FarmerDataEnvelope]) -> list[str]:
    if envelope is None:
        return []

    seen: set[str] = set()
    unions: list[str] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        raw_union = record.get("unionName") or record.get("union_name")
        normalized_union = str(raw_union or "").strip().lower()
        if not normalized_union or normalized_union in seen:
            continue
        seen.add(normalized_union)
        unions.append(normalized_union)
    return unions


def _collect_farmer_accounts(envelope: Optional[FarmerDataEnvelope]) -> list[FarmerAccount]:
    """Extract every (union, society, farmer) account on the caller's mobile.

    A mobile can map to multiple PashuGPT accounts (e.g. a cow account and a
    buffalo account). The milk-collection tool fans out over all of these so
    a farmer's data is never missed because the agent picked one account.
    Deduplicated on (union_code, society_code, farmer_code).
    """
    if envelope is None:
        return []

    seen: set[tuple] = set()
    accounts: list[FarmerAccount] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        union_code = record.get("unionCode") or record.get("union_code")
        society_code = record.get("societyCode") or record.get("society_code")
        farmer_code = record.get("farmerCode") or record.get("farmer_code")
        if not (union_code and society_code and farmer_code):
            continue
        key = (str(union_code), str(society_code), str(farmer_code))
        if key in seen:
            continue
        seen.add(key)
        accounts.append(
            FarmerAccount(
                union_code=str(union_code),
                society_code=str(society_code),
                farmer_code=str(farmer_code),
                farmer_name=record.get("farmerName") or record.get("farmer_name"),
                society_name=record.get("societyName") or record.get("society_name"),
            )
        )
    return accounts


def _collect_farmer_location(envelope: Optional[FarmerDataEnvelope]) -> tuple[Optional[str], Optional[str]]:
    """The caller's (village, district), for the vet-office lookup.

    Read off the raw record rather than FarmerAccount: account collection drops
    any record without all three identity codes, and a caller we cannot book for
    can still be told where their nearest dispensary is. societyName is the
    village fallback — it is the village label the milk society is named for,
    and the one _build_compact_farmer_summary already treats as speakable.
    """
    if envelope is None:
        return None, None

    locations: list[tuple[Optional[str], Optional[str]]] = []
    for farmer in envelope.farmers:
        record = farmer.model_dump()
        village = (
            str(record.get("village") or record.get("Village") or "").strip()
            or (farmer.societyName or "").strip()
            or None
        )
        district = str(record.get("district") or record.get("District") or "").strip() or None
        if village or district:
            locations.append((village, district))

    if not locations:
        return None, None

    # One mobile may represent several household members in different villages.
    # Only use a profile location when every usable record is compatible with one
    # place. In particular, never manufacture a pair by taking the village from
    # one farmer and the district from another.
    def _location_key(value: Optional[str]) -> str:
        return _normalize_farmer_name(value)

    for i, left in enumerate(locations):
        for right in locations[i + 1:]:
            shared_field = False
            for left_value, right_value in zip(left, right):
                if left_value and right_value:
                    shared_field = True
                    if _location_key(left_value) != _location_key(right_value):
                        return None, None
            if not shared_field:
                return None, None

    # Prefer a complete record, but do not fill its missing field from a different
    # record. Compatibility above only proves that overlapping fields agree.
    return max(locations, key=lambda location: sum(bool(value) for value in location))


async def _build_union_scheme_summary(farmer_unions: list[str]) -> str:
    scheme_unions = resolve_supported_unions(farmer_unions, SUPPORTED_SCHEME_CONTEXT_UNIONS)
    if not scheme_unions:
        return ""

    lines = [
        "",
        "## Union schemes available",
        "- The following scheme titles are available from the union scheme cache. Use these titles and links for scheme-related questions. Retrieve full cached scheme details when the user asks about a specific scheme.",
    ]
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
    return "\n".join(lines)


def _dedupe_technicians(technicians: list[dict]) -> list[dict]:
    """Drop duplicate technician rows, preserving order.

    The upstream GetAITUserDetailsBySocietyCode endpoint can return the same
    technician more than once; chat dedupes identically in
    agents/farmer_context.py. Keyed on userId, falling back to name+mobile when
    the id is absent.
    """
    unique: dict[str, dict] = {}
    for technician in technicians:
        key = technician.get("userId") or (
            f"{technician.get('fullName')}|{technician.get('mobileNumber')}"
        )
        unique.setdefault(key, technician)
    return list(unique.values())


def _farmer_record_union_name(record: FarmerRecord) -> str | None:
    data = record.model_dump()
    union_name = data.get("unionName") or data.get("union_name")
    return union_name if isinstance(union_name, str) else None


def _farmer_record_identity(data: dict) -> tuple[str, str, str]:
    return (
        str(data.get("farmerCode") or data.get("farmer_code") or ""),
        str(data.get("societyCode") or data.get("society_code") or ""),
        str(data.get("unionCode") or data.get("union_code") or ""),
    )


def _technician_group_is_banned(
    group: dict,
    banned_identities: set[tuple[str, str, str]],
    banned_union_codes: set[str],
) -> bool:
    """True when this cached group belongs to a union banned from AI-call booking.

    Identity (farmerCode, societyCode, unionCode) is the primary match so a mixed
    mobile keeps Kaira technicians. Farmer+society and union-code fallbacks hide
    leftover Kutch groups when some codes on the group are incomplete.
    """
    identity = (
        str(group.get("farmerCode") or group.get("farmer_code") or ""),
        str(group.get("societyCode") or group.get("society_code") or ""),
        str(group.get("unionCode") or group.get("union_code") or ""),
    )
    if identity != ("", "", "") and identity in banned_identities:
        return True
    farmer_society = (identity[0], identity[1])
    if farmer_society != ("", "") and any(
        (farmer_code, society_code) == farmer_society
        for farmer_code, society_code, _union_code in banned_identities
    ):
        return True
    union_code = identity[2]
    return bool(union_code) and union_code in banned_union_codes


def _append_ai_call_union_ban_lines(lines: list[str], farmer: FarmerRecord) -> None:
    data = farmer.model_dump()
    farmer_name = data.get("farmerName") or data.get("farmer_name") or "Unknown farmer"
    society_name = data.get("societyName") or data.get("society_name") or "Unknown society"
    _, society_code, union_code = _farmer_record_identity(data)
    lines.append(
        f"- Technician group: farmer_name={farmer_name}, society_name={society_name}, "
        f"union_code={union_code or None}, society_code={society_code or None}"
    )
    lines.append("- AI call booking is not allowed for this union.")
    lines.append(f"- Tell the farmer: `{UNION_BANNED_MESSAGE}`")
    lines.append("- Do not ask which technician they want. Do not call `create_ai_call`.")


def _build_ai_technician_summary(envelope: Optional[FarmerDataEnvelope]) -> str:
    if envelope is None:
        return ""

    banned_farmers: list[FarmerRecord] = []
    banned_identities: set[tuple[str, str, str]] = set()
    banned_union_codes: set[str] = set()
    for farmer in envelope.farmers or []:
        if not is_ai_call_banned_union(_farmer_record_union_name(farmer)):
            continue
        banned_farmers.append(farmer)
        data = farmer.model_dump()
        identity = _farmer_record_identity(data)
        if identity != ("", "", ""):
            banned_identities.add(identity)
        if identity[2]:
            banned_union_codes.add(identity[2])

    if banned_farmers:
        logger.info(
            "Skipping AI technician context; union is banned from AI-call booking unions=%s",
            [_farmer_record_union_name(farmer) for farmer in banned_farmers],
        )

    all_farmers_banned = bool(envelope.farmers) and len(banned_farmers) == len(envelope.farmers)
    technician_groups = [] if all_farmers_banned else [
        group
        for group in (envelope.aiTechnicians or [])
        if not _technician_group_is_banned(group, banned_identities, banned_union_codes)
    ]
    lines: list[str] = []
    for farmer in banned_farmers:
        _append_ai_call_union_ban_lines(lines, farmer)

    if technician_groups:
        lines.append("- AI technician options for booking are internal context, not user-provided information.")
        lines.append("- The caller does not know which AI technicians are available unless you tell them by technician name.")
        lines.append("- AI technician options for booking are grouped by farmer and society.")
        lines.append("- Each technician option only has these fields: id, full_name, mobile_number.")
        lines.append("- When asking the farmer to choose a technician, use the technician full name in natural spoken form.")
        lines.append("- Do not ask by technician position, number, option index, or ordinal words such as first, second, or third.")
        lines.append("- Mention phone only if a disambiguating mobile number is needed.")
        # Every group and every technician is listed. A cap here is silent: the
        # model cannot offer a technician it never saw, nor match one the caller
        # names, and nothing marks the list as partial (AMUL-39).
        for group in technician_groups:
            farmer_name = group.get("farmerName") or "Unknown farmer"
            society_name = group.get("societyName") or "Unknown society"
            society_code = group.get("societyCode")
            union_code = group.get("unionCode")
            lines.append(
                f"- Technician group: farmer_name={farmer_name}, society_name={society_name}, "
                f"union_code={union_code}, society_code={society_code}"
            )
            technicians = _dedupe_technicians(group.get("technicians") or [])
            if not technicians:
                if technician_lookup_failed(group):
                    # Distinct from "none exist": the lookup errored, so the
                    # agent must not assert the society has no technicians.
                    lines.append(
                        "- AI technician option: could not be retrieved for this farmer group "
                        "right now; say technician details are temporarily unavailable and ask "
                        "the caller to try again later. Do not say the society has no technicians."
                    )
                else:
                    lines.append("- AI technician option: none available for this farmer group.")
                continue
            for technician in technicians:
                name = strip_ait_name_codes(technician.get("fullName"))
                mobile = technician.get("mobileNumber")
                user_id = technician.get("userId")
                option = "- AI technician option:"
                if user_id:
                    option += f" id={user_id}"
                if name:
                    option += f" full_name={name}"
                if mobile:
                    option += f", mobile_number={mobile}"
                lines.append(option)
    elif not banned_farmers:
        # Say what NOT to do, like the per-group branches above: with only the
        # terse line the model offered the prompt's own example names as if real.
        lines.append("- AI technician options for booking are not available in the current signed-in context.")
        lines.append("- Do NOT name any technician, and do not use a technician name from the instructions or an example.")
    return "\n".join(lines)
