"""
Signed-in farmer-data tools backed by the shared farmer cache.
These are intended to be exposed only for sessions with a resolved mobile number.
"""
import json

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from helpers.gujarati_numbers import mask_tag_identifier


async def _get_envelope(ctx: RunContext[FarmerContext]):
    mobile = ctx.deps.mobile
    if not mobile:
        return None
    # Lazy import: this module is pulled in by agents.tools/__init__, and
    # farmer_cache imports back into agents.tools at module top — importing
    # get_or_fetch_farmer_data here (not at module scope) breaks that cycle so
    # farmer_cache can be imported cold (e.g. by the refresh worker at startup).
    from agents.services.farmer_cache import get_or_fetch_farmer_data

    return await get_or_fetch_farmer_data(mobile)


async def get_farmer_profile(ctx: RunContext[FarmerContext]) -> str:
    """
    Return the signed-in farmer profile from cached farmer data.
    Use this when the caller asks about their registered farmer identity, society, or account-level details.
    """
    envelope = await _get_envelope(ctx)
    if envelope is None or not envelope.farmers:
        return "Farmer profile is not available right now."

    first = envelope.farmers[0].model_dump()
    profile = {
        "source": envelope.source,
        "record_count": len(envelope.farmers),
        "farmer_name": first.get("farmerName"),
        "farmer_code": first.get("farmerCode"),
        "society_name": first.get("societyName"),
        "society_code": first.get("societyCode"),
        "union_name": first.get("unionName"),
        "union_code": first.get("unionCode"),
        "mobile_number": first.get("mobileNumber") or ctx.deps.mobile,
    }
    return json.dumps(profile, ensure_ascii=False)


async def get_herd_summary(ctx: RunContext[FarmerContext]) -> str:
    """
    Return herd-level counts from cached farmer data.
    Use this when the caller asks about total animals, herd composition, or milking-animal counts.
    """
    envelope = await _get_envelope(ctx)
    if envelope is None or not envelope.farmers:
        return "Herd summary is not available right now."

    first = envelope.farmers[0].model_dump()
    summary = {
        "source": envelope.source,
        "record_count": len(envelope.farmers),
        "total_animals": first.get("totalAnimals"),
        "cow_count": first.get("cow") or first.get("Cow"),
        "buffalo_count": first.get("buffalo") or first.get("Buffalo"),
        "milking_animals": first.get("totalMilkingAnimals") or first.get("Milking Animal"),
    }
    return json.dumps(summary, ensure_ascii=False)


async def list_animal_tags(ctx: RunContext[FarmerContext]) -> str:
    """
    Return the registered animal tags from cached farmer data.
    Use this when the caller asks which animals are registered or which tags are available.
    """
    envelope = await _get_envelope(ctx)
    if envelope is None or not envelope.farmers:
        return "Animal tags are not available right now."

    tags: list[str] = []
    for farmer in envelope.farmers:
        raw = farmer.tagNumbers or farmer.tagNo or ""
        for tag in str(raw).split(","):
            masked = mask_tag_identifier(tag.strip())
            if masked and masked not in tags:
                tags.append(masked)

    return json.dumps(
        {
            "source": envelope.source,
            "record_count": len(envelope.farmers),
            "animal_tags": tags,
        },
        ensure_ascii=False,
    )
