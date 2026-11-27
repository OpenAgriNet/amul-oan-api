"""Source-backed district and scheme choices for interactive chat."""
import json
import re
import time
import unicodedata
from uuid import uuid4

from agents.tools import vistaar
from agents.tools.beckn.amul import fetch_authenticated_farmers
from agents.tools.beckn.network import discover_union_schemes
from agents.tools.districts import DISTRICTS, resolve_place, _ALIASES
from agents.tools.scheme_codes import SCHEME_CODES, SCHEME_LABELS, SCHEME_ALIASES
from agents.tools.models.union import resolve_supported_unions
from agents.tools.union_schemes import SUPPORTED_SCHEME_UNIONS
from app.config import settings
from app.agui.models import ErrorDetail, ErrorResponse

PURPOSES = {"mandi_location", "scheme"}
TITLES = {
    "en": ("Choose the search district", "Choose a scheme"),
    "gu": ("શોધ માટે જિલ્લો પસંદ કરો", "યોજના પસંદ કરો"),
    "hi": ("खोज के लिए जिला चुनें", "योजना चुनें"),
    "mr": ("शोधासाठी जिल्हा निवडा", "योजना निवडा"),
}


def normalized(text):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def central_matches(query):
    query = normalized(query)
    return [code for code in SCHEME_CODES if any(
        re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", query)
        for term in (*SCHEME_ALIASES[code], SCHEME_LABELS[code]))] if query else []


class SelectorProvider:
    async def profile(self, mobile, session, run):
        from agents.farmer_context import _collect_farmer_location
        farmers = await fetch_authenticated_farmers(mobile, session_id=session, tool_call_id=run + ":selector-profile", force_refresh=True)
        return dict(district=_collect_farmer_location(farmers or []).get("district"),
                    unions=resolve_supported_unions([f.union_name for f in farmers or []], SUPPORTED_SCHEME_UNIONS))

    async def mandi(self, commodity, area, period):
        where = vistaar.SearchLocation(DISTRICTS[area], "explicit")
        items, used = await vistaar.fetch_mandi_items(commodity, where, *period)
        return dict(items=items, place=vistaar._location_phrase(where, used))

    async def union_schemes(self, union):
        return await discover_union_schemes("", union)

    async def central_scheme(self, code):
        return await vistaar.fetch_scheme_items(code)


def selector_context(record):
    context = []
    for kind, flow in record.get("selectors", {}).items():
        item = dict(kind=kind, stage=flow["stage"], commodity=flow.get("commodity"),
                    area=flow.get("area"), period=flow.get("period"), scheme=flow.get("scheme"))
        result = flow.get("result")
        if result and len(result.encode()) <= 4096:
            item["result"] = result
        context.append(item)
    encoded = json.dumps(context, ensure_ascii=False)
    return encoded if len(encoded.encode()) <= 12 * 1024 else "Saved selector results are too large for context. Request their source again before answering details."


async def advance_selectors(record, request, mobile, provider, message_id, *, intent=None, target=None):
    from app.agui.service import MAX_COMPONENTS, UIError

    flows = record.setdefault("selectors", {})
    if intent is not None:
        kind = intent["kind"]
        previous = flows.get(kind, {})
        for component in record["ui"]["components"]:
            if component["purpose"] in PURPOSES and component["workflow_id"] == previous.get("id") and component["status"] == "pending":
                component.update(status="expired", revision=component["revision"] + 1)
        flow = dict(id=uuid4().hex, stage="starting", options={})
        flows[kind] = flow
    else:
        kind = "schemes" if target["purpose"] == "scheme" else "mandi"
        flow = flows.get(kind)
        if not flow or flow["id"] != target["workflow_id"] or flow["stage"] != target["purpose"]:
            raise UIError(409, "STALE_COMPONENT", "This selection is no longer active.")
        value = flow["options"][request.forwardedProps.ui_action.value.option_id]
        target.update(status="submitted", revision=target["revision"] + 1,
                      selected=request.forwardedProps.ui_action.value.model_dump())

    def result(text, *, error=None):
        flow.update(stage="failed" if error else "completed", result=text, options={})
        return text, error, True

    def fail(code, message):
        return result(message, error=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(exclude_none=True))

    def picker(purpose, choices, notice=""):
        if len(record["ui"]["components"]) >= MAX_COMPONENTS:
            raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")
        title = TITLES.get(request.forwardedProps.target_lang.lower(), TITLES["en"])[
            ("mandi_location", "scheme").index(purpose)]
        options, choices_public = {}, []
        for identity, label, description, terms in choices:
            option = uuid4().hex
            options[option] = identity
            choices_public.append(dict(id=option, label=label, description=description, search_terms=terms))
        flow.update(stage=purpose, options=options)
        record["ui"]["components"].append(dict(id=uuid4().hex, workflow_id=flow["id"], message_id=message_id,
            revision=1, title=title, status="pending", kind="choice", purpose=purpose, searchable=True,
            options=choices_public, selected=None))
        return title + ("\n\n" + notice if notice else ""), None, bool(notice)

    if kind == "mandi":
        if intent is not None:
            choose = intent.get("choose")
            flow.update(commodity=(intent.get("commodity") or "").strip(),
                        area=None, choose_area=choose == "location",
                        period=(previous.get("period") or list(vistaar._resolve_date_range())) if choose and not intent.get("price_date") and not intent.get("price_date_to")
                        else list(vistaar._resolve_date_range(intent.get("price_date"), intent.get("price_date_to"))))
            if choose == "location" and not flow["commodity"]:
                flow["commodity"] = previous.get("commodity")
            if intent.get("location") and not flow["choose_area"]:
                area = resolve_place(intent["location"])
                if area is None:
                    return fail("MANDI_LOCATION_UNSUPPORTED", "That place is outside the supported mandi search areas. Please choose a supported Gujarat district.")
                flow["area"] = area.key
                record["mandi_area"] = dict(key=area.key, expires=time.time() + settings.mandi_location_ttl_s)
        else:
            flow["area"] = value
            record["mandi_area"] = dict(key=value, expires=time.time() + settings.mandi_location_ttl_s)
        if not flow["commodity"]:
            return result("Which commodity would you like mandi prices for?")
        if flow["area"] is None and not flow["choose_area"]:
            remembered = record.get("mandi_area", {})
            if remembered.get("expires", 0) > time.time() and remembered.get("key") in DISTRICTS:
                flow["area"] = remembered["key"]
            else:
                try:
                    profile = await provider.profile(mobile, request.threadId, request.runId)
                    area = resolve_place(profile["district"])
                    flow["area"] = area.key if area else None
                except Exception:
                    pass  # Public market data remains usable after an explicit district choice.
        if flow["area"] is None:
            return picker("mandi_location", [(key, area.display, None, [key, *[alias for alias, district in _ALIASES.items() if district == key]]) for key, area in
                sorted(DISTRICTS.items(), key=lambda item: item[1].display)])
        commodity = flow["commodity"]
        try:
            data = await provider.mandi(commodity, flow["area"], flow["period"])
        except Exception:
            return fail("MANDI_UNAVAILABLE", "Mandi prices are temporarily unavailable. Please try again shortly.")
        start, end = flow["period"]
        if not data["items"]:
            return result(f"No mandi prices were found for {commodity} near {data['place']} between {start} and {end}.")
        return result(f"Mandi prices for {commodity} near {data['place']} ({start} to {end}):\n" + vistaar._format_mandi_items(data["items"]))

    async def central(code):
        flow["scheme"] = dict(source="central", code=code, title=SCHEME_LABELS[code])
        try:
            items = await provider.central_scheme(code)
        except Exception:
            return fail("SCHEMES_UNAVAILABLE", "Central government scheme information is temporarily unavailable.")
        return result("Central government scheme: " + SCHEME_LABELS[code] + "\n" + vistaar._format_items(items)
                      if items else f"No information was found for {SCHEME_LABELS[code]}.")

    def union_detail(item):
        flow["scheme"] = dict(source="union", union=item["union"], title=item["scheme_title"])
        return result("\n".join(str(part) for part in (item["scheme_title"], item.get("description"),
            "Dairy union: " + item["union"], "Source: " + str(item["source"]) if item.get("source") else None) if part))

    if intent is None:
        if value["source"] == "central":
            return await central(value["code"])
        try:
            profile = await provider.profile(mobile, request.threadId, request.runId)
        except Exception:
            return fail("SCHEMES_UNAVAILABLE", "Your dairy-union access could not be verified. Please try again shortly.")
        if value["record"]["union"] not in profile["unions"]:
            return fail("SCHEME_ACCESS_CHANGED", "This scheme is no longer available under your dairy-union profile. Please request the scheme list again.")
        return union_detail(value["record"])

    query, source = intent.get("query"), intent.get("source")
    matches = central_matches(query)
    if len(matches) == 1 and source in (None, "central"):
        return await central(matches[0])
    union_items, warnings, no_union = [], [], False
    if source != "central":
        try:
            profile = await provider.profile(mobile, request.threadId, request.runId)
            if profile["unions"]:
                union_items, warnings = await provider.union_schemes(profile["unions"][0])
            else:
                no_union = True
        except Exception:
            warnings = ["Amul union schemes"]
    union_items = [item for item in union_items if item.get("scheme_title") and item.get("union") and item.get("source_network") == "amul-union"]
    exact = [item for item in union_items if normalized(item["scheme_title"]) == normalized(query) and query]
    if len(exact) == 1 and not matches:
        return union_detail(exact[0])
    choices = [(dict(source="union", record=item), item["scheme_title"], item["union"] + " dairy union", []) for item in union_items]
    if source != "union":
        choices.extend((dict(source="central", code=code), SCHEME_LABELS[code], "Central government", list(SCHEME_ALIASES[code])) for code in SCHEME_CODES)
    notice = "Union scheme information is temporarily unavailable; this list may be incomplete." if warnings else ""
    if no_union and source != "union":
        notice = "No supported dairy-union catalog is available for this profile. Central government schemes are listed."
    if not choices:
        return fail("SCHEMES_UNAVAILABLE", notice) if warnings else result("No supported dairy-union schemes are available for your profile.")
    return picker("scheme", choices, notice)
