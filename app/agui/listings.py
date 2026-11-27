"""Read-only source-backed collections, filters and listing details."""
import json
from types import SimpleNamespace
from uuid import uuid4

from agents.tools import cattle_trade as cattle, fodderinfo as fodder
from agents.tools.models.listings import CattleFilters, FodderFilters
from app.agui.models import ErrorDetail, ErrorResponse

FILTERS = {"cattle": CattleFilters, "fodder": FodderFilters}


def facts(data, prefix=""):
    """Retain source values under stable keys for frontend-localized labels."""
    result = []
    for key, value in data.items():
        key = prefix + key
        if isinstance(value, dict):
            result.extend(facts(value, key + "."))
        elif value is not None and isinstance(value, (str, int, float, bool)):
            result.append(dict(key=key, value=value))
    return result


def cattle_data(item):
    return dict(listing_number=item.number, species=cattle._species(item), breed=cattle._breed_name(item), district=cattle._district_name(item),
        village=item.village_name, asking_price_inr=str(item.cattle_price), milk_per_day_litres=str(item.milk_per_day),
        age=cattle._age(item.age), lactation=item.lactation,
        milking_status=item.milking_status, pregnancy_status=item.pregnancy_status)


class ListingProvider:
    async def search(self, source, filters, district):
        if source == "fodder":
            return await fodder.search_fodder_availability(**filters)
        ctx = SimpleNamespace(deps=SimpleNamespace(get_farmer_district=lambda: district))
        return await cattle.search_verified_cattle_listings(ctx, **filters)

    async def detail(self, source, identity):
        if source == "fodder":
            return await fodder.get_fodder_listing(identity["listing_id"])
        snapshot = await cattle.get_snapshot()
        if snapshot is None:
            return dict(status="unavailable")
        item = next((row for row in snapshot.listings
                     if row.key == identity["key"] and row.number == identity["number"]), None)
        stale = cattle.snapshot_age_seconds(snapshot) > 2 * cattle.settings.cattle_listings_refresh_seconds
        return dict(status="ok" if item else "not_found", listing=item, fetched_at=snapshot.fetched_at,
                    stale=stale, notice="These cached listings may have been sold; current availability could not be confirmed." if stale else None)


def filter_definitions(source, data):
    enums = {"species": ["cow", "buffalo"], "milking_status": ["milking", "dry"],
             "pregnancy_status": ["pregnant", "not pregnant"],
             "form": ["GREEN", "DRY", "HAY", "SILAGE", "CONCENTRATE"],
             "comparison_group": ["green", "dry", "hay", "silage", "concentrate"],
             "delivery": ["delivery", "pickup"], "sort": ["recent", "rate", "qty"]}
    numbers = {"max_price": "INR", "min_milk_per_day": "litres/day", "max_rate_per_kg": "INR/kg", "min_qty_kg": "kg", "limit": None}
    result = []
    for name in FILTERS[source].model_fields:
        options = enums.get(name, [])
        if source == "cattle" and name in ("breed", "district"):
            options = data.get("breeds" if name == "breed" else "districts", [])
        result.append(dict(name=name, kind="number" if name in numbers else "select" if name in enums else "text",
            options=options, minimum=1 if name == "limit" else 0 if name in numbers else None,
            maximum=20 if name == "limit" else None, unit=numbers.get(name)))
    return result


def listing_context(record):
    saved = []
    for source, flow in record.get("listings", {}).items():
        collection = next((c for c in record['ui']['components'] if c['id'] == flow['collection_id']), None)
        if collection:
            saved.append(dict(source=source, filters=flow['filters'], source_status=collection['source_status'],
                match_count=collection['match_count'], displayed_count=collection['displayed_count'],
                stale=collection['stale'], cards=collection['cards'], clarification=flow.get('clarification')))
        details = [c for c in record['ui']['components'] if c['kind'] == 'listing_detail'
                   and c['workflow_id'] == flow['id'] and c['status'] != 'expired']
        if details:
            saved.append(dict(source=source, detail=details[-1]))
    encoded = json.dumps(saved, ensure_ascii=False)
    if len(encoded.encode()) > 16 * 1024:
        for item in saved:
            if 'cards' in item:
                item['cards'] = [{k: card[k] for k in ('listing_ref', 'listing_number', 'title')} for card in item['cards']]
            if 'detail' in item:
                item['detail'] = {k: item['detail'][k] for k in ('listing_ref', 'source_status', 'stale')}
        return json.dumps(dict(collections=saved, notice="Detailed facts omitted; request the source before answering them."), ensure_ascii=False)
    return encoded


async def advance_listings(record, request, provider, message_id, *, intent=None, target=None):
    from app.agui.service import MAX_COMPONENTS, UIError
    flows, components = record.setdefault("listings", {}), record['ui']['components']
    action = request.forwardedProps.ui_action
    source = intent['kind'] if intent else target['purpose']
    flow = flows.get(source)
    opening = bool(intent and intent.get('listing_ref')) or bool(action and action.kind == 'open_listing')
    if (not intent or opening) and (not flow or (target and target['id'] != flow['collection_id'])):
        raise UIError(409, "STALE_COMPONENT", "This listing collection is no longer active.")

    def capacity():
        if len(components) >= MAX_COMPONENTS:
            raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")

    def base(kind, title):
        return dict(id=uuid4().hex, workflow_id=flow['id'], message_id=message_id,
                    revision=1, title=title, status="pending", kind=kind, purpose=source)

    def error(status, text):
        failure = ErrorResponse(error=ErrorDetail(code='LISTINGS_UNAVAILABLE', message=text, retryable=True)).model_dump(exclude_none=True) if status == 'unavailable' else None
        return text, failure

    if opening:
        ref = intent.get('listing_ref') if intent else action.value.listing_ref
        if ref not in flow['refs']:
            raise UIError(422, "INVALID_LISTING", "Choose a listing from the displayed collection.")
        detail = next((c for c in components if c['kind'] == 'listing_detail' and c['workflow_id'] == flow['id'] and c['status'] != 'expired'), None)
        if not detail:
            capacity()
            detail = base('listing_detail', 'Listing details')
            components.append(detail)
        try:
            data = await provider.detail(source, flow['refs'][ref])
        except Exception:
            data = dict(status='unavailable')
        status, item = data['status'], data.get('listing')
        detail.update(message_id=message_id, revision=detail['revision'] + 1, status='submitted',
            collection_id=flow['collection_id'], listing_ref=ref, facts=[], seller=None,
            source_status=status, notice=data.get('notice'), fetched_at=data.get('fetched_at'), stale=data.get('stale', False))
        text = 'This listing is no longer available. Search again to see current listings.'
        if status == 'unavailable':
            text = 'Listing details are temporarily unavailable. Please try again.'
        elif status == 'ok' and item is not None:
            if source == 'cattle':
                detail['facts'] = facts(cattle_data(item))
                detail['seller'] = dict(name=item.farmer_name, mobile=item.farmer_mobile_number)
            else:
                detail['facts'] = facts({k: v for k, v in item.items() if k != 'seller'})
                detail['seller'] = item.get('seller')
            text = 'Listing details are shown below. Confirm availability with the seller before paying.'
        return error(status, text + ('\n' + detail['notice'] if detail['notice'] else ''))

    filters = FILTERS[source].model_validate(intent['filters']) if intent else action.value.filters
    if not isinstance(filters, FILTERS[source]) or (action and action.value.source != source):
        raise UIError(422, "INVALID_FILTERS", "Filters must match the listing source.")
    if action and source == 'cattle':
        for value, valid in ((filters.species, cattle._species_of(filters.species or '') or cattle._match_key(filters.species or '') in cattle._ANY_SPECIES_WORDS),
                             (filters.milking_status, cattle._status(filters.milking_status, cattle._MILKING)),
                             (filters.pregnancy_status, cattle._status(filters.pregnancy_status, cattle._PREGNANCY))):
            if value and not valid:
                raise UIError(422, 'INVALID_FILTERS', 'Choose a supported species or status.')
    if intent:
        capacity()
        if flow:
            for c in components:
                if c['workflow_id'] == flow['id'] and c['status'] != 'expired':
                    c.update(status='expired', revision=c['revision'] + 1)
        flow = dict(id=uuid4().hex, district=intent.get('district'), refs={})
        flows[source] = flow
        collection = base('listing_collection', 'Verified cattle for sale' if source == 'cattle' else 'Fodder for sale')
        flow['collection_id'] = collection['id']
        components.append(collection)
    else:
        collection = target
        collection.update(revision=collection['revision'] + 1, message_id=message_id)
        for c in components:
            if c['kind'] == 'listing_detail' and c['workflow_id'] == flow['id'] and c['status'] != 'expired':
                c.update(status='expired', revision=c['revision'] + 1)
    flow.update(filters=filters.model_dump(), refs={})
    try:
        data = await provider.search(source, flow['filters'], flow['district'])
    except Exception:
        data = dict(status='unavailable')
    clarification = {key: data[key] for key in ('message', 'resolution') if data.get(key) is not None}
    flow['clarification'] = clarification if len(json.dumps(clarification, ensure_ascii=False).encode()) <= 4096 else None
    cards = []
    for item in data.get('listings', []):
        ref = uuid4().hex
        if source == 'cattle':
            flow['refs'][ref] = dict(key=item.key, number=item.number)
            card = dict(listing_ref=ref, listing_number=item.number, title=f'{cattle._breed_name(item)} {cattle._species(item)}',
                        facts=facts(cattle_data(item)))
        else:
            flow['refs'][ref] = dict(listing_id=item['listing_id'])
            summary = {key: item.get(key) for key in ('material', 'location', 'quantity', 'price', 'freshness')}
            card = dict(listing_ref=ref, listing_number=None,
                        title=item.get('material', {}).get('canonical_name_en') or 'Fodder', facts=facts(summary))
        cards.append(card)
    status = data['status']
    collection.update(filters=flow['filters'], filter_definitions=filter_definitions(source, data),
        cards=cards, displayed_count=len(cards), match_count=data.get('match_count') if source == 'cattle' else None,
        source_total=data.get('source_total') if source == 'cattle' else data.get('total'),
        elsewhere_count=data.get('elsewhere_count') if source == 'cattle' else None,
        source_status=status, notice=data.get('notice'), fetched_at=data.get('fetched_at'), stale=data.get('stale', False))
    text = f"Showing {len(cards)} listings. Use filters to narrow the search or open a listing for details."
    if not cards:
        text = {'unavailable': 'Listings are temporarily unavailable. Please try again.',
                'clarification_needed': 'Please clarify the fodder name or place in chat.',
                'location_not_found': 'That place could not be found. Please clarify it in chat.',
                'name_not_resolved': 'That fodder name could not be resolved. Please clarify it in chat.'}.get(status, 'No current listings match these filters.')
    if source == 'fodder' and status in ('clarification_needed', 'location_not_found') and data.get('message'):
        text = data['message']
    if source == 'cattle' and data.get('message'):
        text = data['message'] + ('\n' + text if cards else '')
    return error(status, text + ('\n' + collection['notice'] if collection['notice'] else ''))
