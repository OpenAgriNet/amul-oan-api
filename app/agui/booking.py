"""Server-owned AI booking controls. Only a durably claimed confirm may send."""
from uuid import uuid4

from agents.tools.ai_call import book_ai_call_result, _invalid_booking_identifier, UNCONFIRMED_MESSAGE
from agents.tools.beckn.amul import authenticated_accounts, fetch_authenticated_farmers, search_ai_technicians
from agents.tools.models.ai_call import AISpecies, BookingOutcome
from agents.tools.models.union import any_union_banned_from_ai_calls, union_banned_message
from app.agui.milk import account_identity, account_label
from app.agui.models import ErrorDetail, ErrorResponse


TEXT = {
    "en": ("Choose an AI technician", "Choose the animal", "Confirm AI visit", "Cow", "Buffalo", "AI booking cancelled."),
    "gu": ("બીજદાન માટે ટેકનિશિયન પસંદ કરો", "પશુ પસંદ કરો", "બીજદાનની મુલાકાતની પુષ્ટિ કરો", "ગાય", "ભેંસ", "બીજદાનની બુકિંગ રદ કરી છે."),
    "hi": ("कृत्रिम गर्भाधान तकनीशियन चुनें", "पशु चुनें", "कृत्रिम गर्भाधान की पुष्टि करें", "गाय", "भैंस", "कृत्रिम गर्भाधान की बुकिंग रद्द कर दी गई है।"),
    "mr": ("कृत्रिम रेतन तंत्रज्ञ निवडा", "पशू निवडा", "कृत्रिम रेतन भेटीची पुष्टी करा", "गाय", "म्हैस", "कृत्रिम रेतन बुकिंग रद्द केले आहे."),
}


def words(language):
    return TEXT.get(language.lower(), TEXT["en"])


class BookingProvider:
    async def farmers(self, mobile, session_id, run_id):
        return await fetch_authenticated_farmers(mobile, session_id=session_id,
            tool_call_id=f"{run_id}:booking-profile", force_refresh=True)

    async def technicians(self, account, session_id, run_id):
        return await search_ai_technicians(union_code=account.union_code, society_code=account.society_code,
            session_id=session_id, tool_call_id=f"{run_id}:technicians", force_refresh=True)

    async def book(self, booking, session_id):
        union, society, farmer = booking["identity"]
        technician, species = booking["technician_id"], AISpecies(booking["species"])
        return await book_ai_call_result(union, society, farmer, technician, species, session_id,
            dict(union_code=union, society_code=society, farmer_code=farmer, user_id=technician, species=species.value),
            tool_call_id=f"ai-booking:{booking['id']}")


def booking_error(code, message):
    return ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(exclude_none=True)


def expire_booking(record):
    for item in record["ui"]["components"]:
        if item["purpose"].startswith("ai_") and item["status"] == "pending":
            item.update(status="expired", revision=item["revision"] + 1)


async def advance_booking(record, request, mobile, provider, message_id, *, intent=None, target=None):
    from app.agui.service import MAX_COMPONENTS, UIError

    props, ui = request.forwardedProps, record["ui"]
    text = words(props.target_lang)
    booking = record.get("booking")
    if intent is None:
        workflow_id = props.command.workflow_id if props.command else target["workflow_id"]
        if not booking or booking["id"] != workflow_id or booking["stage"] not in ("technician", "species", "confirm"):
            raise UIError(409, "STALE_COMPONENT", "This booking is no longer awaiting a selection.")
        if target and target["purpose"] != {"technician": "ai_technician", "species": "ai_species", "confirm": "ai_booking"}[booking["stage"]]:
            raise UIError(409, "STALE_COMPONENT", "This booking changed. Use its latest control.")

    if (props.command and props.command.kind == "cancel_ai_booking") or (
            props.ui_action and props.ui_action.kind == "confirm" and not props.ui_action.value.confirmed):
        expire_booking(record)
        if target:
            target.update(status="submitted", revision=target["revision"] + 1, selected={"confirmed": False})
        booking["stage"] = "cancelled"
        return text[5], None, False

    def fail(code, message):
        expire_booking(record)
        if booking:
            booking["stage"] = "unavailable"
        return message, booking_error(code, message), False

    try:
        farmers = await provider.farmers(mobile, request.threadId, request.runId)
    except Exception:
        raise UIError(503, "BOOKING_CONTEXT_UNAVAILABLE", "Your farmer details are temporarily unavailable.", retryable=True) from None
    if any_union_banned_from_ai_calls([f.union_name for f in farmers or []]):
        return fail("AI_BOOKING_NOT_ALLOWED", union_banned_message(props.target_lang))
    accounts = {account_identity(a): a for a in authenticated_accounts(farmers or [])}
    if intent is not None:
        selected = ui["selected_milk_account"]
        option = record["options"].get(selected["option_id"]) if selected else None
        identity = tuple(option["identity"]) if option else None
        # A removed explicit selection must not silently switch to another household member.
        if selected and identity not in accounts:
            return fail("BOOKING_CONTEXT_UNAVAILABLE", "Your selected farmer account is no longer available.")
        if identity is None and len(accounts) == 1:
            identity = next(iter(accounts))
        if identity is None:
            return fail("BOOKING_CONTEXT_UNAVAILABLE", "A unique farmer account could not be resolved for this booking. Please contact your society.")
        expire_booking(record)
        booking = dict(id=uuid4().hex, identity=list(identity), species=intent["species"],
                       stage="technician", options={}, technician_id=None, summary=None, outcome=None)
        record["booking"] = booking
    account = accounts.get(tuple(booking["identity"]))
    if account is None:
        return fail("BOOKING_CONTEXT_UNAVAILABLE", "This farmer account is no longer on your signed-in profile.")
    try:
        technicians = await provider.technicians(account, request.threadId, request.runId)
    except Exception:
        raise UIError(503, "TECHNICIANS_UNAVAILABLE", "AI technicians are temporarily unavailable.", retryable=True) from None
    available = {t.userId: t for t in technicians if not _invalid_booking_identifier(*account_identity(account), t.userId)}
    if not available:
        return fail("NO_AI_TECHNICIANS", "No AI technicians are available for your society. Please contact your society.")

    def component(kind, purpose, title, **fields):
        if len(ui["components"]) >= MAX_COMPONENTS:
            raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")
        item = dict(id=uuid4().hex, workflow_id=booking["id"], message_id=message_id,
                    revision=1, title=title, status="pending", kind=kind, purpose=purpose, selected=None, **fields)
        ui["components"].append(item)
        return title

    def technician_picker():
        expire_booking(record)
        booking.update(stage="technician", technician_id=None, summary=None)
        booking["options"] = {uuid4().hex: dict(technician_id=t.userId, label=t.display_full_name or t.fullName,
                                               description=t.mobileNumber) for t in available.values()}
        return component("choice", "ai_technician", text[0], options=[
            dict(id=k, label=v["label"], description=v["description"]) for k, v in booking["options"].items()])

    def species_picker():
        booking.update(stage="species", species=None)
        return component("choice", "ai_species", text[1], options=[
            dict(id=species, label=label, description=None) for species, label in zip(("cow", "buffalo"), text[3:5])])

    def summary():
        option_id = next(k for k, v in booking["options"].items() if v["technician_id"] == booking["technician_id"])
        technician = available[booking["technician_id"]]
        return dict(account_label=account_label(account),
                    technician=dict(id=option_id, label=technician.display_full_name or technician.fullName, description=technician.mobileNumber),
                    species=booking["species"], species_label=text[3] if booking["species"] == "cow" else text[4])

    def confirm():
        booking.update(stage="confirm", summary=summary())
        return component("confirm", "ai_booking", text[2], summary=booking["summary"], outcome=None)

    if intent is not None:
        return technician_picker(), None, False
    if props.command:
        expire_booking(record)
        if props.command.field == "technician" or booking["technician_id"] not in available:
            return technician_picker(), None, False
        return species_picker(), None, False
    action = props.ui_action
    if target["purpose"] == "ai_technician":
        technician_id = booking["options"][action.value.option_id]["technician_id"]
        if technician_id not in available:
            return technician_picker(), None, False
        booking["technician_id"] = technician_id
    elif booking["technician_id"] not in available:
        return technician_picker(), None, False
    elif target["purpose"] == "ai_species":
        booking["species"] = action.value.option_id
    elif summary() != booking["summary"]:
        expire_booking(record)
        return confirm(), None, False
    else:
        # MilkStore commits this consumed confirmation BEFORE calling provider.book.
        target.update(status="submitted", revision=target["revision"] + 1, selected={"confirmed": True})
        booking["stage"] = "submitting"
        booking["outcome"] = BookingOutcome(status="unconfirmed", message=UNCONFIRMED_MESSAGE).model_dump()
        target["outcome"] = booking["outcome"]
        return UNCONFIRMED_MESSAGE, None, True
    target.update(status="submitted", revision=target["revision"] + 1, selected=action.value.model_dump())
    return (confirm() if booking["species"] else species_picker()), None, False


def complete_booking(record, response, result):
    booking = record["booking"]
    booking.update(stage=result.status, outcome=result.model_dump())
    for item in record["ui"]["components"]:
        if item["workflow_id"] == booking["id"] and item["kind"] == "confirm" and item["status"] == "submitted":
            item["outcome"] = booking["outcome"]
    record["ui"]["state_revision"] += 1
    response.update(text=result.message, state=record["ui"], error=None if result.status == "confirmed" else
                    booking_error("AI_BOOKING_" + result.status.upper(), result.message))
