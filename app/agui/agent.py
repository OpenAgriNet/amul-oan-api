"""Adapt the existing chat pipeline to one committed AG-UI operation."""
from types import MappingProxyType
import json
from uuid import uuid4

from fastapi import BackgroundTasks
from pydantic_core import to_jsonable_python
from redis.exceptions import RedisError

from agents.tools.milk_collection import InteractiveMilk
from agents.tools.ai_call import InteractiveBooking
from agents.tools.farmer import normalize_phone_to_mobile
from agents.tools.session_pmkisan import mask_private_numbers, owner_mobile
from app.agui.milk import account_identity
from app.agui.service import UIError
from app.agui.models import ErrorDetail, ErrorResponse
from app.channels.chat import WEB
from app.services.chat import CHAT_SURFACE, _BackgroundTasksScheduler, _input_method, run_turn
from app.tasks.suggestions import create_suggestions
from app.turn.types import ArtifactEmission, TextEmission, Turn


async def agent_reply(record, history, commit, *, request, user_info, provider, trace_id):
    selected = record["ui"]["selected_milk_account"]
    data = None
    if selected:
        option = record["options"].get(selected["option_id"])
        try:
            accounts = await provider.accounts(normalize_phone_to_mobile(user_info["phone"]), request.threadId, request.runId + ":context")
            owned = option and tuple(option["identity"]) in {account_identity(a) for a in accounts}
        except Exception:
            owned = False
        if owned:
            data = next((c["data"] for c in reversed(record["ui"]["components"])
                         if c["kind"] == "chart" and c["data"]["account"]["option_id"] == selected["option_id"]), None)
        else:
            selected = None
    ui = InteractiveMilk(selected, data, failed=record.get("date_reselection_required", False))
    booking_ui = InteractiveBooking()
    props = request.forwardedProps
    query = await mask_private_numbers(request.messages[0].content, request.threadId, owner_mobile(user_info, "farmer"))
    turn = Turn(query=query, session_id=request.threadId, source_lang=props.source_lang,
                target_lang=props.target_lang, user_id=props.user_id,
                authenticated_user=MappingProxyType(dict(user_info)), history=tuple(history),
                history_session_id=request.threadId, persona="farmer", channel=WEB,
                input_method=_input_method(props.input_method), trace_id=trace_id,
                interactive_milk=True, milk_context=ui.context())
    tasks = BackgroundTasks()
    reply = dict(text="", message_id=uuid4().hex, artifacts=[], history=None,
                 intent=None, booking_intent=None, suggestions_pending=False)
    response = None

    async def capture_history(key, messages, **kwargs):
        reply["history"] = to_jsonable_python(messages)

    def configure(deps):
        deps._milk_ui = ui
        deps._booking_ui = booking_ui
        booking = record.get("booking")
        if booking:
            context = {k: booking.get(k) for k in ("stage", "summary", "outcome")}
            encoded = json.dumps(context, ensure_ascii=False)
            deps.booking_context = encoded if len(encoded.encode()) <= 4096 else json.dumps({"stage": booking["stage"]})

    async def finalize(outcome):
        nonlocal response
        reply["intent"] = ui.request
        reply["booking_intent"] = booking_ui.request
        reply["suggestions_pending"] = any(task.func is create_suggestions for task in tasks.tasks)
        response = await commit(reply)
        if outcome == "error" and response["error"] is None:
            response["error"] = ErrorResponse(error=ErrorDetail(code="AGENT_UNAVAILABLE",
                message=reply["text"] or "Chat is temporarily unavailable. Please try again.", retryable=True)).model_dump(exclude_none=True)
        return "error" if response["error"] else outcome

    try:
        async for emission in run_turn(turn, CHAT_SURFACE, scheduler=_BackgroundTasksScheduler(tasks),
                                       history_writer=capture_history, configure_deps=configure, finalize=finalize):
            if isinstance(emission, TextEmission):
                reply["text"] += emission.text
            elif isinstance(emission, ArtifactEmission):
                reply["artifacts"].extend(dict(a) for a in emission.artifacts)
    except (UIError, RedisError):
        raise
    except Exception:
        raise UIError(503, "AGENT_UNAVAILABLE", "Chat is temporarily unavailable. Please try again.", retryable=True) from None
    response["background_tasks"] = tasks.tasks
    return response
