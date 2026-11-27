"""Adapt the existing chat pipeline to one committed AG-UI operation."""
from types import MappingProxyType
import json
from uuid import uuid4

from fastapi import BackgroundTasks
from pydantic_core import to_jsonable_python
from redis.exceptions import RedisError

from agents.tools.milk_collection import InteractiveMilk
from agents.tools.ai_call import InteractiveBooking
from agents.tools.vistaar import InteractiveSelectors
from agents.tools.session_pmkisan import mask_private_numbers, owner_mobile
from app.agui.service import UIError
from app.agui.models import ErrorDetail, ErrorResponse
from app.agui.selectors import selector_context
from app.channels.chat import WEB
from app.services.chat import CHAT_SURFACE, _BackgroundTasksScheduler, _input_method, run_turn
from app.tasks.suggestions import create_suggestions
from app.turn.types import ArtifactEmission, TextEmission, Turn


async def agent_reply(record, history, commit, *, request, user_info, trace_id,
                      history_session_id, accounts_available):
    selected = record["ui"]["selected_milk_account"] if accounts_available else None
    data = None
    if selected:
        data = next((c["data"] for c in reversed(record["ui"]["components"])
                     if c["kind"] == "chart" and c["data"]["account"]["option_id"] == selected["option_id"]), None)
    ui = InteractiveMilk(selected, data, failed=record.get("date_reselection_required", False))
    booking_ui = InteractiveBooking()
    selector_ui = InteractiveSelectors()
    props = request.forwardedProps
    query = await mask_private_numbers(request.messages[0].content, request.threadId, owner_mobile(user_info, "farmer"))
    turn = Turn(query=query, session_id=request.threadId, source_lang=props.source_lang,
                target_lang=props.target_lang, user_id=props.user_id,
                authenticated_user=MappingProxyType(dict(user_info)), history=tuple(history),
                history_session_id=history_session_id, persona="farmer", channel=WEB,
                input_method=_input_method(props.input_method), trace_id=trace_id,
                interactive_milk=True, milk_context=ui.context())
    tasks = BackgroundTasks()
    reply = dict(text="", message_id=uuid4().hex, artifacts=[], history=None,
                 intent=None, booking_intent=None, selector_intent=None, suggestions_pending=False)
    response = None

    async def capture_history(key, messages, **kwargs):
        reply["history"] = to_jsonable_python(messages)

    def configure(deps):
        deps._milk_ui = ui
        deps._booking_ui = booking_ui
        deps._selector_ui = selector_ui
        deps.selector_context = selector_context(record)
        booking = record.get("booking")
        if booking:
            context = {k: booking.get(k) for k in ("stage", "summary", "outcome")}
            encoded = json.dumps(context, ensure_ascii=False)
            deps.booking_context = encoded if len(encoded.encode()) <= 4096 else json.dumps({"stage": booking["stage"]})

    async def finalize(outcome):
        nonlocal response
        reply["intent"] = ui.request
        reply["booking_intent"] = booking_ui.request
        reply["selector_intent"] = selector_ui.request
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


async def translate_selector_text(text, *, language, thread):
    """Use the normal output translation chain for deterministic UI results too."""
    from app.llm_core.execution import context
    from app.services.chat import _stream_to_client, _translates_output, _turn_root

    if not _translates_output(language.lower()):
        return text

    async def source():
        for line in text.splitlines(keepends=True):
            yield line

    try:
        execution = await context(thread, "chat")
        return "".join([chunk async for chunk in _stream_to_client(source(), translate_to=language,
            max_output_chars=None, execution=execution, output_chunks=[], root=_turn_root.get())])
    except Exception:
        # Match the normal stream's English fallback; never repeat the provider lookup.
        return text
