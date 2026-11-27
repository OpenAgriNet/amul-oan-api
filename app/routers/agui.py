"""Development-only interactive milk endpoint using existing authenticated APIs."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from types import MappingProxyType
from uuid import uuid4
from contextlib import nullcontext
from functools import partial
import asyncio

from agents.tools.farmer import normalize_phone_to_mobile
from app.agui.models import MAX_BODY_BYTES, ChatRunRequest
from app.agui.service import MilkStore, UIError, wire_chunks
from app.auth.jwt_auth import get_current_user
from app.config import settings
from app.core.cache import redis_client
from app.models.requests import ChatRequest
from app.personas import resolve_chat_persona
from app.utils import DEFAULT_CACHE_TTL, history_ttl

router = APIRouter(prefix="/chat", tags=["agui"])


def preview_enabled():
    return settings.environment == "development" and settings.chat_agui_preview_enabled


def require_preview():
    if not preview_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


def get_milk_store():
    return MilkStore(redis_client, settings.redis_key_prefix, history_ttl("chat").get("ttl", DEFAULT_CACHE_TTL))


@router.post("/", dependencies=[Depends(require_preview)])
async def interactive_chat(http_request: Request, background_tasks: BackgroundTasks,
                           user_info: dict = Depends(get_current_user), store=Depends(get_milk_store)):
    try:
        body = bytearray()
        async for chunk in http_request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                raise UIError(413, "REQUEST_TOO_LARGE", "Request body exceeds 256 KiB.")
        try:
            request = ChatRunRequest.model_validate_json(bytes(body))
        except (ValidationError, ValueError):
            # Default validation errors contain user input, potentially Aadhaar/OTP.
            raise UIError(422, "INVALID_REQUEST", "Send one valid message, action, or command; use YYYY-MM-DD dates at most 31 days apart.") from None
        if http_request.query_params:
            raise UIError(422, "INVALID_REQUEST", "POST options belong in the JSON request, not query parameters.")
        props = request.forwardedProps
        persona = resolve_chat_persona(user_info, props.persona)
        owner = str(user_info.get("sub") or "")
        if not settings.chat_agui_enabled or props.channel != "web" or persona != "farmer":
            if not request.messages:
                raise UIError(409, "AGUI_DISABLED", "Interactive milk controls are not enabled for this request.")
            if owner:
                await store.clear(owner, persona, request.threadId)
            from app.routers.chat import chat_endpoint
            result = await chat_endpoint(background_tasks, ChatRequest(
                query=request.messages[0].content, session_id=request.threadId,
                source_lang=props.source_lang, target_lang=props.target_lang, channel=props.channel,
                user_id=props.user_id, persona=props.persona, input_method=props.input_method,
            ), user_info)
            result.headers["X-Chat-Format"] = "text-v1"
            result.headers["Cache-Control"] = "no-store"
            return result
        mobile = normalize_phone_to_mobile(str(user_info.get("phone") or ""))
        if not owner or not mobile or user_info.get("anonymous"):
            raise UIError(403, "FARMER_SIGN_IN_REQUIRED", "Sign in with your registered farmer number to view milk records.")
        from app.channels.chat import WEB
        from app.llm_core import trace as pipeline_trace
        from app.services.chat import _ChatTelemetry, _input_method
        from app.turn.types import Turn

        trace_id = uuid4().hex
        telemetry = None
        if not request.messages:
            turn = Turn(query="Milk UI selection", session_id=request.threadId,
                        source_lang=props.source_lang, target_lang=props.target_lang, user_id=props.user_id,
                        authenticated_user=MappingProxyType(dict(user_info)), history=(), history_session_id=request.threadId,
                        persona=persona, channel=WEB, input_method=_input_method(props.input_method), trace_id=trace_id)
            telemetry = _ChatTelemetry(turn, pipeline_profile="agui-milk", pipeline_trace=pipeline_trace.begin("agui-milk"))
        from app.agui.agent import agent_reply

        with telemetry.root() if telemetry is not None else nullcontext():
            try:
                response = await store.apply(request, owner, persona, mobile, trace_id,
                                             agent_reply=partial(agent_reply, request=request, user_info=user_info,
                                                                 trace_id=trace_id) if request.messages else None,
                                             disconnected=http_request.is_disconnected)
                background_tasks.tasks.extend(response.pop("background_tasks", []))
                if telemetry is not None:
                    telemetry.record_output(response["text"], "ui")
                    telemetry.record_outcome("error" if response["error"] else "success")
            except BaseException as exc:
                if telemetry is not None:
                    telemetry.record_outcome("cancelled" if isinstance(exc, asyncio.CancelledError) else "error")
                raise
    except UIError as exc:
        headers = {"Cache-Control": "no-store"}
        if exc.trace_id:
            headers["X-Trace-Id"] = exc.trace_id
        return JSONResponse(exc.body, status_code=exc.status, headers=headers)

    from app.routers.chat import ClosingStreamingResponse

    async def stream():
        for chunk in wire_chunks(request, response):
            yield chunk

    return ClosingStreamingResponse(stream(), media_type="text/event-stream", headers={
        "X-Chat-Format": "agui-v1", "X-Trace-Id": response["trace_id"], "Cache-Control": "no-store",
    })
