"""Development milk workflow. Provider reads stay behind authenticated identity."""
from __future__ import annotations

import copy
import asyncio
import hashlib
import hmac
import json
import secrets
import time
from uuid import uuid4

from ag_ui.core import (
    CustomEvent, RunErrorEvent, RunFinishedEvent, RunStartedEvent,
    StateSnapshotEvent, TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from redis.exceptions import WatchError

from agents.tools.beckn.amul import authenticated_accounts, fetch_authenticated_farmers, fetch_milk_collection
from app.agui.milk import account_identity, account_label, labels, milk_result
from app.agui.models import ErrorDetail, ErrorResponse, ChatRunRequest, UIState
from app.agui.session import storage_session_id

MAX_COMPONENTS = 50


class UIError(Exception):
    def __init__(self, status, code, message, *, component_id=None, retryable=False, trace_id=None):
        self.status = status
        self.body = ErrorResponse(error=ErrorDetail(code=code, message=message, component_id=component_id, retryable=retryable)).model_dump(exclude_none=True)
        self.trace_id = trace_id


class MilkProvider:
    async def accounts(self, mobile, session_id, run_id):
        farmers = await fetch_authenticated_farmers(mobile, session_id=session_id, tool_call_id=f"{run_id}:accounts", force_refresh=True)
        return authenticated_accounts(farmers or [])

    async def milk(self, account, period, session_id, run_id):
        return await fetch_milk_collection(account, **period, session_id=session_id, tool_call_id=f"{run_id}:milk")


def new_record():
    return dict(salt=secrets.token_hex(32), runs={}, options={}, period=None,
                ui=dict(version=1, state_revision=time.time_ns() // 1_000_000,
                        selected_milk_account=None, components=[]))


def check_run(record, request):
    digest = hmac.new(bytes.fromhex(record["salt"]), request.model_dump_json().encode(), hashlib.sha256).hexdigest()
    previous = record["runs"].get(request.runId)
    if previous:
        code = "RUN_ALREADY_RECORDED" if previous["fingerprint"] == digest else "RUN_ID_CONFLICT"
        raise UIError(409, code, "Use a fresh runId for a new operation.", trace_id=previous["trace_id"])
    if len(record["runs"]) >= 1000:
        raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")
    return digest


def reconcile_accounts(record, accounts):
    """Apply a successful ownership lookup to this turn's uncommitted record."""
    ui = record["ui"]
    by_identity = {account_identity(a): a for a in accounts}
    options = {k: v for k, v in record["options"].items() if tuple(v["identity"]) in by_identity}
    for identity, account in by_identity.items():
        key = next((k for k, v in options.items() if tuple(v["identity"]) == identity), uuid4().hex)
        options[key] = dict(identity=list(identity), label=account_label(account))
    record["options"] = options
    ui["components"] = [c for c in ui["components"]
                        if c["kind"] != "chart" or c["data"]["account"]["option_id"] in options]
    selected = ui["selected_milk_account"]
    removed = selected is not None and selected["option_id"] not in options
    if removed:
        ui["selected_milk_account"] = None
        record["period"] = None
        record["date_reselection_required"] = False
    elif selected:
        selected["label"] = options[selected["option_id"]]["label"]
    for item in ui["components"]:
        if item["status"] == "pending" and (removed or not options):
            item.update(status="expired", revision=item["revision"] + 1)
    return by_identity, removed


async def advance(record, request, mobile, provider, trace_id=None, *, milk_intent=None, accounts=None):
    """Apply a trusted milk intent or a validated control, never model-written UI."""
    record = copy.deepcopy(record)
    digest = check_run(record, request)
    ui, props = record["ui"], request.forwardedProps
    words = labels(props.target_lang)
    message_id, workflow_id = uuid4().hex, uuid4().hex
    trace_id = trace_id or uuid4().hex
    failure = None

    def finish(text):
        ui["state_revision"] += 1
        UIState.model_validate(ui)
        record["runs"][request.runId] = dict(fingerprint=digest, trace_id=trace_id)
        return record, dict(text=text, message_id=message_id, trace_id=trace_id,
                            state=copy.deepcopy(ui), error=failure)

    if request.messages and milk_intent is None:
        return finish("")  # Ordinary agent replies do not consume component capacity.

    def expire():
        for item in ui["components"]:
            if item["status"] == "pending":
                item.update(status="expired", revision=item["revision"] + 1)

    target = None
    if props.ui_action:
        action = props.ui_action
        target = next((c for c in ui["components"] if c["id"] == action.component_id), None)
        if target is None:
            raise UIError(404, "COMPONENT_NOT_FOUND", "Control expired or is unavailable. Start a fresh chat.")
        if target["status"] != "pending" or target["revision"] != action.revision:
            raise UIError(409, "STALE_COMPONENT", "This control was used or changed.", component_id=target["id"])
        if action.kind != target["kind"] or action.kind == "confirm":
            raise UIError(422, "UNSUPPORTED_ACTION", "This action is not supported.", component_id=target["id"])
        if action.kind == "choice" and action.value.option_id not in {o["id"] for o in target["options"]}:
            raise UIError(422, "INVALID_OPTION", "Choose one of the displayed accounts.", component_id=target["id"])
        workflow_id = target["workflow_id"]
    if props.command and props.command.state_revision != ui["state_revision"]:
        raise UIError(409, "STALE_STATE", "The chat changed. Use its latest state revision.")

    if accounts is None:
        try:
            accounts = await provider.accounts(mobile, request.threadId, request.runId)
        except Exception:
            raise UIError(503, "ACCOUNTS_UNAVAILABLE", "Your accounts are temporarily unavailable.", retryable=True) from None
    by_identity, removed = reconcile_accounts(record, accounts)
    options = record["options"]

    def require_component_capacity():
        if len(ui["components"]) >= MAX_COMPONENTS:
            raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")

    def component(kind, title, **fields):
        require_component_capacity()
        item = dict(id=uuid4().hex, workflow_id=workflow_id, message_id=message_id,
                    revision=1, title=title, status="pending", kind=kind, **fields)
        ui["components"].append(item)
        return item

    def account_picker():
        component("choice", words[0], purpose="milk_account", selected=None,
                  options=[dict(id=k, label=v["label"], description=None) for k, v in options.items()])
        return words[0]

    def dates():
        component("date_range", words[1], purpose="milk_period", selected=None,
                  constraints=dict(date_format="YYYY-MM-DD", max_difference_days=31))
        return words[1]

    async def chart():
        nonlocal failure
        # Reject before the provider read; do not consume a control we cannot render.
        require_component_capacity()
        selected = ui["selected_milk_account"]
        identity = tuple(options[selected["option_id"]]["identity"])
        try:
            result = await provider.milk(by_identity[identity], record["period"], request.threadId, request.runId)
            data = milk_result(selected, record["period"], result)
        except Exception:
            record["date_reselection_required"] = True
            failure = ErrorResponse(error=ErrorDetail(code="MILK_UNAVAILABLE", message=words[3], retryable=True)).model_dump(exclude_none=True)
            return words[3]
        item = component("chart", words[2], purpose="milk_records", data=data)
        item["status"] = "submitted"
        record["date_reselection_required"] = False
        return words[2]

    if not options:
        expire()
        ui["selected_milk_account"] = None
        failure = ErrorResponse(error=ErrorDetail(code="NO_FARMER_ACCOUNT", message="No milk account was found for your signed-in farmer number.")).model_dump(exclude_none=True)
        text = failure["error"]["message"]
    elif props.command:
        expire()
        ui["selected_milk_account"] = None
        if record.get("date_reselection_required"):
            record["period"] = None
        text = account_picker()
    elif request.messages or removed:
        expire()
        record["period"] = milk_intent["period"] if milk_intent and not record.get("date_reselection_required") else None
        if not ui["selected_milk_account"] and len(options) == 1:
            key = next(iter(options))
            ui["selected_milk_account"] = dict(option_id=key, label=options[key]["label"])
        if ui["selected_milk_account"]:
            text = await chart() if record["period"] else dates()
        else:
            text = account_picker()
    elif props.ui_action.kind == "choice":
        key = props.ui_action.value.option_id
        if key not in options:
            expire()
            text = account_picker()
        else:
            ui["selected_milk_account"] = dict(option_id=key, label=options[key]["label"])
            target.update(status="submitted", revision=target["revision"] + 1, selected=props.ui_action.value.model_dump())
            text = await chart() if record["period"] else dates()
    else:
        record["period"] = props.ui_action.value.model_dump()
        target.update(status="submitted", revision=target["revision"] + 1, selected=record["period"])
        text = await chart()

    return finish(text)


class MilkStore:
    def __init__(self, redis, prefix, ttl, provider=None):
        self.redis, self.prefix, self.ttl = redis, prefix, ttl
        self.provider = provider or MilkProvider()

    def key(self, owner, persona, thread):
        digest = hashlib.sha256(json.dumps([owner, persona, thread]).encode()).hexdigest()
        return f"{self.prefix}agui-milk-dev:{digest}"

    async def clear(self, owner, persona, thread):
        try:
            await self.redis.delete(self.key(owner, persona, thread))
        except Exception:
            raise UIError(503, "STATE_UNAVAILABLE", "Chat state is temporarily unavailable.", retryable=True) from None

    async def apply(self, request, owner, persona, mobile, trace_id=None, *, agent_reply=None, disconnected=None):
        key = self.key(owner, persona, request.threadId)
        lock = self.redis.lock(key + ":lock", timeout=120, blocking=False)
        acquired = False
        heartbeat = None

        async def renew():
            while True:
                await asyncio.sleep(30)
                await lock.extend(120, replace_ttl=True)

        try:
            acquired = await lock.acquire()
            if not acquired:
                raise UIError(409, "RUN_IN_PROGRESS", "Another operation is active. Try again later.", retryable=True)
            heartbeat = asyncio.create_task(renew())
            async with self.redis.pipeline(transaction=True) as pipe:
                # Lease renewal changes its TTL. Watch the lease only at commit.
                await pipe.watch(key)
                raw = await pipe.get(key)
                if raw is None and not request.messages:
                    raise UIError(404, "COMPONENT_NOT_FOUND", "Control expired or is unavailable. Start a fresh chat.")
                record = json.loads(raw) if raw else new_record()
                check_run(record, request)  # Reject duplicates before running any agent/tools.
                history, history_key = [], None
                accounts = None
                history_session_id = storage_session_id(owner, persona, request.threadId)
                if request.messages and agent_reply is not None:
                    from pydantic_ai.messages import ModelMessagesTypeAdapter
                    from app.utils import HISTORY_SUFFIX, _sanitize_legacy_usage

                    history_key = f"{self.prefix}{history_session_id}_{HISTORY_SUFFIX}"
                    await pipe.watch(history_key)
                    raw_history = await pipe.get(history_key)
                    try:
                        history = ModelMessagesTypeAdapter.validate_python(_sanitize_legacy_usage(json.loads(raw_history))) if raw_history else []
                    except (ValueError, TypeError):
                        history = []  # Match legacy chat's unreadable-history recovery.
                    if record["options"] or record["ui"]["selected_milk_account"] or any(
                        c["kind"] == "chart" for c in record["ui"]["components"]
                    ):
                        try:
                            accounts = await self.provider.accounts(mobile, request.threadId, request.runId + ":context")
                        except Exception:
                            pass  # Unavailable is not evidence of removal; hide context below.
                        else:
                            reconcile_accounts(record, accounts)

                async def commit(reply=None):
                    if disconnected is not None and await disconnected():
                        raise asyncio.CancelledError()
                    updated, response = await advance(record, request, mobile, self.provider, trace_id,
                                                      milk_intent=reply["intent"] if reply else None, accounts=accounts)
                    encoded = json.dumps(updated, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    if len(encoded.encode()) > 4 * 1024 * 1024:
                        raise UIError(409, "SESSION_LIMIT", "Start a fresh chat session.")
                    if heartbeat.done() and heartbeat.exception() is not None:
                        raise UIError(409, "SESSION_BUSY", "Session lease was lost.", retryable=True)
                    if disconnected is not None and await disconnected():
                        raise asyncio.CancelledError()
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                    await pipe.watch(key + ":lock")
                    if not await lock.owned():
                        raise UIError(409, "SESSION_BUSY", "Operation expired. Start a fresh request.", retryable=True)
                    pipe.multi()
                    pipe.set(key, encoded, ex=self.ttl)
                    if reply:
                        if reply["history"] is not None:
                            pipe.set(history_key, json.dumps(reply["history"], allow_nan=False), ex=self.ttl)
                        if reply["suggestions_pending"]:
                            suggestions_key = f"{self.prefix}suggestions_{history_session_id}_{request.forwardedProps.target_lang}"
                            pipe.set(suggestions_key + ":pending", "true", ex=30)
                            pipe.delete(suggestions_key)
                        response["agent"] = reply
                    await pipe.execute()
                    return response

                if request.messages and agent_reply is not None:
                    return await agent_reply(record, history, commit, history_session_id=history_session_id,
                                             accounts_available=accounts is not None)
                return await commit()
        except UIError:
            raise
        except WatchError:
            raise UIError(409, "SESSION_BUSY", "Chat state changed. Start a fresh request.", retryable=True) from None
        except Exception:
            raise UIError(503, "STATE_UNAVAILABLE", "Chat state is temporarily unavailable.", retryable=True) from None
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            if acquired:
                try:
                    await lock.release()
                except Exception:
                    pass  # Never release another worker's lock or undo a successful commit.


def events(request, response):
    yield RunStartedEvent(thread_id=request.threadId, run_id=request.runId)
    agent = response.get("agent")
    if agent and (agent["text"] or agent["artifacts"]):
        yield TextMessageStartEvent(message_id=agent["message_id"], role="assistant")
        yield TextMessageContentEvent(message_id=agent["message_id"], delta=agent["text"] or agent["artifacts"][0].get("title") or "Results")
        yield TextMessageEndEvent(message_id=agent["message_id"])
    if agent and agent["artifacts"]:
        yield CustomEvent(name="amul.artifacts", value=dict(version=1, message_id=agent["message_id"], artifacts=agent["artifacts"]))
    if response["text"]:
        yield TextMessageStartEvent(message_id=response["message_id"], role="assistant")
        yield TextMessageContentEvent(message_id=response["message_id"], delta=response["text"])
        yield TextMessageEndEvent(message_id=response["message_id"])
    yield StateSnapshotEvent(snapshot={"amul_ui": response["state"]})
    if response["error"]:
        yield CustomEvent(name="amul.ui_error", value={"version": 1, **response["error"]})
        yield RunErrorEvent(message=response["error"]["error"]["message"], code=response["error"]["error"]["code"])
    else:
        yield RunFinishedEvent(thread_id=request.threadId, run_id=request.runId)


def wire_chunks(request, response):
    encoder = EventEncoder()
    for event in events(request, response):
        yield encoder.encode(event)
