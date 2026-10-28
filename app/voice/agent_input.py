"""Voice's agent input: what the voice agent is run with on a call turn.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``), the part of
``stream_voice_message`` between pretranslation and the agent. The farmer data the
background fetched becomes the farmer block, accounts and location. On an
outbound call's consent turn the consent gate runs: a no hangs up with the
farewell, a yes reads out milk details, or says there are none. Then the
FarmerContext, with the running moderation check attached so booking tools wait
for it; history cleaned of orphaned tool calls and trimmed, with the runtime
context before it and the per-query hints after it; and the signed-in agent with
a higher request limit. After the answer, the agent's ``conversation_closing``
signal ends the call.

What voice's trace records along the way is recorded here too: the farmer data
and scheme summary stages, the consent verdict, the milk readout routes, and the
agent's run, with the model that answered and whether it fell back (llm_core
runs the attempts, so the per-attempt list voice kept is not recorded).
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from functools import partial
from typing import Optional

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from agents.deps import FarmerAccount, FarmerContext
from agents.tools.farmer_cache import (
    enqueue_farmer_refresh,
    get_farmer_data_cached_only,
    should_refresh_farmer_data,
)
from agents.voice.agent import voice_agents
from agents.voice.services.farmer_identity import (
    has_usable_farmer_identity,
    identity_state_for_envelope,
    identity_tool_groups,
    profile_status_for_state,
    unavailable_capability_lines,
)
from agents.voice.tools.terms import get_ambiguity_hints_for_query
from app.config import settings
from app.llm_core import Step, runtime
from app.model_boundary_capture import boundary_capture_context
from app.turn.types import AgentInput, AgentInputStep, ClassifierResult, Pretranslated, StalenessCheck, Turn
from app.utils import history_ttl, trim_history, update_message_history
from app.voice import outbound as _outbound
from app.voice.classifiers import TELEPHONY_TERMINATE_CALL_TOKEN, RenderForCaller, _canned_for_caller
from app.voice.farmer import (
    _build_ai_technician_summary,
    _build_compact_farmer_summary,
    _build_union_scheme_summary,
    _collect_farmer_accounts,
    _collect_farmer_location,
    _collect_farmer_unions,
    _is_signed_in_session,
)
from app.voice.history import clean_message_history_for_openai, history_pair
from app.voice.liveness import nudge_stopped
from app.voice.trace import check_mode, current_trace, step_tiers
from helpers.utils import get_logger, get_today_date_str

logger = get_logger(__name__)


def _runtime_context_message(deps: FarmerContext) -> str:
    """Compact runtime context that stays outside the static system prompt.

    Voice's ``FarmerContext.get_runtime_context_message``, for this repo's
    FarmerContext, whose identity predicate lives in farmer_identity.
    """
    lines = [
        "Runtime context for this turn:",
        f"- Signed-in session: {'yes' if deps.signed_in else 'no'}",
        f"- Normalized mobile available: {'yes' if deps.mobile else 'no'}",
    ]
    if deps.mobile:
        lines.append(f"- Normalized mobile: {deps.mobile}")
    if deps.farmer_unions:
        lines.append(f"- Farmer unions: {', '.join(deps.farmer_unions)}")
    lines.append("- Core loop language: English")
    # Keyed on resolved identity, not on signed_in: a signed-in caller whose
    # lookup timed out has no usable identity, and this line used to promise
    # farmer-data tools on exactly those turns (issue #282).
    if has_usable_farmer_identity(deps):
        lines.append("- Farmer-data tools may be available for this turn.")
    else:
        lines.append("- Farmer-data tools are not available for this turn.")
    if deps.farmer_info:
        lines.append("- Farmer context summary:")
        lines.append(deps.farmer_info)
    if deps.ai_technician_info:
        lines.append("- Internal AI technician context for booking:")
        lines.append("The caller does not know which AI technicians are available unless you tell them by name.")
        lines.append(deps.ai_technician_info)
    return "\n".join(lines)


_COMPARISON_PATTERNS = (
    r"\bdifference between\b",
    r"\bwhat is the difference\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.\b",
    r"\bdifference\b",
)


_EXPLAINER_PATTERNS = (
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\btell me about\b",
    r"\bexplain\b",
    r"\bmeaning of\b",
)


_SYMPTOM_PATTERNS = (
    r"\bfever\b",
    r"\bnot eating\b",
    r"\bnot come in heat\b",
    r"\bnot coming in heat\b",
    r"\bbleeding\b",
    r"\bdiarrhea\b",
    r"\bloose motion\b",
    r"\bcough\b",
    r"\bbloat\b",
    r"\bmastitis\b",
    r"\bpregnant\b",
    r"\bcalving\b",
    r"\bsick\b",
)


def _voice_answer_mode_for_query(text: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip().lower()
    if not cleaned:
        return None
    if any(re.search(pattern, cleaned) for pattern in _COMPARISON_PATTERNS):
        return "compact_comparison"
    if any(re.search(pattern, cleaned) for pattern in _EXPLAINER_PATTERNS):
        return "compact_explainer"
    if any(re.search(pattern, cleaned) for pattern in _SYMPTOM_PATTERNS):
        return "action_first_symptom"
    return None


def _build_runtime_context_request(deps: FarmerContext) -> ModelRequest:
    """Stable per-call context (constant across a call's turns: date, farmer
    profile, signed-in state, tool groups). Placed BEFORE history so the token
    sequence [system][stable-context][history] stays a single growing prefix that
    vLLM prefix caching can reuse across turns. Per-query content that changes
    every turn lives in _build_query_hints_request() and is appended AFTER history
    so it never breaks this prefix."""
    # Derived from the same predicate that drives the tool gates, so this line
    # can never advertise a group the model was not given. It previously
    # hardcoded "booking", announcing it on exactly the turns where booking was
    # impossible (issue #282).
    tool_groups = identity_tool_groups(deps)
    runtime_context = _runtime_context_message(deps)
    context_lines = [
        "Runtime context for this turn:",
        f"- Today date: {get_today_date_str()}",
        runtime_context.replace("Runtime context for this turn:\n", "", 1),
        f"- Tool groups in this run: {', '.join(tool_groups)}",
    ]
    return ModelRequest(parts=[UserPromptPart(content="\n".join(context_lines))])


def _heat_alert_hint(call) -> str:
    """This turn is the farmer's answer to Sarlaben's heat-alert call."""
    account = (
        f"union {call.union_code}, society {call.society_code}, farmer code {call.farmer_code}"
        if call.union_code and call.society_code and call.farmer_code
        else "the farmer's account from the farmer context"
    )
    species = (
        f"The animal is a {call.species} (from the farmer's animal records): do not ask whether it is a cow or "
        f"a buffalo, use species {call.species} in create_ai_call."
        if call.species
        else "Ask whether the animal is a cow or a buffalo if they have not said, then call create_ai_call."
    )
    said = f" Sarlaben has already said, word for word: «{call.script}»." if call.script else ""
    return (
        "- This is Sarlaben's outbound heat-alert call, not a call the farmer made. The digital "
        f"collar on their animal (animal number {call.animal_no or 'unknown'}, tag {call.tag_no or 'unknown'}) "
        "detected heat signs, and the farmer has just been asked whether to register an artificial "
        f"insemination (AI) ticket for it.{said} The conversation continues from there: the current "
        "message is the farmer's reply, and earlier replies are in the history.\n"
        f"- If they agree: run the AI booking flow for {account}. {species}\n"
        "- If they decline or are unsure: thank them briefly, say they can book later through the "
        "WhatsApp link or by calling, and end the call.\n"
        "- Do not ask what they are calling about and do not offer milk details."
    )


def _build_query_hints_request(deps: FarmerContext) -> Optional[ModelRequest]:
    """Per-query hints derived from the caller's current utterance: disambiguation
    rules (for ambiguous terms) and the voice answer mode. Kept OUT of the stable
    pre-history context — these change every turn, so placing them before history
    would break the cacheable prefix. Appended right before the user message
    instead, where the instructions also sit closest to the query they describe.
    Returns None when the query triggers neither hint."""
    hint_lines: list[str] = []
    # Inject ambiguity hints for the agent so it can decide to clarify vs. answer
    ambiguity_hints = get_ambiguity_hints_for_query(
        deps.query or "",
        threshold=settings.ambiguity_match_threshold,
    )
    if ambiguity_hints:
        hint_lines.append(f"- Disambiguation rules for terms in this query:\n{ambiguity_hints}")
    answer_mode = _voice_answer_mode_for_query(deps.query or "")
    if answer_mode == "compact_comparison":
        hint_lines.append(
            "- Voice answer mode: compact comparison. Give one short contrast sentence, then at most one short practical takeaway. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "compact_explainer":
        hint_lines.append(
            "- Voice answer mode: compact explainer. Give one short plain-language definition or explanation, then at most one short practical takeaway. Do not teach the full topic. Do not enumerate. Do not use labels, colons, or list structure. Do not append an extra follow-up question unless required."
        )
    elif answer_mode == "action_first_symptom":
        hint_lines.append(
            "- Voice answer mode: action-first symptom response. Start with the most useful immediate action in one short sentence. Add at most one short safety or escalation sentence. Do not give long background, multiple causes, or a symptom checklist unless asked."
        )
    if not hint_lines:
        return None
    return ModelRequest(parts=[UserPromptPart(content="\n".join(["Hints for the current user query:", *hint_lines]))])


def _conversation_closing(new_messages) -> bool:
    """Whether the agent called signal_conversation_state("conversation_closing").

    Read off its new messages rather than a contextvar, because pydantic-ai runs
    tools in child tasks whose contextvar writes don't propagate back.
    """
    return any(
        getattr(part, "tool_name", None) == "signal_conversation_state"
        and "conversation_closing" in (getattr(part, "args_as_json_str", lambda: "")() if callable(getattr(part, "args_as_json_str", None)) else str(getattr(part, "args", "")))
        for msg in new_messages
        for part in (getattr(msg, "parts", None) or [])
    )


def _goodbye_after_closing(target_lang: str, session_id: str, process_id: Optional[str], new_messages) -> Optional[str]:
    """The termination token, so the telephony provider disconnects the call,
    when the agent said the conversation is closing."""
    if not _conversation_closing(new_messages):
        return None
    logger.info(
        "Appending goodbye after conversation_closing signal; session_id=%s process_id=%s",
        session_id, process_id,
    )
    return " " + TELEPHONY_TERMINATE_CALL_TOKEN.get(target_lang, TELEPHONY_TERMINATE_CALL_TOKEN["en"])


def _agent_text(new_messages) -> str:
    """The text the agent streamed, as its new messages hold it."""
    return "".join(
        part.content
        for message in new_messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart)
    ).strip()


def _answered_by(execution) -> dict:
    """What voice-oan-api recorded about the tier that answered: the one the
    agent's stream committed to ("failed" as actual when none did), and whether
    an earlier tier failed first."""
    try:
        agent = execution.begin_trace().steps.get(Step.AGENT.value)
    except Exception as exc:  # tracing never breaks the turn
        logger.debug("Voice agent tier lookup failed: %s", exc)
        return {}
    committed = agent.tier_committed if agent is not None else None
    tier, provider, model = (
        (committed.kind, committed.provider, committed.model_name) if committed is not None else (None, None, None)
    )
    return {
        "actual_tier": tier or "failed",
        "actual_provider": provider or "failed",
        "actual_model": model or "failed",
        "first_token_committed_tier": tier,
        "first_token_committed_provider": provider,
        "first_token_committed_model": model,
        "fallback_used": bool(agent is not None and agent.fell_back),
    }


class _AgentRun:
    """The voice agent's run as the trace records it: timed from when it starts
    streaming, and what it produced once its answer is out."""

    def __init__(
        self,
        *,
        execution,
        signed_in: bool,
        request_limit: int,
        session_id: str,
        process_id: Optional[str],
        user_query: str,
    ) -> None:
        self._execution = execution
        self._signed_in = signed_in
        self._request_limit = request_limit
        self._session_id = session_id
        self._process_id = process_id
        self._user_query = user_query
        self._step = None

    @contextmanager
    def observe(self):
        """turn.v1's agent step: the English question in, the English answer out
        (``after_run`` writes it, inside this), pydantic-ai's spans under it.
        ``run_turn`` is handed None so the step's output stays voice's."""
        self._started_at = time.monotonic()
        trace = current_trace()
        with trace.observation("agent", as_type="agent", input=trace.text(self._user_query)) as step:
            self._step = step
            with boundary_capture_context(
                session_id=self._session_id,
                process_id=self._process_id,
                user_query=self._user_query,
            ):
                yield None

    def after_run(self, new_messages) -> None:
        trace = current_trace()
        agent = self._execution.info(Step.AGENT)
        answered_by = _answered_by(self._execution)
        if self._step is not None:
            self._step.record(
                output=trace.text(_agent_text(new_messages)),
                metadata=step_tiers(self._execution, Step.AGENT),
            )
        trace.attach_stage_timing(
            "agent",
            (time.monotonic() - self._started_at) * 1000.0,
            signed_in=self._signed_in,
            request_limit=self._request_limit,
            pipeline_profile=self._execution.profile_name,
            requested_tier=agent.kind,
            requested_model=agent.model_name,
            requested_provider=agent.provider,
            **answered_by,
        )
        trace.set_agent(
            signed_in=self._signed_in,
            output=_agent_text(new_messages),
            new_messages=list(new_messages),
            requested_tier=agent.kind,
            requested_provider=agent.provider,
            requested_model=agent.model_name,
            **answered_by,
        )


async def _voice_agent_input(
    turn: Turn,
    pretranslated: Pretranslated,
    *,
    execution,
    scheduler,
    translate_to: Optional[str],
    background,
    is_stale: Optional[StalenessCheck],
    render: RenderForCaller,
) -> AgentInput | ClassifierResult:
    session_id = turn.session_id
    process_id = turn.call.process_id if turn.call is not None else None
    requested_target_lang = (turn.target_lang or "gu").strip().lower()
    processing_query, processing_lang = pretranslated.query, pretranslated.lang
    history = list(turn.history)
    mobile = background.mobile
    signed_in = _is_signed_in_session(turn.authenticated_user, turn.user_id)

    # Fails closed: a turn whose farmer lookup threw, or never resolved, keeps a
    # negative state, so the identity-taking tools stay hidden. "anonymous" when
    # there is no mobile to look anything up with; "unresolved" when a lookup
    # was attempted and may still fail.
    farmer_identity = "unresolved" if mobile else "anonymous"
    farmer_info = "\n".join(unavailable_capability_lines(farmer_identity))
    farmer_unions: list[str] = []
    farmer_accounts: list[FarmerAccount] = []
    farmer_village: Optional[str] = None
    farmer_district: Optional[str] = None
    ai_technician_info = ""
    trace = current_trace()
    if mobile:
        try:
            # turn.v1's farmer_context step: the farmer block built for the agent,
            # from the fetch to the scheme summary, with the block as its output.
            with trace.stage("farmer_context") as farmer_step:
                envelope = await background.farmer_data()
                if envelope is None:
                    # A concurrent fetch (e.g. the outbound prefetch) may have
                    # landed while ours was giving up. Cheap Redis re-read
                    # before we commit to an unresolved turn.
                    envelope = await get_farmer_data_cached_only(mobile)
                    if envelope is not None:
                        logger.info("Farmer context resolved on re-read; session_id=%s", session_id)
                # Scored AFTER the re-read: a recovered envelope must not be
                # graded UNRESOLVED and have the identity tools withheld.
                farmer_identity = identity_state_for_envelope(envelope)
                farmer_info = _build_compact_farmer_summary(envelope)
                farmer_unions = _collect_farmer_unions(envelope)
                farmer_accounts = _collect_farmer_accounts(envelope)
                farmer_village, farmer_district = _collect_farmer_location(envelope)
                with trace.stage("scheme_summary"):
                    scheme_summary = await _build_union_scheme_summary(farmer_unions)
                if scheme_summary:
                    farmer_info = f"{farmer_info}\n{scheme_summary}" if farmer_info else scheme_summary
                ai_technician_info = _build_ai_technician_summary(envelope)
                farmer_step.record(
                    output=trace.set_farmer_context(
                        source=getattr(envelope, "source", None) if envelope else None,
                        stale=getattr(envelope, "stale", None) if envelope else None,
                        unions=farmer_unions,
                        farmer_info_chars=len(farmer_info),
                        technician_info_chars=len(ai_technician_info),
                    )
                )
            logger.info(
                "Farmer summary loaded from cache; session_id=%s source=%s stale=%s unions=%s summary_chars=%s technician_chars=%s",
                session_id,
                getattr(envelope, "source", None) if envelope else None,
                getattr(envelope, "stale", None) if envelope else None,
                farmer_unions,
                len(farmer_info),
                len(ai_technician_info),
            )
            if mobile and should_refresh_farmer_data(envelope):
                await enqueue_farmer_refresh(mobile)
                logger.info(
                    "Farmer cache refresh scheduled in background; session_id=%s stale=%s status=%s",
                    session_id,
                    getattr(envelope, "stale", None) if envelope else None,
                    getattr(envelope, "lookupStatus", None) if envelope else None,
                )
        except Exception as e:
            logger.warning("Failed to load farmer summary; session_id=%s error=%s", session_id, type(e).__name__)

    # ── Outbound consent gate ─────────────────────────────────────
    # Turn 2 of an outbound call: the farmer's first reply to the scripted
    # consent question. Three-way — a reply that is neither yes nor no
    # (typically the farmer asking their own question) simply falls through
    # to a normal agent turn, which is the outcome we most want to protect.
    outbound_milk_hint: Optional[str] = None
    _consent_wait_started = time.monotonic()
    consent_verdict = await background.consent()
    if consent_verdict is not None:
        trace.attach_stage_timing(
            "outbound_consent",
            (time.monotonic() - _consent_wait_started) * 1000.0,
            intent=consent_verdict.intent,
            failed_open=consent_verdict.failed_open,
        )
        trace.metadata["outbound_consent_intent"] = consent_verdict.intent
        # turn.v1's outbound_consent step: the caller's answer to the opening
        # question. The classifier runs on the non_meaningful step's tier.
        trace.record_child_observation(
            name="outbound_consent",
            as_type="generation",
            output={"intent": consent_verdict.intent},
            metadata={
                "amul.check_mode": check_mode(failed_open=consent_verdict.failed_open),
                **step_tiers(execution, Step.NON_MEANINGFUL),
            },
        )
        logger.info(
            "Outbound consent verdict - session_id=%s process_id=%s intent=%s reason=%r failed_open=%s",
            session_id, process_id, consent_verdict.intent,
            consent_verdict.reason, consent_verdict.failed_open,
        )
        # The opener is done either way: never re-classify on later turns.
        await _outbound.set_stage(session_id, _outbound.STAGE_RESOLVED)

        if consent_verdict.is_negative:
            nudge_stopped("outbound_declined")
            farewell_en = _outbound.OUTBOUND_DECLINE_FAREWELL["en"]
            farewell_for_caller = await _canned_for_caller(
                render, farewell_en, requested_target_lang, _outbound.OUTBOUND_DECLINE_FAREWELL,
                execution=execution,
            )
            logger.info(
                "Outbound consent declined; emitting farewell + hangup - session_id=%s process_id=%s reason=%r",
                session_id, process_id, consent_verdict.reason,
            )
            # Termination token stays exact ASCII "Goodbye." — passing it
            # through the Gujarati output filter would strip it to ".".
            return ClassifierResult(
                canned_text=farewell_for_caller,
                label="outbound_declined",
                history_pair=history_pair(
                    background.history_text or turn.query,
                    f"{farewell_en} {TELEPHONY_TERMINATE_CALL_TOKEN['en']}",
                ),
                outcome="outbound_declined",
                raw_tail=" " + TELEPHONY_TERMINATE_CALL_TOKEN.get(
                    requested_target_lang, TELEPHONY_TERMINATE_CALL_TOKEN["en"],
                ),
            )

        if consent_verdict.is_affirmative:
            prefetched = await _outbound.get_prefetched_milk_summary(session_id)
            if prefetched:
                outbound_milk_hint = _outbound.milk_answer_hint(prefetched)
                trace.set_route("outbound_milk_readout")
            elif signed_in and mobile and farmer_accounts:
                # Prefetch missed (cold cache or slow upstream) — the agent
                # fetches it itself against the same pinned window.
                _from, _to = _outbound.milk_window(settings.outbound_milk_window_days)
                outbound_milk_hint = _outbound.milk_fetch_hint(_from, _to)
                trace.set_route("outbound_milk_readout_cold")
            else:
                # Consented, but there is nothing to read out (no account on
                # this number). Say so and leave the call open rather than
                # reading an upstream failure message aloud.
                no_data_en = _outbound.OUTBOUND_NO_DATA["en"]
                no_data_for_caller = await _canned_for_caller(
                    render, no_data_en, requested_target_lang, _outbound.OUTBOUND_NO_DATA,
                    execution=execution,
                )
                logger.info(
                    "Outbound consent affirmative but no milk data available - "
                    "session_id=%s process_id=%s signed_in=%s accounts=%s",
                    session_id, process_id, signed_in, len(farmer_accounts),
                )
                return ClassifierResult(
                    canned_text=no_data_for_caller,
                    label="outbound_no_data",
                    history_pair=history_pair(background.history_text or turn.query, no_data_en),
                    outcome="outbound_no_data",
                )

    deps = FarmerContext(
        query=processing_query,
        lang_code=processing_lang,
        target_lang=requested_target_lang,
        session_id=session_id,
        process_id=process_id,
        farmer_info=farmer_info,
        farmer_unions=farmer_unions,
        ai_technician_info=ai_technician_info,
        signed_in=signed_in,
        mobile=mobile,
        farmer_profile_status=profile_status_for_state(farmer_identity),
        farmer_accounts=farmer_accounts,
        farmer_village=farmer_village,
        farmer_district=farmer_district,
        heat_alert_event_id=(
            turn.call.heat_alert.event_id if turn.call is not None and turn.call.heat_alert else None
        ),
    )
    # Let side-effecting tools (bookings) self-gate on the concurrent
    # moderation verdict before performing any write.
    deps.set_moderation_task(background.moderation_task)

    user_message = deps.get_user_message()
    runtime_context_request = _build_runtime_context_request(deps)
    # Sizes only: the conversation and the model's message carry the farmer's
    # words and details, which stay out of application logs.
    logger.info(
        "Running voice agent; session_id=%s process_id=%s history_messages=%s user_message_chars=%s",
        session_id, process_id, len(history), len(user_message),
    )

    cleaned_history = clean_message_history_for_openai(history)
    if len(cleaned_history) != len(history):
        logger.warning(f"Cleaned {len(history) - len(cleaned_history)} orphaned tool calls from history")
        if is_stale is None or await is_stale("before_cleaned_history_write") is None:
            await update_message_history(turn.history_session_id, cleaned_history, **history_ttl("voice"))
        history = cleaned_history

    trimmed_history = trim_history(
        history,
        max_tokens=32_000,
        include_system_prompts=False,
        include_tool_calls=True,
    )
    logger.info(f"Trimmed history length: {len(trimmed_history)} messages")
    # Stable context first → [system][stable-context][history] is a single
    # growing prefix vLLM can cache across turns. Per-query hints (if any)
    # go last, right before the user message, so they never break it.
    model_input_history = [runtime_context_request, *trimmed_history]
    query_hints_request = _build_query_hints_request(deps)
    if query_hints_request is not None:
        model_input_history.append(query_hints_request)
    if turn.call is not None and turn.call.heat_alert is not None:
        model_input_history.append(
            ModelRequest(parts=[UserPromptPart(content=(
                "Hints for the current user query:\n" + _heat_alert_hint(turn.call.heat_alert)
            ))])
        )
    if outbound_milk_hint is not None:
        # Appended after the per-query hints, immediately before the user
        # message: this turn's instruction is the readout, and it must sit
        # closest to the reply it acts on.
        model_input_history.append(
            ModelRequest(parts=[UserPromptPart(content=(
                "Hints for the current user query:\n" + outbound_milk_hint
            ))])
        )
    voice_agent, voice_agent_signed_in = voice_agents(runtime.surface_settings("voice").agent_prompt_version)
    active_agent = voice_agent_signed_in if (signed_in and mobile) else voice_agent
    usage_limits = UsageLimits(request_limit=6 if (signed_in and mobile) else 4)
    run = _AgentRun(
        execution=execution,
        signed_in=bool(signed_in and mobile),
        request_limit=usage_limits.request_limit,
        session_id=session_id,
        process_id=process_id,
        user_query=processing_query,
    )

    if settings.retrieval_audit_log:
        logger.info(
            "RETRIEVAL_AUDIT query=%r session_id=%s process_id=%s target_lang=%s",
            processing_query,
            session_id,
            process_id,
            requested_target_lang,
        )

    return AgentInput(
        agent=active_agent,
        prompt=user_message,
        message_history=model_input_history,
        deps=deps,
        history=history,
        observe=run.observe,
        usage_limits=usage_limits,
        closing_line=partial(_goodbye_after_closing, requested_target_lang, session_id, process_id),
        after_run=run.after_run,
    )


def voice_agent_input(render: RenderForCaller) -> AgentInputStep:
    """Voice's agent input for ``SurfaceProfile.agent_input``.

    ``render`` puts an English line into the caller's language, as it does for
    the classifiers, when there is no pinned copy for it.
    """
    return partial(_voice_agent_input, render=render)
