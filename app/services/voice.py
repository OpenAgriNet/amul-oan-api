"""The voice surface, and the voice transport's adapter over ``run_turn``.

``stream_voice_message`` is what voice-oan-api's ``stream_voice_message``
(``app/services/voice.py`` at amul-dev ``3b19835``) did around the turn: work out
whether this is an outbound call's consent turn, wire in the call's staleness
check and the telephony nudge, and release the session once the turn is over.
The turn itself is ``run_turn`` with voice's pieces.
"""
from __future__ import annotations

import dataclasses
from contextlib import aclosing
from functools import partial
from types import MappingProxyType
from typing import AsyncGenerator, Optional

from fastapi import BackgroundTasks, Request

from app.config import settings
from app.services.chat import _BackgroundTasksScheduler, run_turn
from app.turn.types import (
    AgentActivityEmission,
    ArtifactEmission,
    Emission,
    SideChannelEmission,
    Surface,
    SurfaceProfile,
    TelephonyCall,
    TextEmission,
    Turn,
)
from app.utils import SessionRequestOwner, release_session_request_ownership
from app.voice import outbound as _outbound
from app.voice.agent_input import voice_agent_input
from app.voice.background import voice_background
from app.voice.classifiers import voice_classifiers
from app.voice.liveness import CallStaleness, RayaNudgeSender, VoiceLiveness
from app.voice.pretranslation import voice_pretranslation
from app.voice.sink import VoiceSink, _prepare_voice_output, render_for_caller
from app.voice.telemetry import RouteTimings, VoiceTelemetry
from helpers.utils import get_logger

logger = get_logger(__name__)

#: The voice surface's population of the seam.
VOICE_SURFACE = SurfaceProfile(
    surface=Surface.VOICE,
    classifiers=voice_classifiers(render_for_caller),
    sink=VoiceSink,
    telemetry=VoiceTelemetry,
    background=voice_background(render_for_caller),
    liveness=VoiceLiveness,
    pretranslation=voice_pretranslation(render_for_caller),
    agent_input=voice_agent_input(render_for_caller),
)


async def _is_outbound_consent_turn(session_id: str, call_type: str, history: list) -> bool:
    """Whether this turn is the farmer's reply to the provider's opening question.

    The stage is read from Redis (not history) so it survives trimming, and is
    read on every turn while outbound calls are on, because the provider is not
    guaranteed to re-stamp call_type after the first request.
    """
    if not settings.outbound_intro_enabled:
        return False
    stage = await _outbound.get_stage(session_id)
    return (
        (_outbound.is_outbound(call_type) and stage is None and not history)
        # Sessions opened by the removed in-app intro that are still mid-call
        # are past the intro, so their next turn is the consent reply.
        or stage == _outbound.STAGE_INTRO_SENT
    )


async def stream_voice_message(
    *,
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    user_id: str,
    history: list,
    user_info: dict,
    provider: Optional[str],
    process_id: Optional[str],
    call_type: str,
    owner: Optional[SessionRequestOwner],
    http_request: Optional[Request],
    background_tasks: BackgroundTasks,
    route: Optional[RouteTimings] = None,
) -> AsyncGenerator[str, None]:
    """The voice transport's adapter over ``run_turn``.

    Each emission goes out as voice sent it: through voice's output normalizer,
    except a raw one (the hang-up token, which must stay exact). The nudge goes
    out through the telephony provider, not this stream. The session is released
    however the turn ends.
    """
    call_type = _outbound.normalize_call_type(call_type)
    target = (target_lang or "gu").strip().lower()
    try:
        consent_turn = await _is_outbound_consent_turn(session_id, call_type, history)
        if consent_turn:
            logger.info(
                "Outbound consent turn; session_id=%s process_id=%s query_chars=%s",
                session_id, process_id, len(query or ""),
            )
        turn = Turn(
            query=query,
            session_id=session_id,
            source_lang=source_lang,
            target_lang=target_lang,
            user_id=user_id,
            authenticated_user=MappingProxyType(dict(user_info or {})),
            history=tuple(history),
            history_session_id=session_id,
            persona="farmer",
            call=TelephonyCall(
                process_id=process_id,
                outbound_consent_turn=consent_turn,
                provider=provider,
                call_type=call_type,
            ),
        )
        surface = dataclasses.replace(VOICE_SURFACE, telemetry=partial(VoiceTelemetry, route=route))
        turn_stream = run_turn(
            turn,
            surface,
            scheduler=_BackgroundTasksScheduler(background_tasks),
            side_channel=RayaNudgeSender(session_id=session_id, process_id=process_id),
            is_stale=CallStaleness(
                http_request=http_request, owner=owner, session_id=session_id, process_id=process_id
            ),
        )
        # aclosing: a hang-up closes the turn now, so it records the disconnect
        # and releases the session rather than waiting for the event loop.
        async with aclosing(turn_stream) as emissions:
            async for emission in emissions:
                try:
                    chunks = _render_voice_emission(emission, target)
                except Exception as exc:
                    # Rendering failed: the turn failed, the caller did not leave.
                    await emissions.athrow(exc)
                    raise
                for chunk in chunks:
                    yield chunk
    finally:
        released = await release_session_request_ownership(owner)
        if owner is not None:
            logger.info(
                "Session ownership released - session_id=%s process_id=%s epoch=%s released=%s",
                session_id,
                process_id,
                owner.epoch,
                released,
            )


def _render_voice_emission(emission: Emission, target_lang: str) -> tuple[str, ...]:
    """What one emission becomes on the voice wire: zero or one chunk."""
    if isinstance(emission, TextEmission):
        if emission.raw:
            return (emission.text,)
        return (_prepare_voice_output(emission.text, target_lang),)
    if isinstance(emission, (AgentActivityEmission, SideChannelEmission, ArtifactEmission)):
        # The nudge goes through the provider's endpoint, the commit signal is
        # llm_core's, and a private document is never spoken.
        return ()
    raise TypeError(f"voice adapter cannot render emission {emission!r}")
