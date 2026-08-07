"""The chat surface's population of the seam structures.

Chat is the degenerate case of nearly every structure — one classifier where
voice has six, one background consumer where voice has four, no liveness
channel at all. That is the point: the structures are shaped so voice folds in
as a fuller population of the same things, not as a second orchestrator.
"""
from __future__ import annotations

from typing import Optional

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.services.identity_profile import build_identity_profile_table, is_identity_query
from app.turn.types import ClassifierResult, SurfaceProfile, Surface, Turn


async def identity_classifier(turn: Turn) -> Optional[ClassifierResult]:
    """"Who are you?" — answered from a template, without the agent.

    Async despite doing no I/O: the chain is uniform, and voice's classifiers
    (STT signal, consent, farmer identity) do await.
    """
    if not is_identity_query(turn.query):
        return None
    response = build_identity_profile_table(
        source_lang=turn.source_lang,
        target_lang=turn.target_lang,
        query=turn.query,
    )
    return ClassifierResult(
        canned_text=response,
        # One of only two chat exit paths that persist history (the other is
        # normal completion) -- see docs/channel-seam-design.md.
        history_pair=(
            ModelRequest(parts=[UserPromptPart(content=turn.query)]),
            ModelResponse(parts=[TextPart(content=response)]),
        ),
    )


CHAT = SurfaceProfile(
    surface=Surface.CHAT,
    classifiers=(identity_classifier,),
)
