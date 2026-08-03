"""The AGENT_ACTIVITY sentinel — the guard against duplicate side-effecting tools.

pydantic-ai emits a tool-call part BEFORE it runs the tools and long before the
first text delta. That first model event is forwarded as AGENT_ACTIVITY, which
satisfies the TTFT deadline. Without it, a turn already executing a slow tool
(the 20s milk-collection call) trips the deadline, the walker swaps tier, and the
side-effecting tools run a SECOND time — duplicate CreateAICall bookings and SMS.

Mutation testing found all of this unguarded: disabling the sentinel left the
whole suite green. These tests fail if it stops working.
"""
import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.fallback import AGENT_ACTIVITY, with_first_token_deadline


class _Attempt:
    kind = "oss"
    endpoint = "http://oss:8020/v1"
    timeout = 0.20          # a tight TTFT deadline
    model = "gemma-test"
    provider = "vllm"


async def _collect(agen):
    return [c async for c in agen]


def test_sentinel_satisfies_the_deadline_when_text_arrives_late():
    """The scenario it exists for: tools start immediately, text comes much later."""
    async def slow_tool_turn():
        yield AGENT_ACTIVITY            # tool-call event, immediately
        await asyncio.sleep(0.45)       # tools run — well past the 0.20s deadline
        yield "Your milk collection for last month was 320 litres."

    out = asyncio.run(_collect(with_first_token_deadline(_Attempt(), slow_tool_turn())))
    assert out == ["Your milk collection for last month was 320 litres."]


def test_sentinel_is_never_forwarded_to_the_caller():
    async def turn():
        yield AGENT_ACTIVITY
        yield "answer"

    out = asyncio.run(_collect(with_first_token_deadline(_Attempt(), turn())))
    assert AGENT_ACTIVITY not in out
    assert out == ["answer"]


def test_a_hung_endpoint_still_trips_the_deadline():
    """Liveness must survive: no sentinel means no commit, so the swap still happens."""
    async def hung():
        await asyncio.sleep(0.45)
        yield "never reached"

    with pytest.raises(TimeoutError):
        asyncio.run(_collect(with_first_token_deadline(_Attempt(), hung())))


def test_real_first_token_also_satisfies_the_deadline():
    async def prompt_text():
        yield "fast"
        await asyncio.sleep(0.45)
        yield " tail"

    out = asyncio.run(_collect(with_first_token_deadline(_Attempt(), prompt_text())))
    assert out == ["fast", " tail"]
