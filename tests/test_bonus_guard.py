import asyncio
import inspect
from types import SimpleNamespace

from agents.tools import bonus as bonus_tool


def _ctx(unions, mobile="9000000000"):
    return SimpleNamespace(
        deps=SimpleNamespace(
            farmer_unions=unions,
            mobile=mobile,
            session_id="session",
        )
    )


def test_prepare_hides_tool_without_resolved_farmer_or_authenticated_mobile():
    sentinel = object()
    assert (
        asyncio.run(bonus_tool.prepare_get_farmer_bonus_amount(_ctx([]), sentinel))
        is None
    )
    assert (
        asyncio.run(
            bonus_tool.prepare_get_farmer_bonus_amount(_ctx(["banas"], None), sentinel)
        )
        is None
    )


def test_prepare_shows_tool_for_authenticated_resolved_farmer():
    sentinel = object()
    assert (
        asyncio.run(
            bonus_tool.prepare_get_farmer_bonus_amount(_ctx(["banas"]), sentinel)
        )
        is sentinel
    )


def test_model_facing_signature_has_only_ctx():
    parameters = inspect.signature(bonus_tool.get_farmer_bonus_amount).parameters
    assert list(parameters) == ["ctx"]
    assert "union_code" not in parameters
    assert "society_code" not in parameters
    assert "farmer_code" not in parameters


def test_tools_registry_includes_bonus_with_prepare():
    from agents.tools import TOOLS
    from agents.tools.bonus import (
        get_farmer_bonus_amount,
        prepare_get_farmer_bonus_amount,
    )

    registered = [t for t in TOOLS if t.function is get_farmer_bonus_amount]
    assert len(registered) == 1
    assert registered[0].prepare is prepare_get_farmer_bonus_amount
    assert registered[0].takes_ctx is True
