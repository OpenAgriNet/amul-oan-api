"""The farmer context is kept off the chat turn's critical path.

Chat starts loading the caller's farmer context (Beckn reads) as soon as the
classifiers have passed, so it runs alongside pretranslation and moderation,
and waits for it only where the agent's deps need it. A turn that ends before
then cancels it. Inside the load, every account's AI-technician search runs at
once instead of one account after another, and the markdown keeps account order.
"""
import asyncio
import os
import time
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import BackgroundTasks

from agents import farmer_context
from agents.farmer_context import FarmerContextBundle
from agents.tools.beckn.amul import AITechnicianRecord
from agents.tools.models.farmer import FarmerModel
from app.services import chat as chat_service
from tests.test_chat_turn_contract import _Run, patch_turn


# ── the chat turn ───────────────────────────────────────────────────────────


def _drive(monkeypatch, *, farmer_s, pretranslation_s, moderation_s, category="valid_agricultural"):
    """One signed-in Gujarati chat turn with timed stand-ins for the slow stages."""
    seen = patch_turn(monkeypatch)
    seen["farmer"] = []

    async def _farmer(_phone):
        seen["farmer"].append("started")
        try:
            await asyncio.sleep(farmer_s)
        except asyncio.CancelledError:
            seen["farmer"].append("cancelled")
            raise
        seen["farmer"].append("done")
        return FarmerContextBundle(markdown="farmer data", found=True)

    async def _pretranslate(_tier, *, text, **_kw):
        await asyncio.sleep(pretranslation_s)
        return "How much water should I give my cow?"

    async def _moderate(user_message, model=None):
        await asyncio.sleep(moderation_s)
        return SimpleNamespace(output=SimpleNamespace(category=category, action="Please ask about farming."))

    def _agent_iter(**kw):
        seen["agent_started"] = time.monotonic()
        seen["deps"].append(kw["deps"])
        return _Run(["Give clean water daily."])

    async def _no_shc(*_a):
        return None

    async def _as_is(text, **_kw):
        return text

    monkeypatch.setattr(chat_service, "get_farmer_context_bundle_by_mobile", _farmer)
    monkeypatch.setattr(chat_service, "pretranslate_with_tier", _pretranslate)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderate)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _agent_iter)
    monkeypatch.setattr(chat_service, "get_session_shc_context", _no_shc)
    monkeypatch.setattr(chat_service, "_localize_system_text", _as_is)

    async def _go():
        started = time.monotonic()
        chunks = [chunk async for chunk in chat_service.stream_chat_messages(
            query="મારી ગાયને કેટલું પાણી આપવું?",
            session_id="farmer-context-overlap",
            source_lang="gu",
            target_lang="gu",
            channel="web",
            user_id="9876543210",
            history=[],
            user_info={"phone": "9876543210"},
            background_tasks=BackgroundTasks(),
        )]
        seen["elapsed"] = time.monotonic() - started
        if "agent_started" in seen:
            seen["agent_after"] = seen["agent_started"] - started
        # Anything the turn started and left running is still here.
        seen["left_running"] = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return "".join(chunks)

    seen["output"] = asyncio.run(_go())
    return seen


def _outcomes(seen):
    return [
        call.kwargs["value"]
        for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]


def test_only_chat_prefetches_and_voice_runs_as_before():
    from app.services.voice import VOICE_SURFACE

    assert chat_service.CHAT_SURFACE.prefetch is chat_service._chat_prefetch
    # Voice's farmer data comes from its background set; run_turn hands its
    # agent input nothing new.
    assert VOICE_SURFACE.prefetch is None


def test_the_agent_waits_for_the_slowest_stage_not_their_sum(monkeypatch):
    # 0.4 s of farmer context against 0.2 s pretranslation then 0.2 s moderation.
    # One after another the agent starts at ~0.8 s; overlapping moderation only,
    # at ~0.6 s; overlapping both, at ~0.4 s.
    seen = _drive(monkeypatch, farmer_s=0.4, pretranslation_s=0.2, moderation_s=0.2)

    assert seen["agent_after"] < 0.55, f"agent started after {seen['agent_after']:.2f}s"
    deps = seen["deps"][0]
    assert deps.farmer_info == "farmer data"
    assert deps.farmer_profile_status == "found"
    assert _outcomes(seen) == ["success"]


def test_a_declined_query_cancels_the_farmer_context_it_did_not_need(monkeypatch):
    seen = _drive(
        monkeypatch, farmer_s=1.0, pretranslation_s=0.0, moderation_s=0.05, category="non_agricultural"
    )

    assert seen["output"] == "Please ask about farming."
    assert seen["deps"] == [], "a declined query reached the agent"
    assert seen["farmer"] == ["started", "cancelled"]
    assert seen["left_running"] == []
    assert seen["elapsed"] < 0.5, f"the decline waited {seen['elapsed']:.2f}s for the farmer context"
    assert _outcomes(seen) == ["success"]


# ── the AI-technician searches inside the load ──────────────────────────────


def _account(society, name):
    return FarmerModel.model_validate({
        "unionName": "Sabar",
        "unionCode": "U-SABAR",
        "societyCode": society,
        "farmerCode": f"F-{society}",
        "farmerName": name,
    })


def _bundle(monkeypatch, accounts, technicians_by_society, delay_by_society):
    calls = []

    async def _farmers(_mobile, **_kw):
        return accounts

    async def _technicians(*, union_code, society_code, force_refresh=False):
        calls.append(society_code)
        await asyncio.sleep(delay_by_society[society_code])
        return [AITechnicianRecord(userId=f"u-{society_code}", fullName=technicians_by_society[society_code],
                                   mobileNumber="9000000000")]

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", _farmers)
    monkeypatch.setattr(farmer_context, "search_ai_technicians", _technicians)

    async def _go():
        started = time.monotonic()
        bundle = await farmer_context.get_farmer_context_bundle_by_mobile("9876543210")
        return bundle, time.monotonic() - started

    bundle, elapsed = asyncio.run(_go())
    return bundle.markdown, elapsed, calls


def test_each_accounts_technician_search_runs_at_once_and_keeps_account_order(monkeypatch):
    # The first account's search is the slower one, so finishing order is the
    # reverse of account order: one after another 0.5 s, at once 0.3 s.
    markdown, elapsed, _calls = _bundle(
        monkeypatch,
        [_account("S1", "First Farmer"), _account("S2", "Second Farmer")],
        {"S1": "Slow Technician", "S2": "Fast Technician"},
        {"S1": 0.3, "S2": 0.2},
    )

    assert elapsed < 0.42, f"technician searches took {elapsed:.2f}s"
    positions = [
        markdown.index("## Farmer 1"),
        markdown.index("Slow Technician"),
        markdown.index("## Farmer 2"),
        markdown.index("Fast Technician"),
    ]
    assert positions == sorted(positions), markdown


def test_accounts_on_one_society_share_one_technician_search(monkeypatch):
    markdown, _elapsed, calls = _bundle(
        monkeypatch,
        [_account("S1", "Cow Account"), _account("S1", "Buffalo Account")],
        {"S1": "Shared Technician"},
        {"S1": 0.05},
    )

    assert calls == ["S1"]
    assert markdown.count("Shared Technician") == 2
