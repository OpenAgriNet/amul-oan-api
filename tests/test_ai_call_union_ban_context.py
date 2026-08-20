"""Tests for the Kutch AI-call ban: context, cache, prompts, and the booking tool.

Covers the three plan layers: no technician list for banned unions, prompt rules
that do not fall through to "try again later", and a hard refuse in create_ai_call.
"""
import os
import asyncio
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.tools import ai_call as ai_mod
from agents.tools import health_call as hc_mod
from agents.tools.farmer_animal_backends import AITechnicianBySocietyRecord
from app.models.ai_call import AISpecies
from app.models.farmer import FarmerModel
from app.models.union import (
    UNION_BANNED_MESSAGE,
    UNION_BANNED_MESSAGE_GU,
    UNION_BANNED_MESSAGE_HI,
)
from app.models.farmer_transport import FarmerRecord
from app.models.health_call import HealthCaseType
from helpers.utils import get_prompt
import agents.farmer_context as farmer_ctx
import agents.services.farmer_cache as farmer_cache


PROMPTS = ("agrinet_system.md", "agrinet_system_translation_pipeline.md")
BANNED_UNION_ALIASES = ("sarhad", "kutch", "kachchh", "kutchh")


def test_union_banned_message_is_the_agreed_farmer_facing_string():
    assert UNION_BANNED_MESSAGE == "Kindly contact your Milk Society to book the service."
    assert UNION_BANNED_MESSAGE_GU == "કૃપા કરીને આપની દૂધ મંડળીનો સંપર્ક કરશો."
    assert UNION_BANNED_MESSAGE_HI == "कृपया सेवा बुक करने के लिए अपनी दूध मंडली से संपर्क करें।"


def _render_prompt(name):
    return get_prompt(name, context={
        "today_date": "15-08-2026", "today_datetime": "15-08-2026 10:00",
        "farmer_context": None, "ambiguity_hints": None,
        "response_max_chars": None, "loan_max_amount": "5,000",
        "loan_interest_rate_pct": "7", "network_tools_enabled": True,
    })


def _tech():
    return AITechnicianBySocietyRecord(
        userId="ait-1",
        fullName="Ramesh Patel",
        mobileNumber="9999999999",
    )


def _append_markdown(farmer, monkeypatch):
    calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        calls["n"] += 1
        return [_tech()]

    monkeypatch.setattr(farmer_ctx, "get_ai_technicians_by_society_api", fake_api)
    lines = []
    asyncio.run(farmer_ctx._append_ai_technicians_markdown(lines, farmer))
    return "\n".join(lines), calls["n"]


@pytest.mark.parametrize("union_name", BANNED_UNION_ALIASES)
def test_banned_union_farmer_context_skips_technicians_and_instructs_ban(monkeypatch, union_name):
    farmer = FarmerModel(unionName=union_name, unionCode="12", societyCode="S1")
    text, api_calls = _append_markdown(farmer, monkeypatch)
    assert api_calls == 0
    assert UNION_BANNED_MESSAGE in text
    assert "Available AI technicians" not in text
    assert "user_id" not in text
    assert "Ramesh Patel" not in text
    assert "Do not ask which technician" in text
    assert "Do not call `create_ai_call`" in text


def test_kaira_farmer_context_still_lists_technicians(monkeypatch):
    farmer = FarmerModel(unionName="kaira", unionCode="1", societyCode="S1")
    text, api_calls = _append_markdown(farmer, monkeypatch)
    assert api_calls == 1
    assert "Available AI technicians" in text
    assert "Ramesh Patel" in text
    assert "ait-1" in text
    assert UNION_BANNED_MESSAGE not in text


def test_farmer_context_bundle_for_sarhad_omits_technicians(monkeypatch):
    """The chat path injects get_farmer_context_bundle_by_mobile into the prompt."""

    async def fake_farmers(mobile):
        return [FarmerModel(
            unionName="sarhad", unionCode="12", societyCode="S1", farmerName="Kutch Farmer",
        )]

    async def fake_schemes(lines, unions):
        return None

    calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        calls["n"] += 1
        return [_tech()]

    monkeypatch.setattr(farmer_ctx, "get_farmer_data_by_mobile", fake_farmers)
    monkeypatch.setattr(farmer_ctx, "_append_union_scheme_summary_markdown", fake_schemes)
    monkeypatch.setattr(farmer_ctx, "get_ai_technicians_by_society_api", fake_api)

    markdown, unions, _location = asyncio.run(
        farmer_ctx.get_farmer_context_bundle_by_mobile("9876543210")
    )
    assert unions == ["sarhad"]
    assert calls["n"] == 0
    assert UNION_BANNED_MESSAGE in markdown
    assert "Available AI technicians" not in markdown
    assert "user_id" not in markdown
    assert "Ramesh Patel" not in markdown


def _fetch_cache(records, monkeypatch):
    calls = []

    async def fake_api(query, token):
        calls.append((query.union_code, query.society_code))
        return [_tech()]

    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    monkeypatch.setattr(farmer_cache, "get_ai_technicians_by_society_api", fake_api)
    return asyncio.run(farmer_cache._fetch_ai_technicians(records)), calls


def test_cache_skips_technician_fetch_for_sarhad_record(monkeypatch):
    records = [FarmerRecord(
        unionName="sarhad", unionCode="12", societyCode="S1", farmerCode="F1",
        farmerName="Kutch Farmer",
    )]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert groups == []
    assert calls == []


def test_cache_fetches_technicians_for_kaira_and_skips_kutch(monkeypatch):
    records = [
        FarmerRecord(
            unionName="kaira", unionCode="1", societyCode="S-Kaira", farmerCode="F-K",
            farmerName="Kaira Farmer",
        ),
        FarmerRecord(
            unionName="kutch", unionCode="12", societyCode="S-Kutch", farmerCode="F-C",
            farmerName="Kutch Farmer",
        ),
    ]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert calls == [("1", "S-Kaira")]
    assert len(groups) == 1
    assert groups[0]["unionCode"] == "1"
    assert groups[0]["technicians"][0]["userId"] == "ait-1"


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_put_union_ban_ahead_of_technician_selection(name):
    rendered = _render_prompt(name)
    assert UNION_BANNED_MESSAGE in rendered
    ban_at = rendered.find("Union ban (takes precedence)")
    ask_at = rendered.find("When AI technician options are available")
    assert 0 <= ban_at < ask_at
    assert "do **not** ask which technician" in rendered.lower()
    assert "does not say AI calls are banned for this union" in rendered
    # Unqualified fallback would tell a banned-union farmer to retry later.
    assert (
        "If no AI technician options are available in the Farmer Profile context, explain that technician details are unavailable"
        not in rendered
    )


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_keep_technician_selection_for_allowed_unions(name):
    rendered = _render_prompt(name)
    assert "When AI technician options are available, ask the user which technician they want to select." in rendered


def test_default_prompt_lists_localized_ban_lines():
    rendered = _render_prompt("agrinet_system.md")
    assert UNION_BANNED_MESSAGE in rendered
    assert UNION_BANNED_MESSAGE_GU in rendered
    assert UNION_BANNED_MESSAGE_HI in rendered


# ── create_ai_call hard block ─────────────────────────────────────────────────

SPECIES = next(iter(AISpecies))


async def _in_scope():
    return True


def _booking_ctx(session_id="s-ban", unions=None, include_unions=True, lang_code=None):
    deps = SimpleNamespace(session_id=session_id, ensure_in_scope=_in_scope)
    if include_unions:
        deps.farmer_unions = unions
    if lang_code is not None:
        deps.lang_code = lang_code
    return SimpleNamespace(deps=deps)


def test_create_ai_call_refuses_sarhad_without_writing(monkeypatch):
    calls = {"api": 0, "reserve": 0}

    async def fake_api(*args, **kwargs):
        calls["api"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    async def fake_reserve(*args, **kwargs):
        calls["reserve"] += 1
        return True, True

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)
    monkeypatch.setattr(ai_mod, "_reserve_booking_slot", fake_reserve)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(unions=["sarhad"]), "U", "S", "F", "tech1", SPECIES)
    )
    assert out == UNION_BANNED_MESSAGE
    assert calls == {"api": 0, "reserve": 0}


@pytest.mark.parametrize("unions", [["kutch"], ["KACHCHH"], ["kutchh"], ["kaira", "sarhad"]])
def test_create_ai_call_refuses_canonical_and_mixed_banned_unions(monkeypatch, unions):
    calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(unions=unions), "U", "S", "F", "tech1", SPECIES)
    )
    assert out == UNION_BANNED_MESSAGE
    assert calls["n"] == 0


@pytest.mark.parametrize("lang,expected", [
    ("en", UNION_BANNED_MESSAGE),
    ("gu", UNION_BANNED_MESSAGE_GU),
    ("gujarati", UNION_BANNED_MESSAGE_GU),
    ("hi", UNION_BANNED_MESSAGE_HI),
    ("hindi", UNION_BANNED_MESSAGE_HI),
])
def test_create_ai_call_returns_localized_ban_message(monkeypatch, lang, expected):
    calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)

    out = asyncio.run(
        ai_mod.create_ai_call(
            _booking_ctx(unions=["kutch"], lang_code=lang), "U", "S", "F", "tech1", SPECIES,
        )
    )
    assert out == expected
    assert calls["n"] == 0


def test_create_ai_call_still_books_for_kaira(monkeypatch):
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(unions=["kaira"]), "U", "S", "F", "tech1", SPECIES)
    )
    assert calls["n"] == 1
    assert "booked successfully" in out
    assert out != UNION_BANNED_MESSAGE


@pytest.mark.parametrize("unions,include_unions", [([], True), (None, False)])
def test_create_ai_call_empty_or_missing_unions_is_not_banned(monkeypatch, unions, include_unions):
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)

    out = asyncio.run(
        ai_mod.create_ai_call(
            _booking_ctx(unions=unions, include_unions=include_unions),
            "U", "S", "F", "tech1", SPECIES,
        )
    )
    assert calls["n"] == 1
    assert "booked successfully" in out


def test_moderation_block_runs_before_union_ban(monkeypatch):
    calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    async def _out_of_scope():
        return False

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        session_id="s-mod",
        ensure_in_scope=_out_of_scope,
        farmer_unions=["kutch"],
    ))
    out = asyncio.run(ai_mod.create_ai_call(ctx, "U", "S", "F", "tech1", SPECIES))
    assert out == ai_mod.OUT_OF_SCOPE_MESSAGE
    assert calls["n"] == 0


def test_health_call_still_books_for_kutch_union(monkeypatch):
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="H1")

    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    monkeypatch.setattr(hc_mod, "create_health_call_api", fake_api)
    out = asyncio.run(
        hc_mod.create_health_call(
            _booking_ctx(session_id=None, unions=["kutch"]),
            "U", "S", "F", SPECIES, next(iter(HealthCaseType)), "fever",
        )
    )
    assert calls["n"] == 1
    assert "booked successfully" in out
    assert UNION_BANNED_MESSAGE not in out


@pytest.mark.parametrize("source,target,expected", [
    (UNION_BANNED_MESSAGE, "gu", UNION_BANNED_MESSAGE_GU),
    (UNION_BANNED_MESSAGE, "gujarati", UNION_BANNED_MESSAGE_GU),
    (UNION_BANNED_MESSAGE, "hi", UNION_BANNED_MESSAGE_HI),
    (UNION_BANNED_MESSAGE, "hindi", UNION_BANNED_MESSAGE_HI),
    (UNION_BANNED_MESSAGE, "en", UNION_BANNED_MESSAGE),
    (f"`{UNION_BANNED_MESSAGE}`", "gu", UNION_BANNED_MESSAGE_GU),
    (UNION_BANNED_MESSAGE_GU, "gujarati", UNION_BANNED_MESSAGE_GU),
    (UNION_BANNED_MESSAGE_HI, "hindi", UNION_BANNED_MESSAGE_HI),
])
def test_post_translation_uses_canned_union_ban_line(source, target, expected):
    from app.services.translation import _canned_union_ban_translation
    assert _canned_union_ban_translation(source, target) == expected


def test_post_translation_does_not_canned_swap_other_text():
    from app.services.translation import _canned_union_ban_translation
    assert _canned_union_ban_translation("Book an AI call for the cow.", "gu") is None


def test_translate_text_emits_canned_gu_ban_without_model():
    from app.services.translation import translate_text
    assert asyncio.run(translate_text(UNION_BANNED_MESSAGE, "english", "gujarati")) == UNION_BANNED_MESSAGE_GU


def test_translate_stream_emits_canned_hi_ban_without_model():
    from app.services.translation import translate_text_stream_fast

    async def _collect():
        return "".join([c async for c in translate_text_stream_fast(UNION_BANNED_MESSAGE, "english", "hindi")])

    assert asyncio.run(_collect()) == UNION_BANNED_MESSAGE_HI
