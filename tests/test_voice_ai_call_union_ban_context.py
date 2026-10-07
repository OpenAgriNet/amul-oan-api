"""Tests for the Kutch AI-call ban: cache skip, technician-summary copy,
create_ai_call hard refuse, and prompt precedence over try-again-later.
"""
import os
import asyncio
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

import pytest

import agents.tools.farmer_cache as farmer_cache
from agents.tools import ai_call as chat_ai_call
from agents.tools.beckn import network as beckn_network
from agents.tools.beckn.amul import AITechnicianRecord
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from agents.voice.models.ai_call import AISpecies
from agents.voice.models.health_call import HealthCaseType
from agents.voice.tools import ai_call as ai_mod
from agents.voice.tools import health_call as hc_mod
from app.core.cache import ReservationOutcome
from app.voice.models.union import UNION_BANNED_MESSAGE, UNION_BANNED_MESSAGES
from app.voice.farmer import _build_ai_technician_summary
from app.voice.sink import _canned_union_ban_translation, _prepare_voice_output


BANNED_UNION_ALIASES = ("sarhad", "kutch", "kachchh", "kutchh")


def _tech():
    return AITechnicianRecord(
        userId="ait-1",
        fullName="Ramesh Patel",
        mobileNumber="9999999999",
    )


def _fetch_cache(records, monkeypatch):
    calls = []

    async def fake_search(*, union_code, society_code, force_refresh=False):
        calls.append((union_code, society_code))
        return [_tech()]

    monkeypatch.setattr(farmer_cache, "search_ai_technicians", fake_search)
    return asyncio.run(farmer_cache._fetch_ai_technicians(records)), calls


@pytest.mark.parametrize("union_name", BANNED_UNION_ALIASES)
def test_cache_skips_technician_fetch_for_banned_union(monkeypatch, union_name):
    records = [FarmerRecord(
        unionName=union_name, unionCode="12", societyCode="S1", farmerCode="F1",
        farmerName="Kutch Farmer",
    )]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert groups == []
    assert calls == []


def test_cache_fetches_technicians_for_kaira(monkeypatch):
    records = [FarmerRecord(
        unionName="kaira", unionCode="1", societyCode="S-Kaira", farmerCode="F-K",
        farmerName="Kaira Farmer",
    )]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert calls == [("1", "S-Kaira")]
    assert len(groups) == 1
    assert groups[0]["unionCode"] == "1"
    assert groups[0]["technicians"][0]["userId"] == "ait-1"
    assert groups[0]["techniciansLookupFailed"] is False


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


def test_cache_skips_when_union_name_is_snake_case(monkeypatch):
    records = [FarmerRecord(
        union_name="sarhad", unionCode="12", societyCode="S1", farmerCode="F1",
        farmerName="Kutch Farmer",
    )]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert groups == []
    assert calls == []


def test_cache_still_fetches_when_union_name_is_missing(monkeypatch):
    """Unsigned-in / incomplete records are not banned; existing fetch tests
    construct FarmerRecord with codes only."""
    records = [FarmerRecord(unionCode="BANAS", societyCode="SC001")]
    groups, calls = _fetch_cache(records, monkeypatch)
    assert calls == [("BANAS", "SC001")]
    assert len(groups) == 1


def test_skipped_banned_union_is_not_a_failed_lookup():
    envelope = FarmerDataEnvelope(farmers=[FarmerRecord(unionName="sarhad")], aiTechnicians=[])
    assert not any(farmer_cache.technician_lookup_failed(group) for group in envelope.aiTechnicians)


# ── technician summary ────────────────────────────────────────────────────────

def _kaira_group():
    return {
        "farmerName": "Kaira Farmer",
        "farmerCode": "F-K",
        "societyName": "Kaira Society",
        "societyCode": "S-Kaira",
        "unionCode": "1",
        "technicians": [{
            "fullName": "Ramesh Patel",
            "mobileNumber": "9999999999",
            "userId": "ait-1",
        }],
    }


def _kutch_group():
    return {
        "farmerName": "Kutch Farmer",
        "farmerCode": "F1",
        "societyName": "Sarhad Society",
        "societyCode": "S1",
        "unionCode": "12",
        "technicians": [{
            "fullName": "Kutch Tech",
            "mobileNumber": "8888888888",
            "userId": "ait-kutch",
        }],
    }


def _assert_ban_copy(text: str) -> None:
    assert UNION_BANNED_MESSAGE in text
    assert "AI call booking is not allowed for this union." in text
    assert "Do not ask which technician" in text
    assert "Do not call `create_ai_call`" in text
    assert "not available in the current signed-in context" not in text


@pytest.mark.parametrize("union_name", BANNED_UNION_ALIASES)
def test_technician_summary_instructs_ban_when_lookup_was_skipped(union_name):
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(
            unionName=union_name, unionCode="12", societyCode="S1", farmerCode="F1",
            farmerName="Kutch Farmer", societyName="Sarhad Society",
        )],
        aiTechnicians=[],
    )
    text = _build_ai_technician_summary(envelope)
    _assert_ban_copy(text)
    assert "Ramesh Patel" not in text
    assert "ait-1" not in text
    assert "AI technician option:" not in text


@pytest.mark.parametrize("union_name", BANNED_UNION_ALIASES)
def test_technician_summary_hides_cached_technicians_for_banned_union(union_name):
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(
            unionName=union_name, unionCode="12", societyCode="S1", farmerCode="F1",
            farmerName="Kutch Farmer", societyName="Sarhad Society",
        )],
        aiTechnicians=[_kutch_group()],
    )
    text = _build_ai_technician_summary(envelope)
    _assert_ban_copy(text)
    assert "Kutch Tech" not in text
    assert "ait-kutch" not in text
    assert "8888888888" not in text
    assert "AI technician option:" not in text


def test_technician_summary_still_lists_kaira_technicians():
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(
            unionName="kaira", unionCode="1", societyCode="S-Kaira", farmerCode="F-K",
            farmerName="Kaira Farmer",
        )],
        aiTechnicians=[_kaira_group()],
    )
    text = _build_ai_technician_summary(envelope)
    assert "Ramesh Patel" in text
    assert "ait-1" in text
    assert UNION_BANNED_MESSAGE not in text
    assert "Do not call `create_ai_call`" not in text


def test_technician_summary_mixed_unions_hides_kutch_keeps_kaira():
    envelope = FarmerDataEnvelope(
        farmers=[
            FarmerRecord(
                unionName="kaira", unionCode="1", societyCode="S-Kaira", farmerCode="F-K",
                farmerName="Kaira Farmer",
            ),
            FarmerRecord(
                unionName="kutch", unionCode="12", societyCode="S1", farmerCode="F1",
                farmerName="Kutch Farmer",
            ),
        ],
        aiTechnicians=[_kaira_group(), _kutch_group()],
    )
    text = _build_ai_technician_summary(envelope)
    _assert_ban_copy(text)
    assert "Ramesh Patel" in text
    assert "ait-1" in text
    assert "Kutch Tech" not in text
    assert "ait-kutch" not in text


def test_technician_summary_snake_case_union_name_is_banned():
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(
            union_name="sarhad", unionCode="12", societyCode="S1", farmerCode="F1",
            farmerName="Kutch Farmer",
        )],
        aiTechnicians=[_kutch_group()],
    )
    text = _build_ai_technician_summary(envelope)
    _assert_ban_copy(text)
    assert "Kutch Tech" not in text


def test_technician_summary_without_farmers_keeps_cached_groups():
    """Existing unit tests build envelopes with technician groups and no farmers."""
    envelope = FarmerDataEnvelope(farmers=[], aiTechnicians=[_kaira_group()])
    text = _build_ai_technician_summary(envelope)
    assert "Ramesh Patel" in text
    assert UNION_BANNED_MESSAGE not in text


def test_technician_summary_all_banned_farmers_drop_every_cached_group():
    """When every signed-in farmer is banned, leftover cache rows must not leak."""
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(unionName="sarhad", farmerName="Kutch Farmer")],
        aiTechnicians=[_kutch_group()],
    )
    text = _build_ai_technician_summary(envelope)
    _assert_ban_copy(text)
    assert "Kutch Tech" not in text
    assert "ait-kutch" not in text


# ── prompts ───────────────────────────────────────────────────────────────────

PROMPTS = (
    "voice_system_translation_pipeline_en.md",
    "voice_system_translation_pipeline_gpt5_1_en.md",
    "voice_system_translation_pipeline_gemma4_en.md",
)
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "prompts"
UNQUALIFIED_TRY_AGAIN = (
    "If no technician options are available for the selected farmer, say technician details are not available right now and ask them to try again later.",
    "If no technician options exist for the chosen farmer, say technician details are not available right now and ask them to try again later.",
    "Zero technicians → say technician details are not available right now and ask them to try again later.",
)


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_put_union_ban_ahead_of_technician_selection(name):
    rendered = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    assert UNION_BANNED_MESSAGE in rendered
    ban_at = rendered.find("Union ban (takes precedence)")
    ask_at = rendered.find("If more than one technician option is available")
    if ask_at < 0:
        ask_at = rendered.find("Multiple → ask the caller")
    assert 0 <= ban_at < ask_at
    assert "do **not** ask which technician" in rendered.lower()
    assert "does not say AI calls are banned for this union" in rendered
    for leftover in UNQUALIFIED_TRY_AGAIN:
        assert leftover not in rendered


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_keep_technician_selection_for_allowed_unions(name):
    rendered = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    assert (
        "If more than one technician option is available" in rendered
        or "Multiple → ask the caller" in rendered
    )


# ── create_ai_call hard block ─────────────────────────────────────────────────

SPECIES = next(iter(AISpecies))
# create_ai_call rejects identifiers that cannot be real; 24 base64 chars is the
# shape of every real prod technician id.
TECH_ID = "YWl0LXRlY2gtMDAwMDAwMQ=="



async def _in_scope():
    return True


def _booking_ctx(session_id="s-ban", unions=None, include_unions=True):
    deps = SimpleNamespace(session_id=session_id, ensure_in_scope=_in_scope)
    if include_unions:
        deps.farmer_unions = unions
    return SimpleNamespace(deps=deps)


def _patch_booking(monkeypatch, calls, name="network_create_ai_call_result"):
    """Count Beckn confirms and reservations; every confirm books."""
    async def fake_confirm(*args, **kwargs):
        calls["n"] += 1
        return beckn_network.NetworkBookingResult(True, "T1", "Booked successfully. Ticket: T1")

    async def fake_reserve(*args, **kwargs):
        calls["reserve"] += 1
        return ReservationOutcome.ACQUIRED

    monkeypatch.setattr(beckn_network, name, fake_confirm)
    monkeypatch.setattr(chat_ai_call, "reserve", fake_reserve)


def test_create_ai_call_refuses_sarhad_without_writing(monkeypatch):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(unions=["sarhad"]), "159", "00002", "5058", TECH_ID, SPECIES)
    )
    assert out == UNION_BANNED_MESSAGE
    assert calls == {"n": 0, "reserve": 0}


@pytest.mark.parametrize("unions", [["kutch"], ["KACHCHH"], ["kutchh"], ["kaira", "sarhad"]])
def test_create_ai_call_refuses_canonical_and_mixed_banned_unions(monkeypatch, unions):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(unions=unions), "159", "00002", "5058", TECH_ID, SPECIES)
    )
    assert out == UNION_BANNED_MESSAGE
    assert calls == {"n": 0, "reserve": 0}


def test_create_ai_call_still_books_for_kaira(monkeypatch):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls)

    out = asyncio.run(
        ai_mod.create_ai_call(_booking_ctx(session_id=None, unions=["kaira"]), "159", "00002", "5058", TECH_ID, SPECIES)
    )
    assert calls["n"] == 1
    assert "booked successfully" in out.lower()
    assert out != UNION_BANNED_MESSAGE


@pytest.mark.parametrize("unions,include_unions", [([], True), (None, False)])
def test_create_ai_call_empty_or_missing_unions_is_not_banned(monkeypatch, unions, include_unions):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls)

    out = asyncio.run(
        ai_mod.create_ai_call(
            _booking_ctx(session_id=None, unions=unions, include_unions=include_unions),
            "159", "00002", "5058", TECH_ID, SPECIES,
        )
    )
    assert calls["n"] == 1
    assert "booked successfully" in out.lower()


def test_moderation_block_runs_before_union_ban(monkeypatch):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls)

    async def _out_of_scope():
        return False

    ctx = SimpleNamespace(deps=SimpleNamespace(
        session_id="s-mod",
        ensure_in_scope=_out_of_scope,
        farmer_unions=["kutch"],
    ))
    out = asyncio.run(ai_mod.create_ai_call(ctx, "159", "00002", "5058", TECH_ID, SPECIES))
    assert out == "This helpline only handles dairy farming and animal husbandry questions."
    assert calls["n"] == 0


def test_health_call_still_books_for_kutch_union(monkeypatch):
    calls = {"n": 0, "reserve": 0}
    _patch_booking(monkeypatch, calls, name="network_create_health_call_result")
    out = asyncio.run(
        hc_mod.create_health_call(
            _booking_ctx(session_id=None, unions=["kutch"]),
            "159", "00002", "5058", SPECIES, next(iter(HealthCaseType)), "fever",
        )
    )
    assert calls["n"] == 1
    assert "booked successfully" in out.lower()
    assert UNION_BANNED_MESSAGE not in out


# ── canned caller copy ────────────────────────────────────────────────────────

@pytest.mark.parametrize("lang,expected", [
    ("gu", UNION_BANNED_MESSAGES["gu"]),
    ("gujarati", UNION_BANNED_MESSAGES["gu"]),
    ("hi", UNION_BANNED_MESSAGES["hi"]),
    ("hindi", UNION_BANNED_MESSAGES["hi"]),
    ("en", UNION_BANNED_MESSAGES["en"]),
])
def test_canned_union_ban_translation_pins_agreed_copy(lang, expected):
    assert _canned_union_ban_translation(UNION_BANNED_MESSAGE, lang) == expected
    assert _canned_union_ban_translation(f"  {UNION_BANNED_MESSAGE}  ", lang) == expected
    assert _canned_union_ban_translation("Please try again later.", lang) is None


def test_canned_union_ban_gujarati_survives_voice_cleanup():
    spoken = _prepare_voice_output(UNION_BANNED_MESSAGES["gu"], "gu")
    assert "દૂધ મંડળી" in spoken
    assert spoken.strip() == UNION_BANNED_MESSAGES["gu"].strip()

