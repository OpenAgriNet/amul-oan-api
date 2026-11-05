"""The farmer context build is the turn's farmer_context step (turn.v1).

Its lookups swallow a failure into a WARNING and leave the section out, so the
agent answers without it. The step is where that shows: per provider, how many
lookups answered and how many failed, and on any failure level ERROR with the
first reason per provider. The prompt is the same with tracing on or off.
"""
import asyncio
import contextlib

import httpx
import pytest

from agents import farmer_context
from agents.tools.models.animal import AnimalModel
from agents.tools.models.farmer import FarmerModel

TAGS = [f"1000000000{index:02d}" for index in range(23)]


class _Step:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _Client:
    def __init__(self):
        self.steps = []

    @contextlib.contextmanager
    def start_as_current_observation(self, *, name, as_type):
        step = _Step()
        self.steps.append((name, as_type, step))
        yield step


def _sent(client):
    """The one step the build opened, as (name, type, metadata, level, message)."""
    ((name, as_type, step),) = client.steps
    (update,) = step.updates
    return name, as_type, update["metadata"], update.get("level"), update.get("status_message")


def _rejected(*_args, **_kwargs):
    request = httpx.Request("POST", "http://bap.invalid/transactions/init")
    return httpx.HTTPStatusError(
        "Client error '400 Bad Request'", request=request, response=httpx.Response(400, request=request)
    )


def _farmer_context(monkeypatch, *, union="Kaira", cvcc=None, farmers=None, technicians=()):
    """A synthetic farmer with 23 animals; ``cvcc`` answers each CVCC lookup (an exception
    to raise, or a value)."""
    async def _farmers(mobile, **_kw):
        if isinstance(farmers, BaseException):
            raise farmers
        return farmers if farmers is not None else [FarmerModel.model_validate({
            "unionName": union, "unionCode": "2021", "societyCode": "S1", "farmerCode": "F1",
            "farmerName": "Test Farmer", "tagNo": ", ".join(TAGS),
        })]

    async def _animal(tag, **_kw):
        return AnimalModel.model_validate({"tagNumber": tag, "breed": "Gir"})

    async def _cvcc(tag, **_kw):
        if isinstance(cvcc, BaseException):
            raise cvcc
        return cvcc

    async def _technicians(**_kw):
        if isinstance(technicians, BaseException):
            raise technicians
        return list(technicians)

    async def _banas(tag, **_kw):
        return []

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", _farmers)
    monkeypatch.setattr(farmer_context, "fetch_animal_profile", _animal)
    monkeypatch.setattr(farmer_context, "fetch_cvcc_health", _cvcc)
    monkeypatch.setattr(farmer_context, "fetch_banas_visits", _banas)
    monkeypatch.setattr(farmer_context, "search_ai_technicians", _technicians)


def _build(monkeypatch, client):
    monkeypatch.setattr(farmer_context, "get_langfuse_client", lambda: client)
    return asyncio.run(farmer_context.get_farmer_context_bundle_by_mobile("9000000000"))


def test_a_kaira_turn_whose_every_cvcc_lookup_failed_is_an_unavailable_step(monkeypatch):
    """The 10-17..10-29 break: every Kaira CVCC lookup refused with a 400, the turn answered
    without health history, and only WARNING logs said so."""
    _farmer_context(monkeypatch, cvcc=_rejected())
    client = _Client()
    bundle = _build(monkeypatch, client)

    name, as_type, metadata, level, message = _sent(client)
    assert (name, as_type) == ("farmer_context", "span")
    assert metadata == {
        "amul.tool_status": "unavailable",
        "amul.farmer_ok": 1, "amul.farmer_failed": 0,
        "amul.ai_technicians_ok": 1, "amul.ai_technicians_failed": 0,
        "amul.animal_profile_ok": 23, "amul.animal_profile_failed": 0,
        "amul.cvcc_ok": 0, "amul.cvcc_failed": 23,
    }
    assert level == "ERROR"
    assert message == "cvcc 23 of 23 failed: HTTPStatusError: Client error '400 Bad Request'"
    assert bundle.found and "CVCC health details" not in bundle.markdown


def test_the_prompt_is_the_same_with_tracing_on_or_off(monkeypatch):
    _farmer_context(monkeypatch, cvcc=_rejected())
    traced = _build(monkeypatch, _Client())
    untraced = _build(monkeypatch, None)
    assert traced == untraced


def test_a_build_whose_lookups_all_answered_is_a_success(monkeypatch):
    _farmer_context(monkeypatch, union="Banas")
    client = _Client()
    _build(monkeypatch, client)

    _, _, metadata, level, message = _sent(client)
    assert metadata["amul.tool_status"] == "success"
    assert (metadata["amul.banas_visits_ok"], metadata["amul.banas_visits_failed"]) == (23, 0)
    assert "amul.cvcc_ok" not in metadata  # not a Kaira farmer: no CVCC lookup
    assert (level, message) == (None, None)


def test_a_mobile_with_no_farmer_record_is_no_result(monkeypatch):
    _farmer_context(monkeypatch, farmers=[])
    client = _Client()
    bundle = _build(monkeypatch, client)

    assert not bundle.found
    assert _sent(client)[2] == {"amul.tool_status": "no_result", "amul.farmer_ok": 1, "amul.farmer_failed": 0}


def test_a_failed_farmer_fetch_is_unavailable_and_still_raises(monkeypatch):
    _farmer_context(monkeypatch, farmers=httpx.ConnectError("bap down"))
    client = _Client()
    with pytest.raises(httpx.ConnectError):
        _build(monkeypatch, client)

    _, _, metadata, level, message = _sent(client)
    assert (metadata["amul.tool_status"], metadata["amul.farmer_failed"], level) == ("unavailable", 1, "ERROR")
    assert message == "farmer 1 of 1 failed: ConnectError: bap down"


def test_a_failed_technician_search_is_counted(monkeypatch):
    _farmer_context(monkeypatch, union="Banas", technicians=RuntimeError("AI technician discovery is temporarily unavailable"))
    client = _Client()
    bundle = _build(monkeypatch, client)

    _, _, metadata, level, message = _sent(client)
    assert (metadata["amul.ai_technicians_ok"], metadata["amul.ai_technicians_failed"], level) == (0, 1, "ERROR")
    assert message.startswith("ai_technicians 1 of 1 failed: RuntimeError")
    assert "AI technician details could not be fetched right now." in bundle.markdown
