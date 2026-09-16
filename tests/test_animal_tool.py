"""Tests for the single-backend animal tool.

`get_animal_by_tag` has one backend (amulpashudhan), so it formats the
`AnimalModel` it gets back directly — these pin that it stays serializable and
that no second provider is consulted for any society.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
import json

import agents.tools.animal as animal_mod
from app.models.animal import AnimalModel


_ANIMAL = {
    "tagNumber": "123",
    "animalType": "Cow",
    "breed": "Gir",
    "milkingStage": "Mid",
    "lactationNo": 2,
}


def _patch_backend(monkeypatch, value):
    calls = []

    async def fake(tag, token):
        calls.append(tag)
        return value

    monkeypatch.setattr(animal_mod, "fetch_animal_amulpashudhan", fake)
    return calls


def test_get_animal_by_tag_formats_model_as_json(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    _patch_backend(monkeypatch, AnimalModel.model_validate(_ANIMAL))

    out = asyncio.run(animal_mod.get_animal_by_tag("123"))

    assert out.startswith("Animal details for tag 123:")
    payload = json.loads(out.split("\n\n", 1)[1])
    assert payload["tag_number"] == "123"
    assert payload["breed"] == "gir"  # AnimalModel lowercases text fields
    assert payload["lactation_no"] == 2


def test_get_animal_by_tag_uses_one_backend_for_every_society(monkeypatch):
    # Mehsana used to trigger a second provider lookup; it must not any more.
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    calls = _patch_backend(monkeypatch, AnimalModel.model_validate(_ANIMAL))

    asyncio.run(animal_mod.get_animal_by_tag("123", society_name="Mehsana"))

    assert calls == ["123"]


def test_get_animal_by_tag_no_data_message(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    _patch_backend(monkeypatch, None)

    out = asyncio.run(animal_mod.get_animal_by_tag("123"))
    assert "No animal data found" in out


def test_get_animal_data_by_tag_none_without_token(monkeypatch):
    monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)

    async def forbidden(tag, token):
        raise AssertionError("no backend must be called without PASHUGPT_TOKEN")

    monkeypatch.setattr(animal_mod, "fetch_animal_amulpashudhan", forbidden)
    assert asyncio.run(animal_mod.get_animal_data_by_tag("123")) is None
