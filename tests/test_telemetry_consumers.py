"""Format checks for telemetry/consumers.yaml, the list of what reads our telemetry."""

import re
from pathlib import Path

import pytest

from app.services.telemetry_era_registry import default_era_registry_path, load_registry_file, load_yaml_file

CONSUMERS = Path(__file__).resolve().parents[1] / "telemetry" / "consumers.yaml"
STATUSES = {"working", "broken", "retired"}
SOURCES = {"code", "eras.yaml"}
LIST_KINDS = {"trace_name", "trace_field", "metadata_key"}
ERA_ID = re.compile(r"^(chat|voice)\.[a-z0-9]+$")


def check_consumers(payload, known_eras=None):
    """Problems in a parsed consumers.yaml, one line each. Empty means it's usable."""
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return ["consumers.yaml needs version: 1"]
    consumers = payload.get("consumers")
    if not isinstance(consumers, list) or not consumers:
        return ["consumers must be a non-empty list"]

    problems, seen = [], set()
    for index, consumer in enumerate(consumers):
        name = consumer.get("consumer") if isinstance(consumer, dict) else None
        if not isinstance(name, str) or not name:
            problems.append(f"consumers[{index}] needs a consumer name")
            continue
        if name in seen:
            problems.append(f"{name} is listed twice")
        seen.add(name)
        if consumer.get("status") not in STATUSES:
            problems.append(f"{name}.status must be one of {sorted(STATUSES)}")
        source = consumer.get("verified_from")
        if source not in SOURCES:
            problems.append(f"{name}.verified_from must be code or eras.yaml")
        if source == "code" and not consumer.get("verified_on"):
            problems.append(f"{name} was checked against code but has no verified_on date")

        reads = consumer.get("reads")
        if not isinstance(reads, list) or not reads:
            problems.append(f"{name} needs at least one read")
            continue
        for number, read in enumerate(reads):
            where = f"{name}.reads[{number}]"
            problems.extend(_read_problems(where, read, source, known_eras))
            if read.get("dead_since_era") and consumer.get("status") == "working":
                problems.append(f"{where} is dead since {read['dead_since_era']}, so {name} can't be working")
    return problems


def _read_problems(where, read, source, known_eras):
    if not isinstance(read, dict):
        return [f"{where} must be a mapping"]
    problems = []
    kind = read.get("kind")
    if kind in LIST_KINDS:
        values = read.get("values")
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
            problems.append(f"{where} needs values: a list of names")
    elif kind == "metadata_contains":
        if not isinstance(read.get("key"), str) or not isinstance(read.get("value"), str):
            problems.append(f"{where} needs a key and a value")
    else:
        problems.append(f"{where}.kind must be one of {sorted(LIST_KINDS | {'metadata_contains'})}")
    if not isinstance(read.get("used_for"), str) or not read["used_for"]:
        problems.append(f"{where} needs used_for")
    if source == "code" and not read.get("path"):
        problems.append(f"{where} was checked against code but has no path")
    for field in ("written_for_eras", "dead_since_era"):
        eras = read.get(field)
        for era in [eras] if isinstance(eras, str) else eras or []:
            if era == "unknown":
                continue
            if not isinstance(era, str) or not ERA_ID.match(era):
                problems.append(f"{where}.{field} has {era!r}, which isn't an era id")
            elif known_eras is not None and era not in known_eras:
                problems.append(f"{where}.{field} has {era}, which isn't in eras.yaml")
    return problems


def _known_eras():
    path = default_era_registry_path()
    if not path.exists():
        return None
    payload = load_registry_file(path)
    return {era["era_id"] for section in ("chat_eras", "voice_eras") for era in payload.get(section, [])}


def test_consumers_file_has_no_problems():
    assert check_consumers(load_yaml_file(CONSUMERS), _known_eras()) == []


def _consumer(**overrides):
    return {
        "consumer": "dash",
        "status": "broken",
        "verified_from": "eras.yaml",
        "reads": [{"kind": "trace_name", "values": ["agent_journey"], "used_for": "turn count"}],
        **overrides,
    }


def _file(*consumers):
    return {"version": 1, "consumers": list(consumers)}


def test_a_minimal_consumer_passes():
    assert check_consumers(_file(_consumer())) == []


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        ({"consumers": []}, "consumers.yaml needs version: 1"),
        (_file(_consumer(), _consumer()), "dash is listed twice"),
        (_file(_consumer(status="fine")), "dash.status must be one of"),
        (_file(_consumer(verified_from="memory")), "dash.verified_from must be code or eras.yaml"),
        (_file(_consumer(verified_from="code")), "dash was checked against code but has no verified_on date"),
        (_file(_consumer(verified_from="code", verified_on="2026-09-24")), "dash.reads[0] was checked against code but has no path"),
        (_file(_consumer(reads=[{"kind": "trace_name", "values": ["x"]}])), "dash.reads[0] needs used_for"),
        (_file(_consumer(reads=[{"kind": "span", "used_for": "x"}])), "dash.reads[0].kind must be one of"),
        (_file(_consumer(reads=[{"kind": "metadata_contains", "key": "a", "used_for": "x"}])), "dash.reads[0] needs a key and a value"),
        (
            _file(_consumer(reads=[{"kind": "metadata_key", "values": ["v"], "used_for": "x", "dead_since_era": "c5"}])),
            "dash.reads[0].dead_since_era has 'c5', which isn't an era id",
        ),
        (
            _file(_consumer(status="working", reads=[{"kind": "metadata_key", "values": ["v"], "used_for": "x", "dead_since_era": "chat.c5"}])),
            "dash.reads[0] is dead since chat.c5, so dash can't be working",
        ),
    ],
)
def test_check_consumers_names_each_problem(payload, problem):
    assert any(line.startswith(problem) for line in check_consumers(payload))


def test_era_ids_must_exist_in_eras_yaml_when_it_is_there():
    payload = _file(_consumer(reads=[{"kind": "metadata_key", "values": ["v"], "used_for": "x", "dead_since_era": "chat.c99"}]))

    assert check_consumers(payload, known_eras={"chat.c5"}) == [
        "dash.reads[0].dead_since_era has chat.c99, which isn't in eras.yaml"
    ]
