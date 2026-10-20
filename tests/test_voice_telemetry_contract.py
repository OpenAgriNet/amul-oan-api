"""Voice turns served from here send turn.v1 as telemetry/contracts/turn.v1/voice.json says.

turn.v1 is the shape every pipeline sends (amul-telemetry telemetry/turn.v1.yaml); the
contract is what voice's code sends of it, for amul-telemetry's contract check. The
old voice.turn.v1 contract is released and never changes: traces in Langfuse follow it.

The checks run turns down every path voice has, through the real run_turn with
voice's pieces (the stand-ins of tests/test_voice_telemetry.py) and through voice's
modules on their own, and compare what Langfuse was handed with the contract.
"""

import ast
import asyncio
import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

import pytest

from app.config import settings
from app.services import chat as chat_service
from app.services.telemetry_stamps import TURN_ROOT, VOICE_TELEMETRY_SCHEMA_VERSION
from app.voice import classifiers as vc
from app.voice.liveness import VoiceLiveness
from app.voice.moderation import ModerationVerdict
from app.voice.outbound_consent import ConsentVerdict
from tests.test_voice_pretranslation import _pretranslate
from tests.test_voice_pretranslation import _turn as _gujarati_turn
from tests.test_voice_sink import _English, _speak, _Translation
from tests.test_voice_telemetry import _PRIOR, _run, _Sender, _taken_over, _turn, _turn_execution, call, lf, trace  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "telemetry" / "contracts"
CONTRACT = CONTRACTS / VOICE_TELEMETRY_SCHEMA_VERSION / "voice.json"
CONTRACT_NAME = f"telemetry/contracts/{VOICE_TELEMETRY_SCHEMA_VERSION}/voice.json"
NEW_VERSION = (
    f"{VOICE_TELEMETRY_SCHEMA_VERSION} is the shape every pipeline sends (amul-telemetry telemetry/turn.v1.yaml). "
    f"A key, step or outcome added or removed is a registry change there first; then change {CONTRACT_NAME} "
    "and voice's code together."
)
STAMPS = {"amul.schema_version", "amul.service", "amul.release", "amul.kind"}

# The same fingerprints as voice-oan-api's. Key order, spacing and the "note"
# text don't count.
RELEASED_CONTRACTS = {
    "voice.turn.v1": "26b264334db6e50d83b3f0d541cd774f9742daaa72af77473250aba50395ca0e",
}

_GENERIC_NAMES = {"data", "id", "result", "score", "status", "time", "type", "value"}
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")


def _contract():
    assert CONTRACT.exists(), f"No contract for {VOICE_TELEMETRY_SCHEMA_VERSION}. Add {CONTRACT_NAME}."
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _fingerprint(path):
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.pop("note", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ── every path a voice turn can take, and what each handed Langfuse ──────────


@dataclasses.dataclass
class _Sent:
    """What one run handed Langfuse: the root's keys, each step, the trace fields."""

    root_keys: set
    opened: dict
    closed: dict
    io: dict
    root_updates: list
    steps: list
    scores: list


def _take(client):
    sent = _Sent(
        root_keys=set(client.root.kwargs["metadata"]) | set(client.closed),
        opened=client.opened,
        closed=client.closed,
        io=dict(client.io),
        root_updates=list(client.root.updates),
        steps=[o for o in client.observations if o.kwargs["name"] != TURN_ROOT],
        scores=list(client.scores),
    )
    client.observations.clear()
    client.propagated.clear()
    client.scores.clear()
    client.io.clear()
    client.traces.clear()
    return sent


def _every_turn(lf, call, monkeypatch):  # noqa: F811
    """One run down each path, with the words and the caller's number sent, as voice sends them."""

    async def _please_repeat(**_kw):
        return "Please say that again."

    monkeypatch.setattr(vc, "generate_stt_signal_response", _please_repeat)
    runs = []
    answered = _turn(history=_PRIOR)
    _run(dataclasses.replace(answered, call=dataclasses.replace(answered.call, request_epoch=3)))
    runs.append(_take(lf))
    # A request another took the session over from, before its history write.
    _run(_turn(history=_PRIOR), is_stale=_taken_over(monkeypatch, by="ef" * 16, owner=False))
    runs.append(_take(lf))
    for query in ("hello", "What is your name?", "hm", "No audio/User is speaking softly", "please remain on the line"):
        _run(_turn(query))
        runs.append(_take(lf))
    call["moderation"] = ModerationVerdict(category="irrelevant", reason="off topic")
    _run(_turn("Who won the match?", history=_PRIOR))
    runs.append(_take(lf))
    call["moderation"] = ModerationVerdict(category="in_scope", reason="ok")
    call["consent"] = ConsentVerdict(intent="negative", reason="not now")
    _run(_turn("No", history=_PRIOR, consent_turn=True))
    runs.append(_take(lf))
    call["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    call["prefetched"] = "Total 40 litres."
    _run(_turn("Yes", history=_PRIOR, consent_turn=True))
    runs.append(_take(lf))
    call["consent"] = None

    async def _down(key, messages):
        raise RuntimeError("redis down")

    monkeypatch.setattr(chat_service, "update_message_history", _down)
    with pytest.raises(RuntimeError):
        _run(_turn(history=_PRIOR))
    runs.append(_take(lf))
    return runs


def _every_module_step(trace, monkeypatch):  # noqa: F811
    """The steps voice's modules write on their own: pre_translation with a fallback,
    post_translation, and a nudge that went out."""
    _pretranslate(
        monkeypatch, _gujarati_turn("ભેંસ"), replies=[RuntimeError("oss down"), "For a buffalo"], execution=_turn_execution()
    )
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."]])
    _speak(_English(["Give the cow water."]))
    monkeypatch.setattr(settings, "enable_voice_nudges", True)
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 0)

    async def _go():
        liveness = VoiceLiveness(_turn(), started_at=time.monotonic(), send=_Sender(), is_stale=None)
        await asyncio.sleep(0.05)
        await liveness.stop()

    asyncio.run(_go())
    return list(trace.sent.observations)


def _steps_sent(observations):
    """Per step name: the types it was opened as, the keys of its output and metadata."""
    steps = {}
    for observation in observations:
        sent = observation.sent
        step = steps.setdefault(sent["name"], {"types": set(), "output": set(), "metadata_keys": set()})
        step["types"].add(sent["as_type"].upper())
        output = sent.get("output")
        if isinstance(output, dict):
            step["output"].update(output)
        step["metadata_keys"].update(sent.get("metadata") or {})
    return steps


# ── the contract against what is sent ────────────────────────────────────────


def test_contract_matches_the_stamped_schema_version_and_root():
    contract = _contract()

    assert contract["schema_version"] == VOICE_TELEMETRY_SCHEMA_VERSION == "turn.v1"
    assert contract["root"] == TURN_ROOT == "turn"
    assert contract["pipelines"] == ["telephony"]


def test_the_root_sends_exactly_the_metadata_keys_the_contract_lists(lf, call, monkeypatch):  # noqa: F811
    runs = _every_turn(lf, call, monkeypatch)
    sent = set().union(*(run.root_keys for run in runs))
    listed, timed = set(_contract()["metadata_keys"]), _timing_keys()

    assert STAMPS <= sent
    # Each mark and stage a turn reached has its time; which ones depends on the path.
    assert sent - timed == listed - timed, f"the amul.* keys voice sends don't match the contract. {NEW_VERSION}"
    assert sent & timed and sent & timed <= listed
    for run in runs:
        # Trace-level fields on the root alone: the environment for every span, the
        # fields before any step, the end of the turn as the root closes.
        assert set(run.opened) == {"environment", "session_id", "user_id", "metadata"}
        assert run.closed["amul.outcome"] in _contract()["outcomes"]


def test_the_steps_opened_match_the_contract(lf, call, trace, monkeypatch):  # noqa: F811
    observations = [step for run in _every_turn(lf, call, monkeypatch) for step in run.steps]
    observations += _every_module_step(trace, monkeypatch)
    sent = _steps_sent(observations)
    listed = _contract()["steps"]

    assert set(sent) == set(listed), f"the steps voice opens don't match the contract. {NEW_VERSION}"
    for name, step in sent.items():
        assert step["types"] == {listed[name]["type"]}, f"step {name} is opened as {sorted(step['types'])}"
        assert step["output"] == set(listed[name].get("output", ())), f"step {name}'s output fields. {NEW_VERSION}"
        assert step["metadata_keys"] <= set(listed[name]["metadata_keys"]), f"step {name}'s metadata keys. {NEW_VERSION}"
        assert all(key.startswith("amul.") for key in step["metadata_keys"]), name


def test_the_trace_fields_sent_match_the_contract(lf, call, monkeypatch):  # noqa: F811
    runs = _every_turn(lf, call, monkeypatch)
    sent = set()
    for run in runs:
        sent |= {"session_id", "user_id"} & {key for key, value in run.opened.items() if value}
        sent |= set(run.io)
        sent |= {key for update in run.root_updates for key, value in update.items() if value}

    assert sent == set(_contract()["trace_fields"]), f"the trace fields voice sends don't match the contract. {NEW_VERSION}"


def test_every_turn_scores_its_outcome_once(lf, call, monkeypatch):  # noqa: F811
    for run in _every_turn(lf, call, monkeypatch):
        (score,) = run.scores
        assert (score["name"], score["value"], score["data_type"]) == ("turn_outcome", run.closed["amul.outcome"], "CATEGORICAL")


# ── what the code can send, read from it ──────────────────────────────────────


def _voice_nodes():
    for path in sorted((REPO / "app" / "voice").rglob("*.py")):
        yield from ast.walk(ast.parse(path.read_text(encoding="utf-8")))


def _function_nodes(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    (function,) = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    return list(ast.walk(function))


def _run_turn_nodes():
    return _function_nodes(REPO / "app" / "services" / "chat.py", "run_turn")


def _first_string_args(*methods):
    return {
        node.args[0].value
        for node in _voice_nodes()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods
        and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
    }


def _timing_keys():
    """The keys of what voice times: each mark (trace.mark("ttft_ms")) as amul.<mark>, and
    each stage (trace.stage("agent"), attach_stage_timing("history_load")) as amul.<stage>_ms."""
    marks = {f"amul.{mark}" for mark in _first_string_args("mark")}
    stages = {f"amul.{stage}_ms" for stage in _first_string_args("stage", "attach_stage_timing")}
    return marks | stages


def test_every_mark_and_stage_voice_times_has_its_key_in_the_contract():
    listed = {key for key in _contract()["metadata_keys"] if key.endswith("_ms")}

    assert {"amul.ttft_ms", "amul.history_load_ms"} <= _timing_keys()
    # Beside them, the turn's whole time, which no mark or stage is.
    assert _timing_keys() | {"amul.total_ms"} == listed, f"a mark's or stage's key. {NEW_VERSION}"


def _outcomes_in(nodes):
    """Strings passed as the outcome to set_outcome() or finish(), positionally or as outcome=."""
    outcomes = set()
    for node in nodes:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"set_outcome", "finish"}:
            values = list(node.args) + [keyword.value for keyword in node.keywords if keyword.arg == "outcome"]
            for value in values:
                outcomes.update(
                    n.value for n in ast.walk(value) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
    return outcomes


def _answer_outcomes_in(nodes):
    """Outcomes voice's answers carry: ClassifierResult(..., outcome="...")."""
    return {
        keyword.value.value
        for node in nodes
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ClassifierResult"
        for keyword in node.keywords
        if keyword.arg == "outcome" and isinstance(keyword.value, ast.Constant)
    }


def _turn_outcomes_in_run_turn():
    """The outcomes run_turn sets itself: `_turn_outcome = "..."`."""
    return {
        node.value.value
        for node in _run_turn_nodes()
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_turn_outcome" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    }


def _staleness_outcomes():
    """What voice's staleness check stops a turn with."""
    return {
        node.value.value
        for node in _function_nodes(REPO / "app" / "voice" / "liveness.py", "__call__")
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }


def _outcomes_a_voice_turn_can_end_with():
    return (
        _outcomes_in(_voice_nodes()) | _answer_outcomes_in(_voice_nodes()) | _staleness_outcomes() | _turn_outcomes_in_run_turn()
    )


def _score_names_in(nodes):
    """Names given to Langfuse scores, e.g. score_current_trace(name="turn_outcome")."""
    return {
        keyword.value.value
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"score_current_trace", "score_current_span", "score_trace", "create_score"}
        for keyword in node.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    }


def test_scores_match_the_contract():
    listed = set(_contract()["scores"])
    found = _score_names_in(_voice_nodes())

    assert found == listed == {"turn_outcome"}, (
        f"Voice sends the scores {sorted(found)}, the contract lists {sorted(listed)}. turn.v1 records the outcome as "
        "amul.outcome and as the turn_outcome score; pipeline_profile and served_tier are metadata, not scores."
    )


def test_run_turn_sends_no_score_of_its_own():
    """Every score a turn sends belongs to its surface's contract, so run_turn,
    which runs voice's turns too, sends none."""
    calls = [
        node.func.attr
        for node in _run_turn_nodes()
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"score_current_trace", "score_current_span", "create_score"}
    ]

    assert not calls


def test_outcomes_match_the_contract():
    listed = set(_contract()["outcomes"])
    found = _outcomes_a_voice_turn_can_end_with()

    assert found <= listed, (
        f"New outcomes {sorted(found - listed)}. {NEW_VERSION} An outcome is declared in turn.v1.yaml with its class, "
        "or the readers count it as unclassified."
    )
    assert listed <= found, f"Outcomes no longer emitted: {sorted(listed - found)}. {NEW_VERSION}"


def test_outcomes_are_found_however_they_are_passed():
    code = """
trace.set_outcome("stale_request")
trace.finish(outcome="hold_message")
trace.finish(trace.outcome or "error", error=RuntimeError("not an outcome"))
ClassifierResult(canned_text="Goodbye.", label="hold_message", outcome="outbound_declined")
"""
    nodes = list(ast.walk(ast.parse(code)))

    assert _outcomes_in(nodes) == {"stale_request", "hold_message", "error"}
    assert _answer_outcomes_in(nodes) == {"outbound_declined"}


# ── the contract file itself ──────────────────────────────────────────────────


def test_released_contracts_never_change():
    for version, fingerprint in RELEASED_CONTRACTS.items():
        path = CONTRACTS / f"{version}.json"
        assert path.exists(), f"{version} is released: keep telemetry/contracts/{version}.json, old traces follow it."
        # Compared as a bool so a failure doesn't print the new fingerprint to paste in.
        unchanged = _fingerprint(path) == fingerprint
        assert unchanged, (
            f"telemetry/contracts/{version}.json is released and can't change: traces already in Langfuse "
            "follow it. Undo the edit and put the change in a new version."
        )


def test_names_are_specific_snake_case():
    contract = _contract()
    names = [
        *(key.removeprefix("amul.") for key in contract["metadata_keys"]),
        *contract["trace_fields"],
        *contract["steps"],
        *(key.removeprefix("amul.") for step in contract["steps"].values() for key in step["metadata_keys"]),
        *(field for step in contract["steps"].values() for field in step.get("output", ())),
        *contract["scores"],
        *contract["outcomes"],
    ]
    unclear = [name for name in names if not _SNAKE_CASE.fullmatch(name) or name in _GENERIC_NAMES]

    assert not unclear, f"Unclear names {unclear}."
