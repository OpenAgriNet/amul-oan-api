"""Voice turns served from here must match telemetry/contracts/voice.turn.v1.json.

The contract is voice-oan-api's (#308), copied unchanged: readers of voice.turn.v1
must not be able to tell which service sent a turn, except by its ``service``
stamp. A released contract never changes. Any change to what a turn sends goes
out as a new schema version with its own contract file, in both repos.

The checks are voice-oan-api's, pointed at this repo's voice code (app/voice)
and at what ``run_turn`` records for it.
"""

import ast
import hashlib
import inspect
import json
import re
from contextlib import contextmanager
from pathlib import Path

import langfuse

from app.config import settings
from app.services.telemetry_stamps import VOICE_TELEMETRY_SCHEMA_VERSION
from app.voice.telemetry import _OUTCOMES
from app.voice.moderation import ModerationVerdict
from app.voice.trace import _VALID_TEXT_MODES, VoiceTrace

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "telemetry" / "contracts"
CONTRACT = CONTRACTS / f"{VOICE_TELEMETRY_SCHEMA_VERSION}.json"
CONTRACT_NAME = f"telemetry/contracts/{VOICE_TELEMETRY_SCHEMA_VERSION}.json"
NEW_VERSION = (
    f"{VOICE_TELEMETRY_SCHEMA_VERSION} can't change once released, so this needs a new schema version "
    "in voice-oan-api and here: bump VOICE_TELEMETRY_SCHEMA_VERSION in app/services/telemetry_stamps.py, "
    f"copy {CONTRACT_NAME} to the new version's file and make the change there, and add the new version "
    f"to telemetry/mappings/voice.yaml (it can extend {VOICE_TELEMETRY_SCHEMA_VERSION})."
)

# The same fingerprints as voice-oan-api's. Key order, spacing and the "note"
# text don't count.
RELEASED_CONTRACTS = {
    "voice.turn.v1": "26b264334db6e50d83b3f0d541cd774f9742daaa72af77473250aba50395ca0e",
}

_GENERIC_NAMES = {"data", "id", "result", "score", "status", "time", "type", "value"}
_GRANDFATHERED_NAMES = {"error.type"}
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")


def _contract():
    assert CONTRACT.exists(), f"No contract for {VOICE_TELEMETRY_SCHEMA_VERSION}. Add {CONTRACT_NAME}."
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _fingerprint(path):
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.pop("note", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _metadata_key(node):
    """The key in `<x>.metadata["key"]`, or None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "metadata"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _keys_written_in_voice():
    keys = set()
    for node in _voice_nodes():
        if isinstance(node, ast.Assign):
            keys.update(key for target in node.targets if (key := _metadata_key(target)))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and (key := _metadata_key(node.target)):
            keys.add(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "metadata"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
    return keys


class _FakeLangfuse:
    """Records what VoiceTrace hands to Langfuse for the root of one turn."""

    def __init__(self):
        self.sent = {}

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.sent.setdefault("open", kwargs)
        yield self

    @contextmanager
    def propagate_attributes(self, **kwargs):
        self.sent["propagate"] = kwargs
        yield

    def update(self, **kwargs):
        self.sent["update"] = kwargs

    def end(self):
        pass


# A value for every argument a trace block is built from, so a turn can be sent
# with every optional key in it.
_ARGUMENTS = {
    "text": "<redacted question>",
    "provider": "<redacted-provider>",
    "fallback_used": True,
    "requested_tier": "oss",
    "requested_provider": "<redacted-provider>",
    "requested_model": "<redacted-model>",
    "actual_tier": "managed",
    "actual_provider": "<redacted-provider>",
    "actual_model": "<redacted-model>",
    "first_token_committed_tier": "managed",
    "first_token_committed_provider": "<redacted-provider>",
    "first_token_committed_model": "<redacted-model>",
    "attempts": [{"tier": "oss", "status": "error"}, {"tier": "managed", "status": "ok"}],
    "signed_in": True,
    "output": "<redacted answer>",
    "new_messages": [],
    "source": "api",
    "stale": False,
    "unions": ["<redacted union>"],
    "farmer_info_chars": 1,
    "technician_info_chars": 1,
    "category": "irrelevant",
    "reason": "<redacted reason>",
    "raw_output": "<redacted>",
    "failed_open": True,
    "failed_closed": True,
}

# What the turn with every optional key is built from.
_BUILT_FROM = (ModerationVerdict, VoiceTrace.set_pretranslation, VoiceTrace.set_farmer_context, VoiceTrace.set_agent)


def _every_argument(function):
    """Every argument ``function`` takes, each with a value."""
    names = [name for name in inspect.signature(function).parameters if name != "self"]
    missing = [name for name in names if name not in _ARGUMENTS]
    assert not missing, f"Give {missing} a value in _ARGUMENTS, so what they send is checked against the contract."
    return {name: _ARGUMENTS[name] for name in names}


def _send_a_turn(monkeypatch, *, every_optional=False, text_mode="preview_hash"):
    """What VoiceTrace hands Langfuse for one turn, with the optional arguments
    left out or with every one of them given."""
    client = _FakeLangfuse()
    monkeypatch.setattr(langfuse, "propagate_attributes", client.propagate_attributes)
    monkeypatch.setattr(settings, "voice_trace_text_mode", text_mode)
    trace = VoiceTrace(
        session_id="session-redacted",
        user_id="<redacted-user-id>",
        source_lang="gu",
        target_lang="gu",
        query="<redacted question>",
        enabled=False,
    )
    trace.enabled, trace.langfuse_client = True, client
    with trace.request_context():
        trace.set_route("agent")
        if every_optional:
            trace.set_moderation(ModerationVerdict(**_every_argument(ModerationVerdict)))
            trace.set_pretranslation(**_every_argument(trace.set_pretranslation))
            trace.set_farmer_context(**_every_argument(trace.set_farmer_context))
            trace.set_agent(**_every_argument(trace.set_agent))
        else:
            trace.set_moderation(None)
            trace.set_pretranslation(text="<redacted question>", provider="<redacted-provider>", fallback_used=False)
            trace.set_farmer_context()
            trace.set_agent(signed_in=True, output="<redacted answer>")
        trace.record_emit("<redacted answer>")
        trace.finish(error=RuntimeError("<redacted>"))
    return client.sent


def _every_turn(monkeypatch):
    """A turn with the optionals left out, and one with all of them in each text mode."""
    return [_send_a_turn(monkeypatch)] + [
        _send_a_turn(monkeypatch, every_optional=True, text_mode=mode) for mode in sorted(_VALID_TEXT_MODES)
    ]


def _emitted_keys(monkeypatch):
    sent = set().union(*(turn["update"]["metadata"] for turn in _every_turn(monkeypatch)))
    return sent | _keys_written_in_voice()


# Blocks keyed by name (a counter, a mark, a stage), which the contract doesn't list.
_KEYED_BY_NAME = {"counters", "timings_ms", "stage_totals_ms"}


def _block_keys_written_in_code():
    """Keys of the blocks the code fills in itself: `.metadata["block"] = {...}`
    and `.set_nudge(key=...)`."""
    blocks = {}
    for node in _voice_nodes():
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if block := _metadata_key(target):
                    blocks.setdefault(block, set()).update(
                        key.value for key in node.value.keys if isinstance(key, ast.Constant)
                    )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_nudge":
            blocks.setdefault("nudge", set()).update(keyword.arg for keyword in node.keywords if keyword.arg)
    return blocks


def _block_keys_sent(monkeypatch):
    """Every key each metadata block can be sent with, from every turn and the code."""
    blocks = _block_keys_written_in_code()
    for turn in _every_turn(monkeypatch):
        for block, value in turn["update"]["metadata"].items():
            if isinstance(value, dict) and block not in _KEYED_BY_NAME:
                blocks.setdefault(block, set()).update(value)
    return blocks


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
        _outcomes_in(_voice_nodes())
        | _answer_outcomes_in(_voice_nodes())
        | _staleness_outcomes()
        | {_OUTCOMES.get(outcome, outcome) for outcome in _turn_outcomes_in_run_turn()}
    )


def _score_names_in(nodes):
    """Names given to Langfuse scores, e.g. score_current_trace(name="pipeline_profile")."""
    return {
        keyword.value.value
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"score_current_trace", "score_current_span", "create_score"}
        for keyword in node.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str)
    }


_TRACE_PLUMBING = {"as_type", "end_on_exit", "name", "trace_name", "metadata"}


def _trace_fields_sent(sent):
    fields = {**sent["open"], **sent["propagate"], **sent["update"]}
    return {name: value for name, value in fields.items() if name not in _TRACE_PLUMBING}


def test_contract_matches_the_stamped_schema_version():
    assert _contract()["schema_version"] == VOICE_TELEMETRY_SCHEMA_VERSION


def test_released_contracts_never_change():
    for version, fingerprint in RELEASED_CONTRACTS.items():
        path = CONTRACTS / f"{version}.json"
        assert path.exists(), f"{version} is released: keep telemetry/contracts/{version}.json, old traces follow it."
        # Compared as a bool so a failure doesn't print the new fingerprint to paste in.
        unchanged = _fingerprint(path) == fingerprint
        assert unchanged, (
            f"telemetry/contracts/{version}.json is released and can't change: traces already in Langfuse "
            f"follow it. Undo the edit and put the change in a new version. {NEW_VERSION}"
        )


def test_names_are_specific_snake_case():
    contract = _contract()
    names = (
        contract["trace_fields"]
        + contract["metadata_keys"]
        + contract["scores"]
        + [f"{block}.{key}" for block, keys in contract["nested_keys"].items() for key in keys]
    )
    unclear = [
        name
        for name in names
        if name not in _GRANDFATHERED_NAMES
        and name != "amul.schema_version"
        and (
            not all(_SNAKE_CASE.fullmatch(part) for part in name.split("."))
            or name.split(".")[-1] in _GENERIC_NAMES
        )
    ]

    assert not unclear, f"Unclear names {unclear}."


def test_no_contract_key_was_renamed_or_removed(monkeypatch):
    missing = set(_contract()["metadata_keys"]) - _emitted_keys(monkeypatch)

    assert not missing, f"Voice traces no longer send {sorted(missing)}. {NEW_VERSION}"


def test_new_keys_are_listed_in_the_contract(monkeypatch):
    extra = _emitted_keys(monkeypatch) - set(_contract()["metadata_keys"])

    assert not extra, f"New metadata keys {sorted(extra)}. {NEW_VERSION}"


def test_turns_are_still_sent_on_the_contract_root(monkeypatch):
    sent = _send_a_turn(monkeypatch)
    names = {sent["open"]["name"], sent["propagate"]["trace_name"]}

    assert names == {_contract()["root"]}, f"Voice turns are now sent as {sorted(names)}. {NEW_VERSION}"


def test_no_trace_field_was_dropped(monkeypatch):
    fields = _trace_fields_sent(_send_a_turn(monkeypatch))
    missing = [field for field in _contract()["trace_fields"] if not fields.get(field)]

    assert not missing, f"Voice traces no longer send the trace fields {missing}. {NEW_VERSION}"


def test_no_new_trace_field(monkeypatch):
    extra = set(_trace_fields_sent(_send_a_turn(monkeypatch))) - set(_contract()["trace_fields"])

    assert not extra, f"Voice traces now send the trace fields {sorted(extra)}. {NEW_VERSION}"


def test_no_key_inside_a_block_was_renamed_or_removed(monkeypatch):
    sent = _block_keys_sent(monkeypatch)
    changed = {
        block: sorted(missing)
        for block, keys in _contract()["nested_keys"].items()
        if (missing := set(keys) - sent.get(block, set()))
    }

    assert not changed, f"Voice traces no longer send these keys inside metadata blocks: {changed}. {NEW_VERSION}"


def test_no_new_key_inside_a_block(monkeypatch):
    listed = _contract()["nested_keys"]
    added = {
        block: sorted(extra)
        for block, keys in _block_keys_sent(monkeypatch).items()
        if (extra := keys - set(listed.get(block, ())))
    }

    assert not added, f"New keys inside metadata blocks: {added}. {NEW_VERSION}"


def test_every_argument_a_block_is_built_from_is_sent():
    """A key sent only when its argument is given is still checked: the turn with
    every optional key passes every argument there is."""
    for function in _BUILT_FROM:
        _every_argument(function)


def test_scores_match_the_contract():
    listed = set(_contract()["scores"])
    found = _score_names_in(_voice_nodes())

    assert found == listed, f"Voice sends the scores {sorted(found)}, the contract lists {sorted(listed)}. {NEW_VERSION}"


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
        f"New outcomes {sorted(found - listed)}. {NEW_VERSION} Also give each one a bucket in "
        "voice_outcome_vocabulary in telemetry/eras.yaml, or the adapters count it as unclassified."
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
