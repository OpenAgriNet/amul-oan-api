"""Labelling test cases as they come in on dev (#488).

The model and the store are fakes. A label is kept with its case, a bad reply or a
failure is logged without turn text and doesn't stop the worker, a case is queued
once while it waits and labelled once, a failed one is tried again only after the
retry window, and without the LiteLLM settings there is no labeller. No network.
"""
import json
import logging
import threading

import pytest

from app.services import test_case_labelling
from app.services.test_case_labelling import CaseLabeller, get_case_labeller
from app.services.test_case_store import StoredCase

BODY = {
    "schema_version": 1,
    "source": "chat",
    "submitted_by": "kanav",
    "persona": "farmer",
    "language": "gu",
    "description": "Farmer on 98765 43210 wants AI for the cow.",
    "case_types": ["new_flow"],
    "tags": ["ait_booking"],
    "other_topic": "call 98765 43210",
    "turns": [
        {"role": "user", "content": "મારી ગાયને બીજદાન કરાવવું છે, મારો નંબર 9876543210"},
        {"role": "assistant", "content": "Which technician?", "original_content": "When should he come?"},
    ],
}
LABEL = {"group": "ai", "answer_type": "ask:technician"}


class _Store:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save_label(self, case, label):
        if self.fail:
            raise RuntimeError("langfuse is down")
        self.saved.append((case.id, label))
        return True


def _case(case_id="c1", label=None):
    return StoredCase(id=case_id, received_at="2026-11-04T10:00:00+00:00", raw=BODY, label=label)


def test_a_case_gets_its_label_kept_with_it():
    prompts = []
    store = _Store()
    labeller = CaseLabeller(lambda prompt: prompts.append(prompt) or json.dumps(LABEL), "gemma-4-31b-it")

    record = labeller.label(store, _case())

    assert store.saved == [("c1", record)]
    assert {key: record[key] for key in ("group", "answer_type", "model")} == {**LABEL, "model": "gemma-4-31b-it"}
    assert record["labelled_at"].startswith("20")
    prompt = prompts[0]
    assert "9876543210" not in prompt and "98765 43210" not in prompt
    assert "Bot: Which technician? [the tester corrected this; the bot first said: When should he come?]" in prompt
    assert "tester_tags: ait_booking" in prompt and "other_topic: call ******3210" in prompt
    assert "What the tester says the case checks: Farmer on ******3210 wants AI for the cow." in prompt


def test_an_odd_old_body_is_still_labelled():
    odd = {"turns": [{"role": "user", "content": "Hi"}, {"role": "bot", "content": "x"}, "text"], "tags": "ai", "persona": 3}
    prompts = []

    CaseLabeller(lambda prompt: prompts.append(prompt) or json.dumps(LABEL), "m").label(
        _Store(), StoredCase(id="odd", received_at="t", raw=odd)
    )

    assert prompts[0].endswith("Conversation:\nFarmer: Hi")
    assert "Hints:" not in prompts[0]


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("not json", "not_json"),
        (json.dumps({"group": "મારી ગાય", "answer_type": "answer"}), "group"),
        (json.dumps({**LABEL, "મારી ગાયને બીજદાન કરાવવું છે": "book"}), "extra_key"),
        (json.dumps({"મારી ગાયને બીજદાન": LABEL}), "group"),
        ("[1, 2]", "not_a_label"),
    ],
)
def test_a_bad_reply_is_logged_without_the_turns_and_nothing_is_kept(reply, reason, caplog):
    store = _Store()

    with caplog.at_level(logging.WARNING):
        assert CaseLabeller(lambda prompt: reply, "m").label(store, _case()) is None

    assert store.saved == []
    assert f"test_case_label failed id=c1 reason={reason}" in caplog.text
    assert "મારી" not in caplog.text and "9876543210" not in caplog.text


def test_a_model_or_store_failure_is_logged_not_raised(caplog):
    def down(prompt):
        raise ConnectionError("no route")

    with caplog.at_level(logging.WARNING):
        assert CaseLabeller(down, "m").label(_Store(), _case()) is None
        assert CaseLabeller(lambda prompt: json.dumps(LABEL), "m").label(_Store(fail=True), _case()) is None

    assert "reason=ConnectionError" in caplog.text
    assert "test_case_label not_saved id=c1 error=RuntimeError" in caplog.text


def test_a_labelled_case_is_not_labelled_again(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(test_case_labelling.time, "monotonic", lambda: now[0])
    calls = []
    store = _Store()
    labeller = CaseLabeller(lambda prompt: calls.append(prompt) or json.dumps(LABEL), "m")

    assert labeller.enqueue(store, _case("c1")) is True
    assert labeller.enqueue(store, _case("c2", label=LABEL)) is False
    assert labeller.wait()
    now[0] += 10 * test_case_labelling.RETRY_SECONDS

    # A list read cached before the label landed still shows c1 without one.
    assert labeller.enqueue(store, _case("c1")) is False
    assert [case_id for case_id, _ in store.saved] == ["c1"] and len(calls) == 1


def test_a_queued_case_is_queued_once_however_long_it_waits(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(test_case_labelling.time, "monotonic", lambda: now[0])
    gate, calls = threading.Event(), []

    def slow(prompt):
        gate.wait(5)
        calls.append(prompt)
        return json.dumps(LABEL)

    labeller = CaseLabeller(slow, "m")
    store = _Store()
    assert [labeller.enqueue(store, _case(case_id)) for case_id in ("c1", "c2")] == [True, True]

    for _ in range(3):  # list reads while the backlog is still being labelled
        now[0] += test_case_labelling.RETRY_SECONDS
        assert [labeller.enqueue(store, _case(case_id)) for case_id in ("c1", "c2")] == [False, False]
    gate.set()

    assert labeller.wait()
    assert len(calls) == 2


def test_a_failed_case_is_tried_again_only_after_the_retry_window(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(test_case_labelling.time, "monotonic", lambda: now[0])
    replies = iter(["not json", json.dumps(LABEL)])
    store = _Store()
    labeller = CaseLabeller(lambda prompt: next(replies), "m")

    assert labeller.enqueue(store, _case("c1")) is True
    assert labeller.wait()
    now[0] += test_case_labelling.RETRY_SECONDS - 1
    assert labeller.enqueue(store, _case("c1")) is False

    now[0] += 1
    assert labeller.enqueue(store, _case("c1")) is True
    assert labeller.wait()
    assert [case_id for case_id, _ in store.saved] == ["c1"]


def test_a_case_gone_or_archived_meanwhile_is_skipped_not_retried(caplog):
    class _Gone(_Store):
        def save_label(self, case, label):
            return False

    labeller = CaseLabeller(lambda prompt: json.dumps(LABEL), "m")

    with caplog.at_level(logging.INFO):
        assert labeller.label(_Gone(), _case()) is None

    assert "test_case_label skipped id=c1 reason=gone_or_archived" in caplog.text
    assert labeller.enqueue(_Gone(), _case()) is False


def test_the_worker_goes_on_after_label_itself_breaks(monkeypatch, caplog):
    store = _Store()
    labeller = CaseLabeller(lambda prompt: json.dumps(LABEL), "m")
    real_label = labeller.label

    def label(store, case):
        if case.id == "bad":
            raise KeyError("x")
        return real_label(store, case)

    monkeypatch.setattr(labeller, "label", label)

    with caplog.at_level(logging.WARNING):
        labeller.enqueue(store, _case("bad"))
        labeller.enqueue(store, _case("good"))
        assert labeller.wait()

    assert [case_id for case_id, _ in store.saved] == ["good"]
    assert "test_case_label crashed id=bad error=KeyError" in caplog.text


def test_the_worker_goes_on_after_a_case_fails():
    replies = iter(["not json", json.dumps(LABEL)])
    store = _Store()
    labeller = CaseLabeller(lambda prompt: next(replies), "m")

    labeller.enqueue(store, _case("bad"))
    labeller.enqueue(store, _case("good"))
    assert labeller.wait()

    assert [case_id for case_id, _ in store.saved] == ["good"]


def test_there_is_a_labeller_only_with_the_litellm_settings(monkeypatch, caplog):
    monkeypatch.setattr(test_case_labelling, "_labeller", None)
    monkeypatch.setattr(test_case_labelling, "_said_off", False)
    monkeypatch.delenv("SCENARIO_GEMMA_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "key-1")

    with caplog.at_level(logging.INFO):
        assert get_case_labeller() is None and get_case_labeller() is None
        assert caplog.text.count("test_case_label off") == 1

        monkeypatch.setenv("SCENARIO_GEMMA_BASE_URL", "https://litellm.example")
        labeller = get_case_labeller()

    assert isinstance(labeller, CaseLabeller) and get_case_labeller() is labeller
    assert "test_case_label on model=gemma-4-31b-it" in caplog.text and "key-1" not in caplog.text


def test_a_litellm_client_that_cant_be_made_means_no_labeller_not_an_error(monkeypatch, caplog):
    def broken():
        raise FileNotFoundError("ca bundle")

    monkeypatch.setattr(test_case_labelling, "_labeller", None)
    monkeypatch.setattr(test_case_labelling, "_said_off", False)
    monkeypatch.setattr(test_case_labelling, "LiteLLM", broken)

    with caplog.at_level(logging.WARNING):
        assert get_case_labeller() is None

    assert "test_case_label off: the LiteLLM client failed error=FileNotFoundError" in caplog.text
