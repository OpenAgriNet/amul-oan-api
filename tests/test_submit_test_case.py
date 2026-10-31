"""Dev-only test case intake (POST /api/test-cases/, #487), v1 contract.

The route takes one case in the v1 shape (chat, manual or bulk), turns away
anything outside it, stores the body as it came under an id made from the body,
and keeps turn text out of the logs. The store is a fake here; the Langfuse store
is checked against a fake client. No network.
"""
import asyncio
import copy
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.models.case_options import CASE_OPTIONS
from app.models.test_cases import MAX_USER_TURNS
from app.routers import test_cases
from app.services import test_case_store
from app.services.test_case_store import LangfuseCaseStore, get_case_store

REPO = Path(__file__).resolve().parents[1]
QUESTION = "My buffalo is 7 months pregnant and has a swollen leg. Can I give her painkillers?"
TRACE_ID = "9b2f4c1e8a7d4f3b9c0e1d2a3b4c5d6e"

# A chat case as the screen sends it: one answer the tester corrected.
CHAT_CASE = {
    "schema_version": 1,
    "source": "chat",
    "submitted_by": "tester_04",
    "persona": "farmer",
    "language": "en",
    "description": "Pregnant buffalo with a swollen leg: no painkillers without a vet.",
    "case_types": ["should_not_happen"],
    "tags": ["veterinary_advisory"],
    "turns": [
        {"role": "user", "content": QUESTION},
        {
            "role": "assistant",
            "content": "Please do not give any painkiller without a vet.",
            "original_content": "Give her paracetamol.",
            "trace_id": TRACE_ID,
        },
    ],
    "chat": {"session_id": "0b9a1f8e-4c1d-4f2a-9f7e-2d3c4b5a6f70", "removed_turn_count": 0},
}


def _chat(**changes):
    body = copy.deepcopy(CHAT_CASE)
    body.update(changes)
    return body


def _manual(**changes):
    body = {key: value for key, value in copy.deepcopy(CHAT_CASE).items() if key != "chat"}
    body["source"] = "manual"
    body["turns"] = [{"role": turn["role"], "content": turn["content"]} for turn in body["turns"]]
    body.update(changes)
    return body


def _bulk(**changes):
    body = _manual(source="bulk", bulk={"case_id": "vet-001"})
    body.update(changes)
    return body


def _without(body, *keys):
    return {key: value for key, value in body.items() if key not in keys}


def _turns(questions):
    turns = []
    for i in range(questions):
        turns += [{"role": "user", "content": f"Question {i}"}, {"role": "assistant", "content": f"Answer {i}"}]
    return turns


class _FakeStore:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save(self, *, case_id, raw, received_at):
        if self.fail:
            raise RuntimeError("langfuse is down")
        self.saved.append({"case_id": case_id, "raw": raw, "received_at": received_at})


@pytest.fixture
def store():
    return _FakeStore()


def _client(store, *, signed_in=True):
    app = FastAPI()
    app.include_router(test_cases.router, prefix=settings.api_prefix)
    if signed_in:
        app.dependency_overrides[get_chat_user] = lambda: {"sub": "anon-test"}
    app.dependency_overrides[get_case_store] = lambda: store
    return TestClient(app)


def _post(client, body):
    return client.post("/api/test-cases/", json=body)


def _post_raw(client, raw):
    return client.post("/api/test-cases/", content=raw, headers={"Content-Type": "application/json"})


def test_a_chat_case_is_stored_exactly_as_sent(store):
    response = _post(_client(store), CHAT_CASE)

    assert response.status_code == 202
    case_id = response.json()["id"]
    assert response.json() == {"id": case_id, "status": "received"}
    uuid.UUID(case_id)
    assert len(store.saved) == 1
    assert store.saved[0]["case_id"] == case_id
    assert store.saved[0]["raw"] == CHAT_CASE
    assert store.saved[0]["received_at"].endswith("+00:00")


@pytest.mark.parametrize(
    "body",
    [
        _chat(turns=[]),
        _chat(turns=[{"role": "assistant", "content": "Hello"}]),
        _chat(turns=[{"role": "user", "content": "Hello"}]),
        _chat(turns=[{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hello"}]),
        _chat(turns=[{"role": "user", "content": "   "}, {"role": "assistant", "content": "Hi"}]),
        _chat(turns=[{"role": "user", "content": "\u3000\u00a0"}, {"role": "assistant", "content": "Hi"}]),
        _chat(turns=[{"role": "farmer", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]),
        _chat(turns=_turns(MAX_USER_TURNS + 1)),
        _chat(description="  too short, 19 ch  "),
        _chat(description="Nineteen characters\u3000"),
        _without(CHAT_CASE, "description"),
        _chat(source="email"),
        _without(CHAT_CASE, "source"),
        _chat(persona="vet"),
        _without(CHAT_CASE, "schema_version"),
        _chat(schema_version=2),
        _chat(client_submission_id="c-1"),
        _chat(case_type="wrong_answer"),
        _chat(metadata={"persona": "farmer"}),
        _chat(case_types="wrong_answer"),
        _chat(case_types=["Wrong answer"]),
        _chat(submitted_by="Tester 4"),
        _chat(language="gujarati"),
        _chat(test_account="98765 43210"),
        _chat(other_topic="Fodder storage"),
        _chat(tags=["general"], other_topic="   "),
        _chat(turns=[{"role": "user", "content": "Hi", "trace_id": TRACE_ID}, {"role": "assistant", "content": "Hello"}]),
        _chat(turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "trace_id": "abc123"}]),
        _chat(turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "original_content": " "}]),
        _chat(turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "edited": True}]),
        _without(CHAT_CASE, "chat"),
        _chat(chat={"session_id": "s-1", "removed_turn_count": -1}),
        _manual(chat=CHAT_CASE["chat"]),
        _manual(turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "original_content": "Hey"}]),
        _manual(turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "trace_id": TRACE_ID}]),
        _without(_bulk(), "bulk"),
        _chat(schema_version=True),
        _chat(schema_version=1.0),
        _chat(schema_version="1"),
        _chat(chat={"session_id": "s-1", "removed_turn_count": "3"}),
        _chat(chat={"session_id": "s-1", "removed_turn_count": 3.0}),
        _chat(chat={"session_id": "s-1", "removed_turn_count": 1001}),
        _manual(test_account="9876543210"),
        _chat(submitted_by="9876543210"),
        _chat(test_account="f1"),
    ],
    ids=[
        "no turns",
        "no farmer question",
        "no bot answer",
        "starts with the bot",
        "empty turn",
        "turn of non-ASCII spaces",
        "unknown role",
        "too many questions",
        "short description",
        "short description with an ideographic space",
        "no description",
        "unknown source",
        "no source",
        "unknown persona",
        "no schema_version",
        "schema_version 2",
        "unknown field",
        "old case_type",
        "old metadata",
        "case_types not a list",
        "free-text case type",
        "submitted_by not an id",
        "language not a code",
        "test_account not an alias",
        "other_topic without the general tag",
        "empty other_topic",
        "trace_id on a farmer turn",
        "trace_id not a trace id",
        "empty original_content",
        "old edited flag",
        "chat case without chat",
        "negative removed_turn_count",
        "manual case with chat",
        "manual case with original_content",
        "manual case with trace_id",
        "bulk case without bulk",
        "schema_version true",
        "schema_version 1.0",
        "schema_version as text",
        "removed_turn_count as text",
        "removed_turn_count as a float",
        "removed_turn_count too big",
        "test_account that is a phone number",
        "submitted_by that is a phone number",
        "chat case with test_account",
    ],
)
def test_a_body_outside_the_contract_is_a_422(store, body):
    response = _post(_client(store), body)

    assert response.status_code == 422
    assert store.saved == []


@pytest.mark.parametrize(
    "body",
    [
        _chat(turns=[
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "content": "Yes"},
        ]),
        _chat(turns=_turns(MAX_USER_TURNS)),
        _chat(case_types=[], tags=[]),
        _chat(case_types=["a_type_from_a_newer_list"], tags=["a_tag_from_a_newer_list"]),
        _chat(tags=["general"], other_topic="Fodder storage"),
        _chat(persona="doctor"),
        _chat(chat={"session_id": "s-1", "removed_turn_count": 3}),
        _manual(),
        _manual(test_account="f1"),
        _bulk(),
        # 18 code points, but 21 in JavaScript's .length, which is what the screen counts.
        _chat(description="Buffalo 🐃 fever 🤒!!"),
    ],
    ids=[
        "same role twice",
        "exactly the cap",
        "no types or tags",
        "ids from a newer list",
        "other_topic with the general tag",
        "doctor persona",
        "removed turns",
        "manual case",
        "manual case with a test account",
        "bulk case",
        "emoji counted the way the screen counts them",
    ],
)
def test_what_the_contract_allows_is_accepted(store, body):
    response = _post(_client(store), body)

    assert response.status_code == 202
    assert len(store.saved) == 1


def test_a_422_names_each_field_the_way_the_screen_can_show_it(store):
    body = _chat(
        turns=[{"role": "user", "content": QUESTION}, {"role": "assistant", "content": "  "}],
        client_submission_id="c-1",
    )

    response = _post(_client(store), body)

    assert response.status_code == 422
    errors = response.json()["errors"]
    assert {"loc": ["turns", 1, "content"], "msg": "A turn is empty"} in errors
    assert {"loc": ["client_submission_id"], "msg": "Extra inputs are not permitted"} in errors
    assert response.json()["detail"] == ": ".join([".".join(map(str, errors[0]["loc"])), errors[0]["msg"]])


@pytest.mark.parametrize(
    "body, loc",
    [
        (_without(CHAT_CASE, "source"), ["source"]),
        (_chat(other_topic="Fodder storage"), ["other_topic"]),
        (_chat(turns=_turns(MAX_USER_TURNS + 1)), ["turns"]),
        (_without(CHAT_CASE, "chat"), ["chat"]),
    ],
    ids=["source", "other_topic", "turn rules", "chat block"],
)
def test_case_level_rules_point_at_their_field(store, body, loc):
    response = _post(_client(store), body)

    assert [error["loc"] for error in response.json()["errors"]] == [loc]


def test_a_body_of_unknown_keys_gets_a_small_422(store):
    body = _chat(**{f"unknown_{i}": i for i in range(5000)})

    response = _post(_client(store), body)

    assert response.status_code == 422
    assert len(response.json()["errors"]) == 20
    assert len(response.content) < 4096


def test_the_same_case_sent_again_gets_the_same_id(store):
    client = _client(store)

    first = _post(client, CHAT_CASE).json()["id"]
    again = _post(client, CHAT_CASE).json()["id"]
    reordered = _post_raw(client, json.dumps(dict(reversed(list(CHAT_CASE.items())))).encode()).json()["id"]
    edited = _post(client, _chat(description=CHAT_CASE["description"] + " Also no ice packs.")).json()["id"]

    assert first == again == reordered != edited
    assert [saved["case_id"] for saved in store.saved] == [first, first, first, edited]


def test_the_id_depends_on_the_dataset(store, monkeypatch):
    client = _client(store)

    monkeypatch.setattr(settings, "test_cases_raw_dataset", "amul-test-case-submissions")
    first = _post(client, CHAT_CASE).json()["id"]
    monkeypatch.setattr(settings, "test_cases_raw_dataset", "amul-test-case-submissions-v2")
    second = _post(client, CHAT_CASE).json()["id"]

    assert first != second


def test_text_that_is_not_unicode_is_a_400(store):
    raw = json.dumps(CHAT_CASE).replace("without a vet", "without a vet \\ud83d").encode()

    response = _post_raw(_client(store), raw)

    assert response.status_code == 400
    assert store.saved == []


def test_a_body_over_the_cap_is_a_413(store, monkeypatch):
    monkeypatch.setattr(settings, "test_cases_max_body_bytes", 200)

    response = _post(_client(store), CHAT_CASE)

    assert response.status_code == 413
    assert store.saved == []


def _scope():
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/api/test-cases/", "raw_path": b"/api/test-cases/",
        "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
        "headers": [(b"host", b"test"), (b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")],
    }


def test_a_chunked_body_over_the_cap_is_cut_off_while_reading(store, monkeypatch):
    monkeypatch.setattr(settings, "test_cases_max_body_bytes", 4096)
    app = _client(store).app
    pulled = 0
    sent = []

    async def receive():
        nonlocal pulled
        pulled += 1
        if pulled > 1000:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"x" * 1024, "more_body": True}

    async def send(message):
        sent.append(message)

    asyncio.run(app(_scope(), receive, send))

    assert pulled <= 6
    assert sent[0]["status"] == 413


def test_a_client_that_drops_mid_upload_is_a_400_not_a_500(store):
    app = _client(store).app
    messages = iter([
        {"type": "http.request", "body": b'{"source": "chat"', "more_body": True},
        {"type": "http.disconnect"},
    ])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(app(_scope(), receive, send))

    assert sent[0]["status"] == 400
    assert store.saved == []


def test_a_body_full_of_bad_elements_gets_a_small_422(store):
    head = b'{"source":"chat","description":"' + b"x" * 25 + b'","turns":['
    limit = settings.test_cases_max_body_bytes
    raw = head + b",".join([b"{}"] * ((limit - len(head) - 2) // 3)) + b"]}"
    assert len(raw) <= limit

    response = _post_raw(_client(store), raw)

    assert response.status_code == 422
    assert len(response.content) < 4096


@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]", b'"a string"', b"[" * 100_000 + b"]" * 100_000])
def test_a_body_that_is_not_a_json_object_is_a_400(store, raw):
    response = _post_raw(_client(store), raw)

    assert response.status_code == 400
    assert store.saved == []


def test_without_a_token_the_route_is_a_401(store):
    response = _post(_client(store, signed_in=False), CHAT_CASE)

    assert response.status_code == 401
    assert store.saved == []


def test_with_no_store_the_rules_still_answer_first():
    client = _client(None)

    assert _post(client, _chat(turns=[])).status_code == 422
    response = _post(client, CHAT_CASE)
    assert response.status_code == 503
    assert response.json()["detail"] == "Test case store is not configured"


def test_a_store_failure_is_a_503_and_the_logs_hold_no_turn_text(caplog):
    caplog.set_level(logging.DEBUG)

    response = _post(_client(_FakeStore(fail=True)), CHAT_CASE)

    assert response.status_code == 503
    assert response.json()["detail"] == "Test case store is unavailable"
    assert QUESTION not in caplog.text
    assert "paracetamol" not in caplog.text


def test_a_stored_case_logs_no_turn_text(store, caplog):
    caplog.set_level(logging.DEBUG)

    assert _post(_client(store), CHAT_CASE).status_code == 202
    assert "submit_test_case stored" in caplog.text
    assert QUESTION not in caplog.text


# ── the dropdown lists ──────────────────────────────────────────────────────


def test_the_dropdown_lists_need_a_token(store):
    response = _client(store, signed_in=False).get("/api/test-cases/options/")

    assert response.status_code == 401


def test_the_dropdown_lists_are_served_as_they_are_kept(store):
    response = _client(store).get("/api/test-cases/options/")

    assert response.status_code == 200
    assert response.json() == CASE_OPTIONS.model_dump()
    for name in ("testers", "test_accounts", "case_types", "tags"):
        ids = [option["id"] for option in response.json()[name]]
        assert ids and len(ids) == len(set(ids)), name


def test_every_id_in_the_lists_passes_the_contract(store):
    client = _client(store)
    options = CASE_OPTIONS

    for tester in options.testers:
        assert _post(client, _chat(submitted_by=tester.id)).status_code == 202, tester.id
    for account in options.test_accounts:
        assert _post(client, _manual(test_account=account.id)).status_code == 202, account.id
    every_type = [option.id for option in options.case_types]
    every_tag = [option.id for option in options.tags]
    body = _chat(case_types=every_type, tags=every_tag, other_topic="Fodder storage")
    assert _post(client, body).status_code == 202
    assert "general" in every_tag


# ── the Langfuse store ──────────────────────────────────────────────────────


class _FakeLangfuse:
    def __init__(self):
        self.datasets = []
        self.items = []

    def create_dataset(self, *, name, description=None):
        self.datasets.append(name)

    def create_dataset_item(self, **kwargs):
        self.items.append(kwargs)


def test_the_langfuse_store_makes_the_dataset_once_and_keeps_the_body_as_input():
    client = _FakeLangfuse()
    store = LangfuseCaseStore(client, "amul-test-case-submissions")

    store.save(case_id="id-1", raw={"a": 1}, received_at="t1")
    store.save(case_id="id-2", raw={"b": 2}, received_at="t2")

    assert client.datasets == ["amul-test-case-submissions"]
    assert client.items == [
        {"dataset_name": "amul-test-case-submissions", "id": "id-1", "input": {"a": 1}, "metadata": {"received_at": "t1"}},
        {"dataset_name": "amul-test-case-submissions", "id": "id-2", "input": {"b": 2}, "metadata": {"received_at": "t2"}},
    ]


class _LangfuseLike(_FakeLangfuse):
    """Langfuse's rule: an item for a dataset that doesn't exist is an error."""

    def create_dataset_item(self, **kwargs):
        if kwargs["dataset_name"] not in self.datasets:
            raise RuntimeError("Dataset not found")
        super().create_dataset_item(**kwargs)


def test_the_langfuse_store_makes_the_dataset_again_after_it_was_deleted():
    client = _LangfuseLike()
    store = LangfuseCaseStore(client, "amul-test-case-submissions")
    store.save(case_id="id-1", raw={"a": 1}, received_at="t1")
    client.datasets.clear()  # deleted in the Langfuse UI

    with pytest.raises(RuntimeError):
        store.save(case_id="id-2", raw={"b": 2}, received_at="t2")
    store.save(case_id="id-3", raw={"c": 3}, received_at="t3")

    assert [item["id"] for item in client.items] == ["id-1", "id-3"]


def test_there_is_a_store_only_with_a_dataset_name_and_langfuse(monkeypatch):
    client = _FakeLangfuse()
    monkeypatch.setattr(test_case_store, "get_langfuse_client", lambda: client)

    monkeypatch.setattr(settings, "test_cases_raw_dataset", None)
    assert get_case_store() is None
    monkeypatch.setattr(settings, "test_cases_raw_dataset", "  ")
    assert get_case_store() is None

    monkeypatch.setattr(settings, "test_cases_raw_dataset", "amul-test-case-submissions")
    assert isinstance(get_case_store(), LangfuseCaseStore)

    monkeypatch.setattr(test_case_store, "get_langfuse_client", lambda: None)
    assert get_case_store() is None


# ── mounting ────────────────────────────────────────────────────────────────


def test_the_route_is_registered_only_when_switched_on():
    def _paths(enabled):
        script = "import main; print(sorted(r.path for r in main.app.routes if 'test-case' in r.path))"
        env = {**os.environ, "OPENAI_API_KEY": "test-key", "TEST_CASES_ROUTE_ENABLED": enabled}
        out = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout.strip().splitlines()[-1]

    assert _paths("false") == "[]"
    assert _paths("true") == "['/api/test-cases/', '/api/test-cases/options/']"
