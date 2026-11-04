"""The dev test case API (/api/test-cases/): keep, list and label cases, with a fake
Langfuse client and a fake model. No network."""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.auth.jwt_auth import get_chat_user  # noqa: E402
from app.config import settings  # noqa: E402
from app.routers import test_cases, ui_flags  # noqa: E402
from app.services import test_cases as cases  # noqa: E402

CASE = {
    "schema_version": 1,
    "source": "chat",
    "submitted_by": "suraj",
    "tags": ["veterinary_advisory"],
    "description": "Farmer on 98765 43210 asks about a leg swollen since 2025-11-02.",
    "turns": [
        {"role": "user", "content": "My buffalo has a swollen leg, call 9876543210"},
        {"role": "assistant", "content": "Please call a vet.", "original_content": "Give paracetamol."},
    ],
}


class NotFound(Exception):
    status_code = 404


class FakeLangfuse:
    """Like Langfuse: a write keeps the fields it doesn't send, and moves createdAt."""

    def __init__(self):
        self.items = {}
        self.api = SimpleNamespace(dataset_items=SimpleNamespace(get=self.get_item, create=self.write))

    def add(self, item_id, day, input, metadata=None, status="ACTIVE"):
        created_at = datetime(2026, 11, day, tzinfo=timezone.utc)
        self.items[item_id] = SimpleNamespace(
            id=item_id, input=input, metadata=metadata, status=status, created_at=created_at
        )

    def get_item(self, item_id):
        if item_id not in self.items:
            raise NotFound()
        return self.items[item_id]

    def write(self, *, dataset_name, id, **fields):
        if id not in self.items:
            self.add(id, 30, None)
        moved = datetime(2026, 11, 30, tzinfo=timezone.utc)
        self.items[id] = SimpleNamespace(**{**vars(self.items[id]), **fields, "created_at": moved})

    def create_dataset(self, **_):
        pass

    def get_dataset(self, name, fetch_items_page_size=50, version=None):
        return SimpleNamespace(items=list(self.items.values()))


@pytest.fixture
def setup(monkeypatch):
    langfuse, prompts = FakeLangfuse(), []
    monkeypatch.setattr(cases, "get_langfuse_client", lambda: langfuse)
    monkeypatch.setattr(settings, "test_cases_raw_dataset", "cases")
    monkeypatch.setenv("SCENARIO_GEMMA_BASE_URL", "https://litellm.example")
    monkeypatch.setenv("LITELLM_API_KEY", "key")
    monkeypatch.setattr(cases, "_tried", {})
    monkeypatch.setattr(cases, "_waiting", SimpleNamespace(put=lambda job: cases.label_case(*job)))  # label right away
    reply = {"text": json.dumps({"group": "vet-health", "answer_type": "vet"})}
    monkeypatch.setattr(cases, "_complete", lambda prompt: prompts.append(prompt) or reply["text"])
    app = FastAPI()
    app.include_router(test_cases.router, prefix="/api")
    app.include_router(ui_flags.router, prefix="/api")
    app.dependency_overrides[get_chat_user] = lambda: {"phone": "1"}
    return TestClient(app), langfuse, prompts, reply


def test_a_case_is_kept_as_sent_and_labelled_once(setup):
    client, langfuse, prompts, _ = setup
    first = client.post("/api/test-cases/", json=CASE)
    cases._tried.clear()  # so only the stored label can stop a second model call
    again = client.post("/api/test-cases/", json=dict(reversed(list(CASE.items()))))

    assert first.status_code == 202 and first.json()["status"] == "received"
    assert again.json()["id"] == first.json()["id"] and len(langfuse.items) == 1
    item = langfuse.items[first.json()["id"]]
    assert item.input == CASE and item.metadata["received_at"] and item.metadata["label"]["labelled_at"]
    assert (item.metadata["label"]["group"], item.metadata["label"]["answer_type"]) == ("vet-health", "vet")
    assert len(prompts) == 1
    assert "9876543210" not in prompts[0] and "******3210" in prompts[0] and "2025-11-02" in prompts[0]
    assert "[the tester corrected this; the bot first said: Give paracetamol.]" in prompts[0]


@pytest.mark.parametrize(
    "body, code",
    [("[1, 2]", 400), ("{not json", 400), ('{"turns": [1], "x": NaN}', 400), ('{"turns": []}', 422), ("{}", 422)],
)
def test_a_bad_body_is_refused(setup, body, code):
    client, langfuse, _, _ = setup
    sent = client.post("/api/test-cases/", content=body, headers={"Content-Type": "application/json"})
    assert sent.status_code == code and sent.json()["detail"] and not langfuse.items


def test_no_store_is_a_503(setup, monkeypatch):
    client = setup[0]
    monkeypatch.setattr(settings, "test_cases_raw_dataset", None)
    assert client.post("/api/test-cases/", json=CASE).status_code == 503
    assert client.get("/api/test-cases/").status_code == 503


def test_the_list_is_the_dataset_with_numbers_cut(setup):
    client, langfuse, prompts, _ = setup
    trace = {"role": "assistant", "content": "ok", "trace_id": "4f3b1234567e8a7d4f3b9c0e1d2a3b4c"}
    photo = SimpleNamespace(reference_string="@@@langfuseMedia:type=image/png|id=m1|source=bytes@@@")
    turns = [{"role": "user", "content": "call 9876543210"}, trace]
    langfuse.add("old", 1, {"turns": turns, "photo": photo}, {"received_at": "2026-11-01"})
    langfuse.add("gone", 2, {"turns": []}, status="ARCHIVED")
    langfuse.add("hand-made", 2, {"turns": []})  # no received_at: its createdAt counts
    label = {"group": "milk", "answer_type": "answer"}
    langfuse.add("new", 3, {"turns": []}, {"received_at": "2026-11-03", "label": label})

    listed = client.get("/api/test-cases/")

    shown = listed.json()["items"]
    assert [case["id"] for case in shown] == ["new", "hand-made", "old"]
    assert shown[0] == {"id": "new", "input": {"turns": []}, "metadata": {"received_at": "2026-11-03", "label": label}}
    assert "9876543210" not in listed.text and "******3210" in listed.text
    assert trace["trace_id"] in listed.text and photo.reference_string in listed.text
    old = langfuse.items["old"]
    assert old.metadata["label"]["group"] == "vet-health" and old.metadata["received_at"] == "2026-11-01"
    assert [case["id"] for case in client.get("/api/test-cases/").json()["items"]] == ["new", "hand-made", "old"]
    assert langfuse.items["hand-made"].metadata["received_at"] == "2026-11-02T00:00:00+00:00"


@pytest.mark.parametrize("text", ["not json", '{"group": "weather", "answer_type": "answer"}', '{"group": "ai"}'])
def test_a_bad_reply_leaves_no_label_and_waits(setup, text):
    client, langfuse, prompts, reply = setup
    reply["text"] = text
    case_id = client.post("/api/test-cases/", json=CASE).json()["id"]
    client.get("/api/test-cases/")
    assert "label" not in langfuse.items[case_id].metadata and len(prompts) == 1


def test_a_case_archived_while_labelled_stays_archived(setup, monkeypatch):
    client, langfuse, _, _ = setup
    langfuse.add("race", 1, {"turns": []})

    def archive_then_answer(prompt):
        langfuse.add("race", 1, {"turns": []}, status="ARCHIVED")  # a fresh read, as Langfuse would give
        return json.dumps({"group": "milk", "answer_type": "answer"})

    monkeypatch.setattr(cases, "_complete", archive_then_answer)
    client.get("/api/test-cases/")
    assert langfuse.items["race"].status == "ARCHIVED" and langfuse.items["race"].metadata is None


def test_a_label_leaves_the_rest_of_the_item_alone(setup):
    client, langfuse, prompts, _ = setup
    langfuse.add("curated", 1, {"turns": []}, {"received_at": "2026-11-01", "note": "kanav"})
    langfuse.items["curated"].expected_output = "Ask for the species first."
    langfuse.add("note", 2, {"turns": []}, "from kanav: keep")  # no label goes on metadata that isn't an object
    client.get("/api/test-cases/")
    item = langfuse.items["curated"]
    assert item.metadata == {"received_at": "2026-11-01", "note": "kanav", "label": item.metadata["label"]}
    assert item.expected_output == "Ask for the species first." and item.input == {"turns": []}
    assert langfuse.items["note"].metadata == "from kanav: keep" and len(prompts) == 1


def test_a_deeply_nested_case_does_not_break_the_list(setup):
    client = setup[0]
    deep = "call 9876543210"
    for _ in range(3000):
        deep = [deep]
    assert client.post("/api/test-cases/", json={**CASE, "extra": deep}).status_code == 202
    listed = client.get("/api/test-cases/")
    assert listed.status_code == 200 and "9876543210" not in listed.text


def test_options_and_flags(setup):
    client = setup[0]
    options = client.get("/api/test-cases/options/").json()
    assert {"testers", "case_types", "tags", "groups", "answer_types"} <= set(options)
    assert "general" in {tag["id"] for tag in options["tags"]}  # the screen asks for other_topic on it
    assert set(client.get("/api/ui-flags/").json()) == {"test_case_entry", "doctor_persona_selector"}
