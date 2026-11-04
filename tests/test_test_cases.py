"""The dev test case API (/api/test-cases/) with a fake Langfuse client. No network."""
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.auth.jwt_auth import get_chat_user  # noqa: E402
from app.config import settings  # noqa: E402
from app.routers import test_cases  # noqa: E402
from app.services import test_cases as cases  # noqa: E402

CASE = {
    "source": "chat",
    "submitted_by": "suraj",
    "turns": [{"role": "user", "content": "My buffalo has a swollen leg, call 9876543210"}],
}


class NotFound(Exception):
    status_code = 404


class FakeLangfuse:
    def __init__(self):
        self.items = {}
        self.async_api = SimpleNamespace(
            datasets=SimpleNamespace(create=self.create_dataset),
            dataset_items=SimpleNamespace(get=self.get, create=self.create, list=self.list),
        )

    async def create_dataset(self, **_):
        pass

    async def get(self, item_id):
        if item_id not in self.items:
            raise NotFound()
        return self.items[item_id]

    async def create(self, *, dataset_name, id, input=None, metadata=None, status="ACTIVE"):
        created_at = datetime.now(timezone.utc)
        self.items[id] = SimpleNamespace(id=id, input=input, metadata=metadata, status=status, created_at=created_at)

    async def list(self, *, dataset_name, version, page, limit):
        return SimpleNamespace(data=list(self.items.values()), meta=SimpleNamespace(total_pages=1))


@pytest.fixture
def setup(monkeypatch):
    langfuse = FakeLangfuse()
    monkeypatch.setattr(cases, "get_langfuse_client", lambda: langfuse)
    monkeypatch.setattr(settings, "test_cases_raw_dataset", "cases")
    monkeypatch.delenv("SCENARIO_GEMMA_BASE_URL", raising=False)
    app = FastAPI()
    app.include_router(test_cases.router, prefix="/api")
    app.dependency_overrides[get_chat_user] = lambda: {"phone": "1"}
    return TestClient(app), langfuse


@pytest.mark.parametrize(
    "body, code",
    [("[1, 2]", 400), ("{not json", 400), ('{"turns": [1], "x": NaN}', 400), ('{"turns": []}', 422), ("{}", 422)],
)
def test_a_bad_body_is_refused(setup, body, code):
    client, langfuse = setup
    sent = client.post("/api/test-cases/", content=body, headers={"Content-Type": "application/json"})
    assert sent.status_code == code and sent.json()["detail"] and not langfuse.items


def test_the_same_case_sent_twice_gets_one_id(setup):
    client, langfuse = setup
    first = client.post("/api/test-cases/", json=CASE)
    again = client.post("/api/test-cases/", json=dict(reversed(list(CASE.items()))))
    assert first.status_code == again.status_code == 202
    assert again.json()["id"] == first.json()["id"] and len(langfuse.items) == 1
    assert langfuse.items[first.json()["id"]].input == CASE


def test_the_list_hides_archived_cases_and_masks_numbers(setup):
    client, langfuse = setup
    kept = client.post("/api/test-cases/", json=CASE).json()["id"]
    gone = SimpleNamespace(id="gone", input={"turns": []}, metadata={}, status="ARCHIVED", created_at=None)
    langfuse.items["gone"] = gone
    listed = client.get("/api/test-cases/")
    assert [case["id"] for case in listed.json()["items"]] == [kept]
    assert "9876543210" not in listed.text and "******3210" in listed.text
