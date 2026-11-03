"""The test case review list (GET /api/test-cases/ and /coverage/, #517).

The list reads the raw store back: newest first, archived items left out, old
shapes read leniently, phone and Aadhaar numbers masked, filters and pages applied,
and nothing of the turns in the logs. The store is a fake here; the Langfuse store
is checked against a fake client. No network.
"""
import copy
import logging
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.models.case_options import CASE_OPTIONS
from app.routers import test_cases
from app.services import test_case_store
from app.services.personal_data import mask_numbers
from app.services.test_case_store import LangfuseCaseStore, StoredCase, get_case_store

BASE = {
    "schema_version": 1,
    "source": "manual",
    "submitted_by": "suraj",
    "persona": "farmer",
    "language": "en",
    "description": "Pregnant buffalo with a swollen leg: no painkillers without a vet.",
    "case_types": ["wrong_answer"],
    "tags": ["veterinary_advisory"],
    "turns": [
        {"role": "user", "content": "My buffalo has a swollen leg. Can I give painkillers?"},
        {"role": "assistant", "content": "Please call your vet first."},
    ],
}


def _case(case_id, received_at, **changes):
    raw = copy.deepcopy(BASE)
    raw.update(changes)
    return StoredCase(id=case_id, received_at=received_at, raw=raw)


class _FakeStore:
    def __init__(self, cases=(), fail=False):
        self.cases = list(cases)
        self.fail = fail

    def list_cases(self):
        if self.fail:
            raise RuntimeError("langfuse is down")
        return sorted(self.cases, key=lambda case: case.received_at, reverse=True)


def _client(store, *, signed_in=True):
    app = FastAPI()
    app.include_router(test_cases.router, prefix=settings.api_prefix)
    if signed_in:
        app.dependency_overrides[get_chat_user] = lambda: {"sub": "anon-test"}
    app.dependency_overrides[get_case_store] = lambda: store
    return TestClient(app)


CASES = [
    _case("c1", "2026-11-01T10:00:00+00:00"),
    _case(
        "c2",
        "2026-11-02T10:00:00+00:00",
        source="chat",
        submitted_by="kanav",
        language="gu",
        description="Milk total for the last 10 days, asked in Gujarati.",
        case_types=["edge_case"],
        tags=["milk_collection", "follow_up"],
        turns=[
            {"role": "user", "content": "મારું દૂધ કેટલું થયું?"},
            {"role": "assistant", "content": "Your milk was 120 litres.", "original_content": "Wrong total.", "trace_id": "9b2f4c1e8a7d4f3b9c0e1d2a3b4c5d6e"},
        ],
        chat={"session_id": "s-1", "removed_turn_count": 0},
    ),
    _case("c3", "2026-11-03T10:00:00+00:00", source="bulk", submitted_by="yash", tags=[], case_types=["new_flow"], bulk={"case_id": "7"}),
]


# --- masking -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, masked",
    [
        ("9876543210", "******3210"),
        ("Call 98765 43210 now", "Call ******3210 now"),
        ("+91 98765-43210", "******3210"),
        ("Aadhaar 1234 5678 9012 please", "Aadhaar ******9012 please"),
        ("123456789012", "******9012"),
        ("૯૮૭૬૫૪૩૨૧૦ પર ફોન કરો", "******3210 પર ફોન કરો"),
        ("mob9876543210", "mob******3210"),
        ("987-654-3210", "******3210"),
    ],
)
def test_phone_and_aadhaar_numbers_keep_only_their_last_4_digits(text, masked):
    assert mask_numbers(text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "Bonus of 125000 rupees",
        "₹1,25,000",
        "On 02-11-2026",
        "In 2024 2025",
        "fat 6.5 snf 8.5",
        "10 20 30 40 50",
        "Farmer code M001, society NA4192",
        "Give 10 ml twice a day for 3 days",
        "૧૨૦ લિટર",
        "",
    ],
)
def test_amounts_dates_and_short_numbers_stay_as_they_are(text):
    assert mask_numbers(text) == text


# --- the list ------------------------------------------------------------------


def test_the_list_is_newest_first_with_empty_fields_left_out():
    response = _client(_FakeStore(CASES)).get("/api/test-cases/")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]] == ["c3", "c2", "c1"]
    assert (body["total"], body["page"], body["limit"]) == (3, 1, 20)
    first = body["items"][2]
    assert first == {
        "id": "c1",
        "received_at": "2026-11-01T10:00:00+00:00",
        "source": "manual",
        "submitted_by": "suraj",
        "persona": "farmer",
        "language": "en",
        "description": BASE["description"],
        "case_types": ["wrong_answer"],
        "tags": ["veterinary_advisory"],
        "turns": BASE["turns"],
    }
    chat_answer = body["items"][1]["turns"][1]
    assert chat_answer["original_content"] == "Wrong total."
    assert chat_answer["trace_id"] == "9b2f4c1e8a7d4f3b9c0e1d2a3b4c5d6e"


def test_numbers_are_masked_in_every_text_shown():
    case = _case(
        "c9",
        "2026-11-03T11:00:00+00:00",
        description="Farmer on 9876543210 asks about the bonus.",
        tags=["general"],
        other_topic="Call back on 98765 43210",
        turns=[
            {"role": "user", "content": "My Aadhaar is 1234 5678 9012"},
            {"role": "assistant", "content": "Okay.", "original_content": "Noted 123456789012."},
        ],
    )

    item = _client(_FakeStore([case])).get("/api/test-cases/").json()["items"][0]

    assert item["description"] == "Farmer on ******3210 asks about the bonus."
    assert item["other_topic"] == "Call back on ******3210"
    assert item["turns"][0]["content"] == "My Aadhaar is ******9012"
    assert item["turns"][1]["original_content"] == "Noted ******9012."


@pytest.mark.parametrize(
    "query, ids",
    [
        ("tag=milk_collection", ["c2"]),
        ("tag=follow_up&tag=veterinary_advisory", ["c2", "c1"]),
        ("tag=safety", []),
        ("case_type=new_flow", ["c3"]),
        ("case_type=edge_case&case_type=wrong_answer", ["c2", "c1"]),
        ("source=chat", ["c2"]),
        ("persona=farmer&language=gu", ["c2"]),
        ("submitted_by=yash", ["c3"]),
        ("q=SWOLLEN", ["c3", "c1"]),
        ("q=દૂધ", ["c2"]),
        ("q=wrong total", ["c2"]),
        ("q=%20%20", ["c3", "c2", "c1"]),
        ("tag=veterinary_advisory&submitted_by=suraj", ["c1"]),
    ],
)
def test_filters_narrow_the_list(query, ids):
    body = _client(_FakeStore(CASES)).get(f"/api/test-cases/?{query}").json()

    assert [item["id"] for item in body["items"]] == ids
    assert body["total"] == len(ids)


def test_pages_split_the_matches_and_total_counts_them_all():
    client = _client(_FakeStore(CASES))

    second = client.get("/api/test-cases/?page=2&limit=2").json()
    past_the_end = client.get("/api/test-cases/?page=3&limit=2").json()

    assert [item["id"] for item in second["items"]] == ["c1"]
    assert (second["total"], second["page"], second["limit"]) == (3, 2, 2)
    assert past_the_end["items"] == [] and past_the_end["total"] == 3


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "page=0", "q=" + "x" * 201, "source=" + "x" * 65])
def test_out_of_range_parameters_are_a_422(query):
    assert _client(_FakeStore(CASES)).get(f"/api/test-cases/?{query}").status_code == 422


def test_cases_sent_under_an_older_contract_still_show():
    cases = [
        _case("old", "2026-10-31T10:00:00+00:00", test_account="f1", turns=[
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "content": "Yes"},
        ]),
        _case("odd", "2026-10-31T09:00:00+00:00", persona=None, tags="milk", case_types=[3, "edge_case"], turns=[
            {"role": "farmer", "content": "skipped"},
            {"role": "user", "content": 42},
            "not a turn",
            {"role": "user", "content": "Kept"},
        ]),
        _case("no-turns", "2026-10-31T08:00:00+00:00", turns="none"),
        _case("only-junk", "2026-10-31T07:00:00+00:00", turns=[{"role": "bot", "content": "x"}]),
    ]

    items = _client(_FakeStore(cases)).get("/api/test-cases/").json()["items"]

    assert [item["id"] for item in items] == ["old", "odd"]
    assert len(items[0]["turns"]) == 3
    odd = items[1]
    assert (odd["persona"], odd["tags"], odd["case_types"]) == ("", [], ["edge_case"])
    assert odd["turns"] == [{"role": "user", "content": "Kept"}]


def test_the_list_and_coverage_need_a_token():
    client = _client(_FakeStore(CASES), signed_in=False)

    assert client.get("/api/test-cases/").status_code == 401
    assert client.get("/api/test-cases/coverage/").status_code == 401


@pytest.mark.parametrize("path", ["/api/test-cases/", "/api/test-cases/coverage/"])
def test_with_no_store_the_list_is_a_503(path):
    response = _client(None).get(path)

    assert response.status_code == 503
    assert response.json()["detail"] == "Test case store is not configured"


def _app_log(caplog):
    # The test client logs each request URL itself; only the app's own lines count.
    return "\n".join(record.getMessage() for record in caplog.records if record.name.startswith("app")).casefold()


@pytest.mark.parametrize("path", ["/api/test-cases/", "/api/test-cases/coverage/"])
def test_a_store_that_cant_be_read_is_a_503_and_the_logs_hold_no_turn_text(path, caplog):
    caplog.set_level(logging.DEBUG)

    response = _client(_FakeStore(CASES, fail=True)).get(path)

    assert response.status_code == 503
    assert response.json()["detail"] == "Test case store is unavailable"
    assert "store_failed error=runtimeerror" in _app_log(caplog)
    assert "swollen" not in _app_log(caplog)


def test_a_listed_page_logs_no_turn_or_search_text(caplog):
    caplog.set_level(logging.DEBUG)

    _client(_FakeStore(CASES)).get("/api/test-cases/?q=swollen")

    assert "list_test_cases stored=3 matched=2" in _app_log(caplog)
    assert "swollen" not in _app_log(caplog)


# --- coverage ------------------------------------------------------------------


def test_coverage_counts_every_tag_and_case_type_with_zeros_and_older_ids():
    cases = [*CASES, _case("c4", "2026-11-03T12:00:00+00:00", tags=["milk_collection", "milk_collection", "old_tag"], case_types=["should_not_happen"])]

    body = _client(_FakeStore(cases)).get("/api/test-cases/coverage/").json()

    assert (body["total"], body["untagged"]) == (4, 1)
    tags = {tag["id"]: tag["count"] for tag in body["tags"]}
    assert [tag["id"] for tag in body["tags"]] == [option.id for option in CASE_OPTIONS.tags] + ["old_tag"]
    assert (tags["milk_collection"], tags["veterinary_advisory"], tags["follow_up"], tags["safety"], tags["old_tag"]) == (2, 1, 1, 0, 1)
    assert body["tags"][-1] == {"id": "old_tag", "label": "old_tag", "count": 1}
    types = {case_type["id"]: case_type["count"] for case_type in body["case_types"]}
    assert (types["wrong_answer"], types["edge_case"], types["new_flow"], types["good_answer"], types["should_not_happen"]) == (1, 1, 1, 0, 1)


def test_coverage_of_an_empty_store_lists_every_tag_at_zero():
    body = _client(_FakeStore([])).get("/api/test-cases/coverage/").json()

    assert (body["total"], body["untagged"]) == (0, 0)
    assert {tag["count"] for tag in body["tags"]} == {0}
    assert len(body["tags"]) == len(CASE_OPTIONS.tags)


# --- the Langfuse store --------------------------------------------------------


def _item(item_id, received_at=None, status="ACTIVE", raw=None, created_at=None):
    return SimpleNamespace(
        id=item_id,
        status=status,
        input={"turns": []} if raw is None else raw,
        metadata={"received_at": received_at} if received_at else None,
        created_at=created_at,
    )


class _NotFound(Exception):
    status_code = 404


class _FakeLangfuse:
    def __init__(self, items=(), error=None):
        self.items = list(items)
        self.error = error
        self.reads = 0

    def get_dataset(self, name, *, fetch_items_page_size):
        self.reads += 1
        if self.error:
            raise self.error
        return SimpleNamespace(items=list(self.items))

    def create_dataset(self, **kwargs):
        pass

    def create_dataset_item(self, **kwargs):
        pass


def test_the_langfuse_store_lists_active_items_newest_first():
    client = _FakeLangfuse([
        _item("a", "2026-11-01T10:00:00+00:00"),
        _item("archived", "2026-11-03T10:00:00+00:00", status="ARCHIVED"),
        _item("b", created_at=datetime(2026, 11, 2, 10, tzinfo=timezone.utc)),
        _item("not-a-body", "2026-11-03T09:00:00+00:00", raw="text"),
    ])

    cases = LangfuseCaseStore(client, "amul-test-case-submissions").list_cases()

    assert [case.id for case in cases] == ["b", "a"]
    assert cases[0].received_at == "2026-11-02T10:00:00+00:00"


def test_with_no_dataset_yet_the_list_is_empty_and_other_errors_raise():
    assert LangfuseCaseStore(_FakeLangfuse(error=_NotFound()), "x").list_cases() == []
    with pytest.raises(RuntimeError):
        LangfuseCaseStore(_FakeLangfuse(error=RuntimeError("down")), "x").list_cases()


def test_the_langfuse_store_reads_once_per_window_and_again_after_a_save(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(test_case_store.time, "monotonic", lambda: now[0])
    client = _FakeLangfuse([_item("a", "2026-11-01T10:00:00+00:00")])
    store = LangfuseCaseStore(client, "amul-test-case-submissions")

    store.list_cases()
    store.list_cases()
    assert client.reads == 1

    store.save(case_id="b", raw={"a": 1}, received_at="t")
    store.list_cases()
    assert client.reads == 2

    now[0] += test_case_store.LIST_CACHE_SECONDS
    store.list_cases()
    assert client.reads == 3


def test_a_save_that_lands_during_a_read_is_not_hidden_by_the_cache(monkeypatch):
    monkeypatch.setattr(test_case_store.time, "monotonic", lambda: 1000.0)

    class _SaveMidRead(_FakeLangfuse):
        def get_dataset(self, name, *, fetch_items_page_size):
            listed = super().get_dataset(name, fetch_items_page_size=fetch_items_page_size)
            if self.reads == 1:
                store.save(case_id="b", raw={"a": 1}, received_at="t")  # another request, mid-read
            return listed

    client = _SaveMidRead([_item("a", "2026-11-01T10:00:00+00:00")])
    store = LangfuseCaseStore(client, "amul-test-case-submissions")

    store.list_cases()
    store.list_cases()

    assert client.reads == 2
