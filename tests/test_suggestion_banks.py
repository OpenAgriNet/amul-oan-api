"""Schema checks for curated suggestion question banks (step 1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BANKS_PATH = Path(__file__).resolve().parents[1] / "assets" / "suggestion_banks.json"
POLICY_PATH = Path(__file__).resolve().parents[1] / "assets" / "gu_term_policy.json"

EXPECTED_TOOLS = {
    "animal_health": {"create_health_call"},
    "milk_quantity": {"get_farmer_milk_collection_details"},
    "ai_call": {"create_ai_call"},
    "vistaar": {"get_vistaar_weather", "get_vistaar_mandi_prices"},
    "loan_eligibility": {"check_loan_eligibility"},
}


@pytest.fixture(scope="module")
def banks() -> dict:
    return json.loads(BANKS_PATH.read_text(encoding="utf-8"))


def test_suggestion_banks_file_exists():
    assert BANKS_PATH.is_file()


def test_suggestion_banks_domains_and_tools(banks: dict):
    assert banks.get("version") == 1
    domains = banks["domains"]
    assert set(domains) == set(EXPECTED_TOOLS)
    assert not (set(domains) & {"union_schemes", "vistaar_schemes", "schemes"})
    for domain, expected_tools in EXPECTED_TOOLS.items():
        assert set(domains[domain]["opens_on_tools"]) == expected_tools


def test_suggestion_banks_question_counts_and_trilingual(banks: dict):
    seen_ids: set[str] = set()
    for domain, meta in banks["domains"].items():
        questions = meta["questions"]
        if domain == "vistaar":
            by_tag: dict[str, int] = {"weather": 0, "mandi": 0}
            for q in questions:
                tag = q["tag"]
                assert tag in by_tag
                by_tag[tag] += 1
                _assert_question(q, seen_ids)
            for tag, count in by_tag.items():
                assert 3 <= count <= 4, f"vistaar/{tag} expected 3-4, got {count}"
        else:
            assert 3 <= len(questions) <= 4, f"{domain} expected 3-4, got {len(questions)}"
            for q in questions:
                assert "tag" not in q
                _assert_question(q, seen_ids)


def test_suggestion_banks_gujarati_avoids_forbidden_terms(banks: dict):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    forbidden = [term for term in policy.get("forbidden", {}) if term]
    for domain, meta in banks["domains"].items():
        for q in meta["questions"]:
            text = q["gu"]
            for term in forbidden:
                assert term not in text, f"{domain}/{q['id']} contains forbidden '{term}'"


def _assert_question(q: dict, seen_ids: set[str]) -> None:
    qid = q["id"]
    assert qid and qid not in seen_ids
    seen_ids.add(qid)
    assert q["en"].strip()
    assert q["gu"].strip()
    assert q["hi"].strip()
    assert q["en"].endswith(("?", ".")) or "؟" in q["en"]
