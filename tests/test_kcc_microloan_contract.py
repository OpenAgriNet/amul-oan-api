"""Regression contract: keep KCC and KDCC micro-loan as separate use-cases.

These tests intentionally validate prompt/tool guidance text only. They do not
enforce runtime pre-routing, and they do not require ENABLE_NETWORK changes.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_agrinet_prompts_route_kcc_to_vistaar():
    for prompt_path in (
        "assets/prompts/agrinet_system.md",
        "assets/prompts/agrinet_system_translation_pipeline.md",
    ):
        prompt = _read(prompt_path)
        assert 'scheme_code="kcc"' in prompt
        assert "Do **not** call `check_loan_eligibility` for KCC." in prompt


def test_agrinet_prompts_limit_micro_loan_to_kdcc_facility():
    for prompt_path in (
        "assets/prompts/agrinet_system.md",
        "assets/prompts/agrinet_system_translation_pipeline.md",
    ):
        prompt = _read(prompt_path)
        assert "KDCC/Kheda cooperative micro-loan" in prompt
        assert "not KCC" in prompt


def test_moderation_prompt_marks_kcc_and_kdcc_as_distinct_intents():
    moderation = _read("assets/prompts/moderation_system.md")
    assert "distinct intents" in moderation
    assert "KCC is a government scheme query" in moderation
    assert "KDCC/cooperative micro-loan is the cooperative micro-loan facility" in moderation


def test_loan_tool_contract_excludes_kcc_queries():
    loan_tool = _read("agents/tools/loan.py")
    assert "ONLY for the KDCC/Kheda cooperative micro-loan facility" in loan_tool
    assert "Do NOT use this tool for KCC" in loan_tool
    assert 'get_vistaar_scheme_info("kcc")' in loan_tool


def test_vistaar_tool_contract_includes_kcc_routing_guard():
    vistaar_tool = _read("agents/tools/vistaar.py")
    assert 'scheme_code="kcc"' in vistaar_tool
    assert "Do NOT route KCC requests to check_loan_eligibility" in vistaar_tool


def test_agrinet_prompts_route_local_language_kcc_queries_to_vistaar():
    for prompt_path in (
        "assets/prompts/agrinet_system.md",
        "assets/prompts/agrinet_system_translation_pipeline.md",
    ):
        prompt = _read(prompt_path)
        assert "(or local-language equivalent)" in prompt
        assert "call `get_vistaar_scheme_info` with `scheme_code=\"kcc\"`" in prompt
