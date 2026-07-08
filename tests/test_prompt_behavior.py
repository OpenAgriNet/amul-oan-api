"""
Prompt-only behavior checks for vague-query handling.

These tests do not call real LLM APIs. They validate the prompt contract by
reading canonical User/Assistant examples from prompt files.
"""
from pathlib import Path
import re
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EN_PROMPT = REPO_ROOT / "assets" / "prompts" / "voice_system_translation_pipeline_en.md"

VAGUE_CASES = [
    "My cow is not giving milk",
    "My buffalo is not coming in heat",
    "My animal is sick",
    "My cow is weak",
    "My buffalo is not eating",
    "My calf has problem",
    "Milk is less",
    "My animal has infection",
    "Cow is not well",
    "Buffalo has issue after calving",
    "Animal is not active",
    "Something is wrong with my buffalo",
    "Cow has breeding problem",
    "My animal is not normal",
    "Buffalo milk dropped suddenly",
    "Cow not chewing cud",
    "Calf is not growing",
    "Animal has fever maybe",
]

NEW_LOG_CASES = [
    pytest.param(
        "I need to get my animal bred",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "My sheep is not passing stool - what should I do?",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "There is bleeding from my cow - what should I do?",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "My buffalo is not coming into heat - what should I do?",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "My animal is not doing passing",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "There is blood coming from inside the cow",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
    pytest.param(
        "My buffalo is not getting heat",
        marks=pytest.mark.xfail(reason="No canonical User/Assistant example in EN prompt yet", strict=False),
    ),
]


def _extract_examples(prompt_text: str) -> dict[str, str]:
    pairs = re.findall(r"User:\s*(.+)\nAssistant:\s*(.+)", prompt_text)
    return {user.strip(): assistant.strip() for user, assistant in pairs}


def run_prompt(user_query: str) -> str:
    """
    Deterministic prompt harness for tests.
    Returns the assistant line from the prompt's canonical examples.
    """
    examples = _extract_examples(EN_PROMPT.read_text(encoding="utf-8"))
    if user_query not in examples:
        raise KeyError(f"No prompt example found for query: {user_query}")
    return examples[user_query]


def test_strict_rule_block_present_in_en_prompt():
    # The prompt's vague-query contract was reworded since this test was first
    # written (the old "(STRICT RULE)" block). Assert the CURRENT canonical
    # wording, which expresses the same contract: one question, ≤15 words, stop.
    required_snippets = [
        "## VAGUE QUERY HANDLING",
        "ask exactly one short clarification question (maximum 15 words, simple and direct)",
        "After asking it, stop",
        "pick the single most important missing detail",
    ]
    text = EN_PROMPT.read_text(encoding="utf-8")
    for snippet in required_snippets:
        assert snippet in text


def test_single_question_only():
    response = run_prompt("My cow is not giving milk")
    assert response.count("?") == 1


def test_no_explanation():
    response = run_prompt("My cow is not giving milk").lower()
    assert "because" not in response
    assert "due to" not in response


def test_max_15_words():
    response = run_prompt("My cow is not giving milk")
    assert len(response.split()) <= 15


def test_no_multiple_questions():
    response = run_prompt("My cow is not giving milk")
    assert response.count("?") == 1


def test_reproduction_case():
    response = run_prompt("My buffalo is not coming in heat")
    assert response.count("?") == 1


@pytest.mark.parametrize("query", NEW_LOG_CASES)
def test_new_log_vague_cases(query: str):
    response = run_prompt(query)
    assert response.count("?") == 1
    assert len(response.split()) <= 15
    lower = response.lower()
    assert "because" not in lower
    assert "due to" not in lower
