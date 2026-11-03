"""Label a test case with Gemma 4 through LiteLLM (#488).

One label per case: the feature it tests (one id from the screen's tag list), a few
short sub-tags, the use case and the edge case in a few words, and what the bot
should do, in eval set v0's ``expected_action`` terms. Text is masked
(app/services/personal_data.py) before it goes into the prompt. A reply that isn't
JSON, or names a feature or answer type outside the lists, is an error, not a label.

The model is reached the way the judge in oan-evaluation reaches it
(evaluation/amul/judge.py): ``SCENARIO_GEMMA_BASE_URL``, ``LITELLM_API_KEY`` and
``SCENARIO_GEMMA_MODEL`` (default ``gemma-4-31b-it``). Nothing here writes anywhere,
and it calls the model with a plain HTTP request, so nothing is traced to Langfuse.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.models.case_options import CASE_OPTIONS
from app.services.personal_data import mask_personal_data

FEATURES = {option.id: option for option in CASE_OPTIONS.tags}
# Eval set v0's expected_action values (oan-evaluation, inference/amul_eval_set_v0_metadata.json).
ANSWER_TYPES = ("answer", "book", "refer_vet", "urgent_vet", "say_unavailable", "out_of_scope")
_ASK = re.compile(r"^ask:[a-z_]{1,30}$")
DEFAULT_MODEL = "gemma-4-31b-it"


class LabelError(Exception):
    """The model's reply is not a usable label."""


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "assistant"
    text: str


@dataclass(frozen=True)
class Case:
    set_name: str
    case_id: str
    turns: tuple[Turn, ...]
    description: str = ""
    # What the set already says about the case (its category, tools, expected action).
    hints: dict[str, str] = field(default_factory=dict)


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feature: str
    sub_tags: list[str]
    use_case: str
    edge_case: str
    answer_type: str

    @field_validator("feature")
    @classmethod
    def _known_feature(cls, value: str) -> str:
        if value not in FEATURES:
            raise ValueError(f"{value!r} is not on the feature list")
        return value

    @field_validator("sub_tags")
    @classmethod
    def _short_tags(cls, value: list[str]) -> list[str]:
        if len(value) > 5 or any(not tag.strip() or len(tag) > 40 for tag in value):
            raise ValueError("up to 5 sub-tags of 1 to 40 characters")
        return value

    @field_validator("use_case")
    @classmethod
    def _use_case(cls, value: str) -> str:
        if not value.strip() or len(value) > 80:
            raise ValueError("the use case takes 1 to 80 characters")
        return value

    @field_validator("edge_case")
    @classmethod
    def _edge_case(cls, value: str) -> str:
        if len(value) > 80:
            raise ValueError("the edge case takes up to 80 characters")
        return value

    @field_validator("answer_type")
    @classmethod
    def _answer_type(cls, value: str) -> str:
        if value not in ANSWER_TYPES and not _ASK.match(value):
            raise ValueError(f"{value!r} is not an answer type")
        return value


_INSTRUCTIONS = """You label one test case of Amul's chatbot for dairy farmers in Gujarat.
Reply with one JSON object and nothing else:
{{"feature": "...", "sub_tags": ["..."], "use_case": "...", "edge_case": "...", "answer_type": "..."}}

feature: the one feature the case tests, as an id from this list:
{features}

sub_tags: up to 5 short tags a person can read, like "swollen leg" or "AI booking negative case".
use_case: what the farmer wants, in a few words.
edge_case: what makes the case unusual, like "out of scope", "Gujarati typed in English letters", "a tool error" or "missing data", or "" when nothing does.
answer_type: what the bot should do: {answer_types}, or ask:<what it should ask for>, like ask:species.

Personal data in the text is masked: ******1234 is a phone or Aadhaar number, [code] is an OTP or a farmer code.
The hints come from the set the case is in and may be wrong or out of date.
"""


def build_prompt(case: Case) -> str:
    features = "\n".join(f"- {option.id}: {option.label}. {option.hint}" for option in FEATURES.values())
    parts = [_INSTRUCTIONS.format(features=features, answer_types=", ".join(ANSWER_TYPES))]
    if case.hints:
        parts.append("Hints: " + "; ".join(f"{key}: {value}" for key, value in case.hints.items() if value))
    if case.description:
        parts.append("What the tester says the case checks: " + mask_personal_data(case.description))
    conversation = "\n".join(
        f"{'Farmer' if turn.role == 'user' else 'Bot'}: {mask_personal_data(turn.text)}" for turn in case.turns
    )
    parts.append("Conversation:\n" + conversation)
    return "\n\n".join(parts)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_label(content: str) -> Label:
    try:
        data = json.loads(_FENCE.sub("", content.strip()))
    except (ValueError, TypeError) as exc:
        raise LabelError("the reply is not JSON") from exc
    try:
        return Label.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(map(str, first["loc"])) or "reply"
        raise LabelError(f"{where}: {first['msg'].removeprefix('Value error, ')}") from exc


class LiteLLM:
    """Chat completions on the LiteLLM proxy, as oan-evaluation's judge calls it."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        base = os.environ.get("SCENARIO_GEMMA_BASE_URL")
        key = os.environ.get("LITELLM_API_KEY")
        if not (base and key):
            raise LabelError("SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY are required")
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = (os.environ.get("SCENARIO_GEMMA_MODEL") or DEFAULT_MODEL).removeprefix("openai/")
        self._key = key
        self._client = client or httpx.Client(timeout=120)

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {self._key}"}
        response = self._client.post(self.url, json={**body, "response_format": {"type": "json_object"}}, headers=headers)
        if response.status_code == 400:
            # The proxy may refuse JSON mode; the judge then asks again without it.
            response = self._client.post(self.url, json=body, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise LabelError("the reply has no text")
        return content


def label_case(case: Case, complete: Callable[[str], str]) -> Label:
    return parse_label(complete(build_prompt(case)))
