"""Label a test case with Gemma 4 through LiteLLM (#488).

Two labels per case, both broad so it's obvious where a case goes: its group (ai,
vet-health, milk, scheme, bonus or other), and what the bot should do, in eval set
v0's ``expected_action`` terms with its refer_vet and urgent_vet as one, vet.
Subgroups wait until a group has about 15 questions. Text is masked
(app/services/personal_data.py) before it goes into the prompt. A reply that isn't
JSON, or names a group or answer type outside the lists, is an error, not a label.

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

from app.services.personal_data import mask_personal_data

# other holds the rest until it has enough questions to be split.
GROUPS = {
    "ai": "anything about artificial insemination (AI): booking an AI call, the technicians, semen, when to inseminate, past AI records",
    "vet-health": (
        "animal health, disease and treatment, heat and pregnancy, feed and fodder, how to raise milk yield or fat, "
        "the farmer's animals and their records, vet visits and the nearest vet office"
    ),
    "milk": (
        "milk pouring records (what was poured, its fat and SNF, and how that changed), the milk price, "
        "payments and deductions, and the SMS after pouring"
    ),
    "bonus": "bonus and bhav fer (the price difference)",
    "scheme": "union and government schemes, subsidies, insurance, loans and PM-KISAN, and their helplines",
    "other": (
        "anything else, like weather, mandi prices, the soil health card, cattle trade (animal rates, buying and selling), "
        "greetings or off-topic questions"
    ),
}
# Eval set v0's expected_action values (oan-evaluation, inference/amul_eval_set_v0_metadata.json),
# with refer_vet and urgent_vet as one, vet: how urgent doesn't matter for curation.
ANSWER_TYPES = {
    "answer": "it answers",
    "book": "it books an AI call",
    "vet": (
        "the main advice is to get a vet to see the animal, as for fever, bloat, swelling, a retained placenta "
        "or repeat breeding, with or without the bot's own advice. Care, feed and dose questions are answer, "
        'and so is "see a vet if it gets worse"'
    ),
    "say_unavailable": "it says it can't do this here, or has no data for it",
    "out_of_scope": "it declines a question it shouldn't answer, like an off-topic one or one about another farmer's data",
}
V0_VET_ACTIONS = ("refer_vet", "urgent_vet")
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
    # What the set already says about the case (its category, tools, the tester's tags),
    # never its expected action.
    hints: dict[str, str] = field(default_factory=dict)


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    group: str
    answer_type: str

    # Small slips in an otherwise good reply are not errors: "Answer", " AI ", "vet_health"
    # or "out-of-scope". A group or answer type off the lists still is.
    @field_validator("group", "answer_type", mode="before")
    @classmethod
    def _tidy_id(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("group")
    @classmethod
    def _known_group(cls, value: str) -> str:
        group = value.replace("_", "-")
        if group not in GROUPS:
            raise ValueError(f"{value!r} is not a group")
        return group

    @field_validator("answer_type")
    @classmethod
    def _answer_type(cls, value: str) -> str:
        answer_type = value.replace("-", "_")
        if answer_type not in ANSWER_TYPES and not _ASK.match(answer_type):
            raise ValueError(f"{value!r} is not an answer type")
        return answer_type


_INSTRUCTIONS = """You label one test case of Amul's chatbot for dairy farmers in Gujarat.
Reply with one JSON object and nothing else:
{{"group": "...", "answer_type": "..."}}

group: the one group the case is about:
{groups}
A chat that moves from one group to another goes in the group of the farmer's last message.

answer_type: what the bot should do after the farmer's last message:
{answer_types}
- ask:<what it should ask for>: it asks the farmer for something first, like ask:species.

The only thing the bot can book is an AI call. For that it needs the animal (cow or buffalo) and the technician: it asks for whichever the farmer hasn't given, ask:species first, then ask:technician, and books once it has both. For an AI call it never asks for a date, a time or an animal number. If the case shows booking isn't open to the farmer (the union doesn't allow it, there's no technician, or no farmer profile), it's say_unavailable.
The bot can't book a vet or doctor visit for an animal, or a health call: a request for one is vet-health, and the answer_type is vet.
If the hints say persona: doctor, the user is a vet, not a farmer. That bot only answers from its documents: it books nothing and doesn't tell them to see a vet.

Personal data in the text is masked: ******1234 is a phone or Aadhaar number, [code] is an OTP or a farmer code.
The hints come from the set the case is in and may be wrong or out of date.
"""


def _options(options: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {meaning}" for key, meaning in options.items())


def build_prompt(case: Case) -> str:
    parts = [_INSTRUCTIONS.format(groups=_options(GROUPS), answer_types=_options(ANSWER_TYPES))]
    if case.hints:
        parts.append("Hints: " + "; ".join(f"{key}: {value}" for key, value in case.hints.items() if value))
    if case.description:
        parts.append("What the tester says the case checks: " + mask_personal_data(case.description))
    conversation = "\n".join(
        f"{'Farmer' if turn.role == 'user' else 'Bot'}: {mask_personal_data(turn.text)}" for turn in case.turns
    )
    parts.append("Conversation:\n" + conversation)
    return "\n\n".join(parts)


def _json_object(content: str):
    """The reply as JSON. A model without JSON mode may wrap the object in a code
    fence or a sentence, so the text from the first { to the last } is tried too."""
    try:
        return json.loads(content)
    except ValueError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end < start:
            raise
        return json.loads(content[start : end + 1])


def parse_label(content: str) -> Label:
    try:
        data = _json_object(content.strip())
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
