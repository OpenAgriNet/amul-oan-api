"""Test cases from the amul-ui test case screen, kept in a Langfuse dataset (#487, #517, #488).

The dataset is the data model. Each case is one item: ``input`` is the body exactly
as the screen sent it, ``metadata.received_at`` is when it came in (Langfuse moves
``createdAt`` on every write), and ``metadata.label`` is its label once the labeller has
run. Archiving an item in Langfuse takes it off the list.

The labeller gives each new case a category and an answer type with Gemma 4 through
LiteLLM, reached the way oan-evaluation's judge reaches it: SCENARIO_GEMMA_BASE_URL,
LITELLM_API_KEY and SCENARIO_GEMMA_MODEL. It is off while the first two are unset.
Testers, case types, tags and answer types are in assets/test_case_options.json. The tags
are the bot's features: a tester picks any as hints, and the category is the one feature
the labeller files the case under.
"""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import settings
from app.observability import get_langfuse_client
from helpers.utils import get_logger

logger = get_logger(__name__)

_OPTIONS_FILE = Path(__file__).resolve().parents[2] / "assets" / "test_case_options.json"
OPTIONS = json.loads(_OPTIONS_FILE.read_text(encoding="utf-8"))
# A case can be filed under any tag the options file doesn't mark "category": false.
CATEGORIES = [tag for tag in OPTIONS["tags"] if tag.get("category", True)]

# Same namespace as before, so a case sent again keeps its id.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "amul-oan-api/test-cases")
# Phone and Aadhaar numbers: a run of 7+ digits, or 2-4 groups of 10-13 digits in all.
# Dates, amounts and cycles like 2025-26 stay.
_LONG_NUMBER = re.compile(r"(?<!\d)\+?(?:\d{7,}|(?=(?:\d[ -]?){9,12}\d(?!\d))\d{2,6}(?:[ -]\d{2,6}){1,3})(?!\d)")
# Trace and session ids: 32 hex characters, or a UUID. Masking leaves them as they are.
_ID = re.compile(r"[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_dataset_ready = False

_PROMPT = """You label one test case of Amul's chatbot for dairy farmers in Gujarat. AI here means artificial insemination.
Reply with one JSON object and nothing else:
{{"category": "...", "answer_type": "..."}}

category: the one feature the case tests. It is one of these ids and nothing else:
{categories}
A long chat can touch several features. Its category is the feature the case was written for, not the one its last message is about. The tester's tags name the features the tester thinks it covers: if any of them fits the chat, the category is one of those. With several that fit, it is the one most of the farmer's messages are about, or the first of them in the tester's list when the chat is spread evenly. Take a feature the tester didn't tag only when none of the tags fits or there are none; it is then the feature most of the farmer's messages are about. A tester's tag that isn't in the list above is not a category.

answer_type: what the bot should do after the farmer's last message. It is one of these and nothing else:
{answer_types}
- ask:<what it should ask for>: it asks the farmer for something first, like ask:species, or ask:clarify when it can't tell what the farmer means.

The farmer's last message is the last one that asks or tells the bot something: a thank you or ok at the end doesn't count.
The only thing the bot can book is an AI call. For that it needs the animal (cow or buffalo) and the technician: it asks for whichever the farmer hasn't given, ask:species first, then ask:technician, and books once it has both. In an AI booking chat, a reply like "the first one" or "the second one" picks one of the technicians the bot listed. For an AI call it never asks for a date, a time or an animal number. If the case shows booking isn't open to the farmer (the union doesn't allow it, there's no technician, or no farmer profile), it's say_unavailable.
The bot can't book a vet or doctor visit for an animal, or a health call: that is not an AI booking. The answer_type of a case that asks for one is vet, even if it asks about other things too.
If the hints say persona: doctor, the user is a vet, not a farmer. That bot only answers from its documents: it books nothing and doesn't tell them to see a vet.

Phone and Aadhaar numbers in the text are cut to their last 4 digits, like ******1234.
The hints come from the tester and may be wrong.

{case}"""


def store() -> Optional[tuple[Any, str]]:
    """The Langfuse client and dataset name, or None while either is missing."""
    client, name = get_langfuse_client(), (settings.test_cases_raw_dataset or "").strip()
    if client is None or not name:
        logger.warning("test_cases no_store: set TEST_CASES_RAW_DATASET and the LANGFUSE_* keys")
        return None
    return client, name


def case_id(body: dict) -> str:
    """The same body always gets the same id. Raises ValueError for NaN or a lone surrogate."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_ID_NAMESPACE, f"{(settings.test_cases_raw_dataset or '').strip()}:{digest}"))


async def get_item(client: Any, item_id: str) -> Any:
    """The item as stored now, or None when there is none."""
    try:
        return await client.async_api.dataset_items.get(item_id)
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            return None
        raise


async def save_case(client: Any, name: str, item_id: str, **fields: Any) -> None:
    """Write one item. A new id makes a new item; on an existing one, Langfuse keeps every
    field that isn't sent (input, metadata, status, expected output)."""
    global _dataset_ready
    try:
        if not _dataset_ready:
            # Creating a dataset upserts on its name.
            await client.async_api.datasets.create(name=name, description="Test cases from the dev test case screen.")
            _dataset_ready = True
        await client.async_api.dataset_items.create(dataset_name=name, id=item_id, **fields)
    except Exception:
        _dataset_ready = False  # the dataset may have been deleted in Langfuse
        raise


def received_at(item: Any) -> str:
    """When the case came in, or createdAt for an item made by hand in Langfuse."""
    value = item.metadata.get("received_at") if isinstance(item.metadata, dict) else None
    return value if isinstance(value, str) else (item.created_at.isoformat() if item.created_at else "")


async def list_items(client: Any, name: str) -> list:
    """Every item that isn't archived, newest first."""
    # One snapshot: Langfuse pages by version time, so a write between two pages would
    # show a case twice and leave another out.
    version, page, items = datetime.now(timezone.utc), 1, []
    try:
        while True:
            batch = await client.async_api.dataset_items.list(dataset_name=name, version=version, page=page, limit=100)
            items += batch.data
            if page >= batch.meta.total_pages:
                break
            page += 1
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:  # nothing was sent yet
            return []
        raise
    items = [item for item in items if item.status != "ARCHIVED" and isinstance(item.input, dict)]
    return sorted(items, key=received_at, reverse=True)


def mask(value: Any) -> Any:
    """A copy of a JSON value with phone and Aadhaar numbers cut to their last 4 digits.
    It walks the value in a loop, not by recursion, so no case is nested too deep for it."""
    top = [value]
    todo = [(top, 0)]
    while todo:
        parent, key = todo.pop()
        node = parent[key]
        if isinstance(node, str):
            if not _ID.fullmatch(node):
                parent[key] = _LONG_NUMBER.sub(lambda m: "******" + re.sub(r"\D", "", m.group())[-4:], node)
        elif isinstance(node, (list, dict)):
            parent[key] = node = node.copy()
            todo.extend((node, k) for k in (range(len(node)) if isinstance(node, list) else node))
    return top[0]


def labeller_on() -> bool:
    return bool(os.environ.get("SCENARIO_GEMMA_BASE_URL") and os.environ.get("LITELLM_API_KEY"))


def _text(value: Any) -> str:
    return mask(value)[:4000] if isinstance(value, str) else ""


def build_prompt(body: dict) -> str:
    hints = []
    for key in ("tags", "case_types", "persona", "language", "other_topic"):
        value = body.get(key)
        value = ", ".join(map(_text, value[:20])) if isinstance(value, list) else _text(value)
        if value:
            hints.append(f"{key}: {value}")
    lines = [f"Hints: {'; '.join(hints)}"] if hints else []
    if _text(body.get("description")):
        lines.append(f"What the tester says the case checks: {_text(body.get('description'))}")
    lines.append("Conversation:")
    for turn in (body.get("turns") if isinstance(body.get("turns"), list) else [])[-40:]:
        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
            continue
        line = ("Farmer: " if turn["role"] == "user" else "Bot: ") + _text(turn.get("content"))
        if _text(turn.get("original_content")):
            first = "the farmer first wrote" if turn["role"] == "user" else "the bot first said"
            line += f" [the tester corrected this; {first}: {_text(turn.get('original_content'))}]"
        lines.append(line)
    listed = {
        key: "\n".join(f"- {o['id']}: {o['hint']}" for o in options)
        for key, options in (("categories", CATEGORIES), ("answer_types", OPTIONS["answer_types"]))
    }
    return _PROMPT.format(**listed, case="\n".join(lines))


def parse_label(reply: Any) -> Optional[dict]:
    """The model's reply as {category, answer_type}, or None when it isn't a usable label."""
    if not isinstance(reply, str):
        return None
    try:
        data = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    category, answer_type = (
        str(data.get(key, "")).strip().lower().replace("-", "_") for key in ("category", "answer_type")
    )
    known_types = {o["id"] for o in OPTIONS["answer_types"]}
    if category in {o["id"] for o in CATEGORIES} and (
        answer_type in known_types or re.fullmatch(r"ask:[a-z_]{1,30}", answer_type)
    ):
        return {"category": category, "answer_type": answer_type}
    return None


def _model() -> str:
    return (os.environ.get("SCENARIO_GEMMA_MODEL") or "gemma-4-31b-it").removeprefix("openai/")


async def _complete(prompt: str) -> Any:
    url = os.environ["SCENARIO_GEMMA_BASE_URL"].rstrip("/") + "/chat/completions"
    body = {"model": _model(), "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"}
    async with httpx.AsyncClient(timeout=120) as http:
        reply = await http.post(url, json={**body, "response_format": {"type": "json_object"}}, headers=headers)
        if reply.status_code == 400:  # the proxy may refuse JSON mode
            reply = await http.post(url, json=body, headers=headers)
        reply.raise_for_status()
        return reply.json()["choices"][0]["message"]["content"]


async def label_case(client: Any, name: str, item_id: str, body: dict, metadata: dict) -> None:
    """Label one new case and keep the label with it. Never raises: labelling is extra."""
    try:
        label = parse_label(await _complete(build_prompt(body)))
        if label is None:
            logger.warning("test_case_label failed id=%s reason=bad_reply", item_id)
            return
        label.update(model=_model(), labelled_at=datetime.now(timezone.utc).isoformat())
        # Only metadata is sent, so Langfuse keeps the input, status and expected output.
        await save_case(client, name, item_id, metadata={**metadata, "label": label})
        logger.info(
            "test_case_label saved id=%s category=%s answer_type=%s", item_id, label["category"], label["answer_type"]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("test_case_label failed id=%s error=%s", item_id, type(exc).__name__)
