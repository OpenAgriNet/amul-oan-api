"""Contract for the dev-only test case intake (POST /api/test-cases/), v1.

One test case per request from the amul-ui test case screen: a chat, a case typed
in the Write tab, or one case of a bulk upload. The turn rules are the screen's
own (``validateDraft`` in src/lib/test-cases/draft.ts), with the cap at 20 farmer
questions. A field outside this contract is a 422, so a typo can't sit in the raw
store unread: a new field goes into the backend first.

Case types and tags are hints, not labels. Any id in the right format passes,
since the lists are still being reviewed (#486).
"""
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo, field_validator, model_validator

from app.personas import ChatPersona

MIN_DESCRIPTION_LENGTH = 20
# Farmer questions per case. The screen's MAX_EXCHANGES must be the same number.
MAX_USER_TURNS = 20

# The screen checks in JavaScript: trim() strips this set, and .length counts UTF-16
# code units (an emoji is 2). Python's strip() and len() differ on both, which
# would reject cases the screen accepted.
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _js_trim(value: str) -> str:
    return value.strip(_JS_WHITESPACE)


def _js_length(value: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in value)


_HintId = Annotated[str, Field(pattern=r"^[a-z0-9_]{1,64}$")]
# A tester or a test farmer, as a short alias. It starts with a letter, so it
# can't be a phone number.
_Alias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
_Language = Annotated[str, Field(pattern=r"^[a-z]{2}$")]
# The chat response's X-Trace-Id: uuid4().hex (app/routers/chat.py).
_TraceId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


class _Strict(BaseModel):
    # strict: no coercion, so "3" or 3.0 is not an int and the raw store only
    # ever holds the types below.
    model_config = ConfigDict(extra="forbid", strict=True)


class CaseTurn(_Strict):
    """One message, as the tester kept it."""

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not _js_trim(value):
            raise ValueError("A turn is empty")
        return value


class ChatTurn(CaseTurn):
    """A turn imported from a chat. original_content is the text before the
    tester's edit, sent only when there was one, for either role. trace_id names a
    bot answer's turn.v1 trace."""

    original_content: Optional[str] = None
    trace_id: Optional[_TraceId] = None

    @field_validator("original_content")
    @classmethod
    def _original_not_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _js_trim(value):
            raise ValueError("original_content is empty")
        return value

    @model_validator(mode="after")
    def _trace_on_answers(self) -> "ChatTurn":
        if self.trace_id is not None and self.role != "assistant":
            raise ValueError("trace_id goes on bot answers only")
        return self


class ChatInfo(_Strict):
    session_id: Annotated[str, Field(min_length=1, max_length=128)]
    # Turns the tester struck out. The bot's history still held them.
    removed_turn_count: Annotated[int, Field(ge=0, le=1000)]


class BulkInfo(_Strict):
    # The sheet's case_id. With submitted_by it finds the case in the tester's sheet.
    case_id: Annotated[str, Field(min_length=1, max_length=64)]


class _Case(_Strict):
    schema_version: Literal[1]
    submitted_by: _Alias
    persona: ChatPersona
    language: _Language
    description: str
    # fail_fast: a list reports its first bad element only, so a body full of bad
    # elements can't turn into a huge 422 built on the event loop.
    case_types: list[_HintId] = Field(fail_fast=True)
    tags: list[_HintId] = Field(fail_fast=True)
    other_topic: Optional[str] = None
    turns: list[CaseTurn] = Field(fail_fast=True)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _exactly_one(cls, value):
        # Literal[1] matches by ==, so true and 1.0 would pass even in strict mode.
        if type(value) is not int:
            raise ValueError("Input should be 1")
        return value

    @field_validator("description")
    @classmethod
    def _describes_the_case(cls, value: str) -> str:
        if _js_length(_js_trim(value)) < MIN_DESCRIPTION_LENGTH:
            raise ValueError(f"The description needs at least {MIN_DESCRIPTION_LENGTH} characters")
        return value

    @field_validator("other_topic")
    @classmethod
    def _topic(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        if value is None:
            return value
        if not _js_trim(value):
            raise ValueError("other_topic is empty")
        # tags is checked first; if it failed, its own error says so.
        if "tags" in info.data and "general" not in info.data["tags"]:
            raise ValueError("other_topic goes with the general tag only")
        return value

    @field_validator("turns")
    @classmethod
    def _conversation(cls, turns: list) -> list:
        questions = sum(1 for turn in turns if turn.role == "user")
        if questions == 0:
            raise ValueError("Add at least one farmer question")
        if not any(turn.role == "assistant" for turn in turns):
            raise ValueError("Add at least one bot answer")
        if turns[0].role != "user":
            raise ValueError("The conversation has to start with a farmer question")
        if questions > MAX_USER_TURNS:
            raise ValueError(f"Keep it to {MAX_USER_TURNS} questions or fewer")
        return turns


class ChatCase(_Case):
    source: Literal["chat"]
    turns: list[ChatTurn] = Field(fail_fast=True)
    chat: ChatInfo


class ManualCase(_Case):
    source: Literal["manual"]
    # For a case whose answer needs a farmer's data. Chat cases have it in their traces.
    test_account: Optional[_Alias] = None


class BulkCase(_Case):
    source: Literal["bulk"]
    bulk: BulkInfo
    test_account: Optional[_Alias] = None


CASE_SOURCES = ("chat", "manual", "bulk")

CaseSubmission = TypeAdapter(
    Annotated[Union[ChatCase, ManualCase, BulkCase], Field(discriminator="source")]
)


class SubmitTestCaseResponse(BaseModel):
    """What the screen gets back (``SubmitTestCaseResult`` in amul-ui)."""

    id: str
    status: Literal["received"] = "received"
