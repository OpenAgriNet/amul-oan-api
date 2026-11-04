"""What the test case review list sends back (#517).

GET /api/test-cases/ lists the cases already sent, newest first, and
GET /api/test-cases/coverage/ counts them per tag, case type and labeller group, so
testers can see what is covered before they add more. Text is shown with phone and
Aadhaar numbers masked (app/services/personal_data.py). Empty optional fields are
left out, so a case the labeller hasn't labelled yet has no group or answer_type.
"""
from typing import Literal, Optional

from pydantic import BaseModel


class TurnView(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # What the bot said before the tester corrected it, in a chat case.
    original_content: Optional[str] = None
    trace_id: Optional[str] = None


class CaseView(BaseModel):
    id: str
    received_at: str
    source: str
    submitted_by: str
    persona: str
    language: str
    description: str
    case_types: list[str]
    tags: list[str]
    other_topic: Optional[str] = None
    turns: list[TurnView]
    # The labeller's label (#488): one of its groups, and what the bot should do.
    group: Optional[str] = None
    answer_type: Optional[str] = None


class CaseList(BaseModel):
    items: list[CaseView]
    # Cases that match the filters, across all pages.
    total: int
    page: int
    limit: int


class CountedOption(BaseModel):
    id: str
    label: str
    count: int


class CaseCoverage(BaseModel):
    total: int
    # Cases sent with no tag at all.
    untagged: int
    # Every tag and case type on the screen's lists, in their order, zeros included;
    # then any other id found in the cases (an older list), with the id as its label.
    tags: list[CountedOption]
    case_types: list[CountedOption]
    # Every labeller group in its order, zeros included, then any other group found.
    groups: list[CountedOption]
    # Cases the labeller hasn't labelled yet.
    unlabelled: int
