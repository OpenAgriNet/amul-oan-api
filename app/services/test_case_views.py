"""Stored test cases as the review list shows them (#517).

The raw store holds every body as it was sent, and the contract has changed since
the first ones (test_account, the pair rule), so a stored case is read leniently:
a field of the wrong type is left empty, a turn that isn't a question or an answer
with text is skipped, and a case with no turns left is skipped. Text is masked on
the way out. The labeller's group and answer type come from the stored label.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from app.models.case_options import CASE_OPTIONS, HintOption
from app.models.test_case_views import CaseCoverage, CaseView, CountedOption, TurnView
from app.services.personal_data import mask_numbers
from app.services.test_case_labeller import GROUPS
from app.services.test_case_store import StoredCase


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _ids(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _turn(turn: Any) -> Optional[TurnView]:
    if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
        return None
    content, original, trace_id = turn.get("content"), turn.get("original_content"), turn.get("trace_id")
    if not isinstance(content, str):
        return None
    return TurnView(
        role=turn["role"],
        content=mask_numbers(content),
        original_content=mask_numbers(original) if isinstance(original, str) else None,
        trace_id=trace_id if isinstance(trace_id, str) else None,
    )


def to_view(case: StoredCase) -> Optional[CaseView]:
    raw = case.raw
    turns = raw.get("turns")
    views = [view for view in map(_turn, turns if isinstance(turns, list) else []) if view is not None]
    if not views:
        return None
    other_topic = raw.get("other_topic")
    label = case.label or {}
    return CaseView(
        id=case.id,
        received_at=case.received_at,
        source=_text(raw, "source"),
        submitted_by=_text(raw, "submitted_by"),
        persona=_text(raw, "persona"),
        language=_text(raw, "language"),
        description=mask_numbers(_text(raw, "description")),
        case_types=_ids(raw, "case_types"),
        tags=_ids(raw, "tags"),
        other_topic=mask_numbers(other_topic) if isinstance(other_topic, str) else None,
        turns=views,
        group=_text(label, "group") or None,
        answer_type=_text(label, "answer_type") or None,
    )


@dataclass(frozen=True)
class CaseFilter:
    """Tags, case types, groups and answer types match when a case has any of them.
    The text search is case-insensitive over the description, the topic and every
    turn, as masked."""

    tags: frozenset[str] = frozenset()
    case_types: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()
    answer_types: frozenset[str] = frozenset()
    source: Optional[str] = None
    persona: Optional[str] = None
    language: Optional[str] = None
    submitted_by: Optional[str] = None
    text: Optional[str] = None

    def matches(self, view: CaseView) -> bool:
        if self.tags and self.tags.isdisjoint(view.tags):
            return False
        if self.case_types and self.case_types.isdisjoint(view.case_types):
            return False
        if self.groups and view.group not in self.groups:
            return False
        if self.answer_types and view.answer_type not in self.answer_types:
            return False
        for field in ("source", "persona", "language", "submitted_by"):
            wanted = getattr(self, field)
            if wanted and getattr(view, field) != wanted:
                return False
        if self.text:
            needle = self.text.casefold()
            parts = [view.description, view.other_topic or ""]
            for turn in view.turns:
                parts += [turn.content, turn.original_content or ""]
            if not any(needle in part.casefold() for part in parts):
                return False
        return True


def _counted(options: Iterable[HintOption], counts: Counter) -> list[CountedOption]:
    listed = [CountedOption(id=option.id, label=option.label, count=counts[option.id]) for option in options]
    known = {option.id for option in listed}
    return listed + [
        CountedOption(id=other, label=other, count=count)
        for other, count in sorted(counts.items())
        if other not in known
    ]


def coverage(views: list[CaseView]) -> CaseCoverage:
    tags = Counter(tag for view in views for tag in set(view.tags))
    case_types = Counter(case_type for view in views for case_type in set(view.case_types))
    groups = Counter(view.group for view in views if view.group)
    return CaseCoverage(
        total=len(views),
        untagged=sum(1 for view in views if not view.tags),
        tags=_counted(CASE_OPTIONS.tags, tags),
        case_types=_counted(CASE_OPTIONS.case_types, case_types),
        groups=[CountedOption(id=group, label=group, count=groups[group]) for group in GROUPS]
        + [CountedOption(id=other, label=other, count=count) for other, count in sorted(groups.items()) if other not in GROUPS],
        unlabelled=sum(1 for view in views if not view.group),
    )
