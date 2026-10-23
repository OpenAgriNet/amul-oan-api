"""Simple statistics over milk collection slips, computed in code.

The milk tools used to return slips only, so every total the farmer heard was
the model's own arithmetic, and on prod chat it was wrong in 80% of the answers
that stated one (29 Sep-12 Oct 2026). Both the chat and the voice milk tools now
put these figures in front of the slips, and the prompts say to quote them.

Deductions are summed with the sign each society records: a negative row is
either a reversal of an earlier entry or that society's convention, so no
"amount after deductions" is derived here.
"""
from dataclasses import dataclass, field
from math import fsum

SHIFT_LABELS = {"M": "morning", "E": "evening"}


@dataclass
class MilkSummary:
    slips: int = 0
    days: int = 0
    qty: float = 0.0
    amount: float = 0.0
    shift_slips: dict[str, int] = field(default_factory=dict)
    shift_qty: dict[str, float] = field(default_factory=dict)
    shift_amount: dict[str, float] = field(default_factory=dict)
    fat: float | None = None
    snf: float | None = None
    rate: float | None = None
    qty_per_day: float | None = None
    highest_day: tuple[str, float] | None = None
    lowest_day: tuple[str, float] | None = None
    incomplete_slips: int = 0
    deduction_entries: int = 0
    deductions: float = 0.0


def shift_label(shift: str | None) -> str:
    return SHIFT_LABELS.get((shift or "").strip().upper(), shift or "unknown")


def _weighted(records, attr: str) -> float | None:
    """Average of ``attr`` weighted by litres, as a dairy averages FAT and SNF."""
    pairs = [(getattr(r, attr), r.qty) for r in records if getattr(r, attr) is not None and r.qty]
    litres = fsum(q for _, q in pairs)
    return fsum(v * q for v, q in pairs) / litres if litres else None


def summarize_milk(milk, deductions) -> MilkSummary:
    """Totals, morning/evening split, litre-weighted FAT/SNF, rate and best/worst day.

    Works on any records with ``date/shift/qty/fat/snf/amount`` (chat and voice
    keep separate model classes). Pass every account's records together for a
    combined summary: a day's litres then add up across accounts.
    """
    summary = MilkSummary(slips=len(milk))
    per_day: dict[str, float] = {}
    for r in milk:
        if r.qty is None or r.amount is None:
            summary.incomplete_slips += 1
        shift = shift_label(r.shift)
        summary.shift_slips[shift] = summary.shift_slips.get(shift, 0) + 1
        summary.shift_qty[shift] = summary.shift_qty.get(shift, 0.0) + (r.qty or 0)
        summary.shift_amount[shift] = summary.shift_amount.get(shift, 0.0) + (r.amount or 0)
        per_day[r.date or "unknown"] = per_day.get(r.date or "unknown", 0.0) + (r.qty or 0)

    summary.qty = fsum(r.qty or 0 for r in milk)
    summary.amount = fsum(r.amount or 0 for r in milk)
    poured = {day: qty for day, qty in per_day.items() if qty > 0}
    summary.days = len(poured)
    if poured:
        summary.qty_per_day = summary.qty / len(poured)
    if len(set(poured.values())) > 1:
        summary.highest_day = max(poured.items(), key=lambda item: item[1])
        summary.lowest_day = min(poured.items(), key=lambda item: item[1])
    summary.fat = _weighted(milk, "fat")
    summary.snf = _weighted(milk, "snf")
    if summary.qty:
        summary.rate = summary.amount / summary.qty

    summary.deduction_entries = len(deductions)
    summary.deductions = fsum(d.amount or 0 for d in deductions)
    return summary
