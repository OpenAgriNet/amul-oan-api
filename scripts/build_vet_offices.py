"""Turn the AH Department office sheet into assets/vet_offices.json.

Source: "Offices of Dept of AH Gujarat State.xlsx" (Dept of Animal Husbandry,
Gujarat). Columns: Sr.No. | Office Name | Village | Taluka | District |
Category (English) | Office Type (Gujarati).

The sheet is a lookup table, not prose, so it is NOT indexed in Marqo — 3.2k
near-identical rows ("Veterinary Dispensary, <village>, <taluka>") embed to
almost the same vector and location matching is a filter, not a similarity. The chat
agent reads this JSON directly (agents/tools/vet_offices.py).

This script only transcribes. The sheet is the department's, not ours, so every
place name is carried across exactly as written — including the variant
spellings (Kutch/Kachchh, Dohad/Dahod, BHAVNAGAR) and the outright typos
(Vadpdara). Only surrounding whitespace is trimmed, which changes no name.
Treating those variants as the same place is a lookup-time decision and lives in
agents/tools/vet_offices.py, where it can be read and corrected without
rewriting the department's data.

Usage:
    python scripts/build_vet_offices.py "path/to/Offices of Dept of AH Gujarat State.xlsx"
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

OUTPUT = Path("assets/vet_offices.json")


def _clean(value) -> str:
    """Trim surrounding and repeated whitespace. Spelling is left untouched."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def build(source: Path) -> dict:
    # data_only: Category and Office Type are XLOOKUP formulas against the
    # sheet's second tab; without it openpyxl hands back the formula object.
    sheet = load_workbook(source, read_only=True, data_only=True).worksheets[0]

    offices = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not row or not any(row):
            continue
        _, office_name, village, taluka, district, category, office_type_gu = (
            list(row) + [None] * 7
        )[:7]
        district = _clean(district)
        taluka = _clean(taluka)
        if not district and not taluka:
            continue  # nothing to locate it by
        offices.append(
            {
                "name": _clean(office_name),
                "village": _clean(village),
                "taluka": taluka,
                "district": district,
                "category": _clean(category),
                "office_type_gu": _clean(office_type_gu),
            }
        )

    offices.sort(key=lambda o: (o["district"], o["taluka"], o["village"], o["category"]))
    return {
        "source": source.name,
        "office_count": len(offices),
        "offices": offices,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source = Path(sys.argv[1])
    if not source.exists():
        print(f"No such sheet: {source}")
        return 1

    data = build(source)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    categories = Counter(o["category"] for o in data["offices"])
    print(f"Wrote {OUTPUT} — {data['office_count']} offices")
    print(f"  district spellings: {len({o['district'] for o in data['offices']})}")
    print(f"  taluka spellings:   {len({o['taluka'] for o in data['offices']})}")
    for category, count in categories.most_common(6):
        print(f"    {count:>5}  {category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
