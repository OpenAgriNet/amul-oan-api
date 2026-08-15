#!/usr/bin/env python
"""Probe every district coordinate in agents/tools/districts.py against Bharat Vistaar.

**Why this is a script and not a test.** When the coordinate table was written
on 2026-08-13, BV was down — their BAP answered `responses: []` in ~9 s and both
legs timed out — so the newly added coordinates could not be validated live.
Run this against a healthy target and promote rows **on evidence**.

What it does
------------
For each district, for each candidate coordinate, for each commodity, it issues a
real mandi search through the same code path the tool uses (seeker by default)
and reports the rows and markets returned. It then prints, per district:

  * which candidate answered first, and how far down the list it was;
  * the markets/districts/states actually returned — cross-district and even
    cross-state results are NORMAL for a ~50 km radius search, so the report
    prints them rather than judging them;
  * a suggested `verified=True` set, which is a suggestion, not an edit.

⚠️ Read before believing the output
-----------------------------------
* **A zero-row cell can be correct.** That commodity, that market, that day.
  South Gujarat (Surat, Navsari, Valsad, Tapi, Dang) is genuinely grain-empty —
  zero rows for Wheat/Bajra/Castor/Groundnut at *any* coordinate. Do not "fix" a
  coordinate that is answering correctly.
* **The `vistaar` leg flaps**, and has produced three wrong root-cause
  conclusions in this codebase. `--repeat 2` (the default) probes every cell
  twice; treat a cell whose row count, returned markets, or error state disagrees
  with itself as unknown, not as a result.
* **Bharuch and Narmada resolve to Sendhwa APMC, Madhya Pradesh** — an upstream
  stored-coordinate bug ~2° of longitude off, not something a better coordinate
  here can fix.
* Each probe costs ~2.2 s. The retained table is 52 candidates × 5 commodities
  × 2 repeats = 520 probes ≈ 20 minutes. Pace it; the upstream is a
  single non-redundant sandbox.
* **Archive the output somewhere that is not `/tmp` on VM5** — that box reboots
  nightly at ~01:31 UTC and wipes it. A previous 388-probe sweep was lost
  exactly that way and had to be reconstructed from a document. When `--json`
  is set, this script checkpoints after every candidate so an interrupted run
  retains its completed cells.

Usage
-----
    .venv/bin/python scripts/validate_district_coords.py                  # full sweep
    .venv/bin/python scripts/validate_district_coords.py --districts kutch banaskantha
    .venv/bin/python scripts/validate_district_coords.py --commodities Onion Wheat
    .venv/bin/python scripts/validate_district_coords.py --json out.json  # machine-readable
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tools import vistaar  # noqa: E402
from agents.tools.districts import DISTRICTS  # noqa: E402

# Five commodities with different market coverage, on purpose: market selection
# is PER-COMMODITY (Junagadh + Wheat answers from Junagadh APMC, Junagadh +
# Cotton from Jetpur APMC in Rajkot district), so a single commodity tells you
# almost nothing about a coordinate.
# Onion retains the known vegetable control; Wheat/Bajra/Castor/Groundnut are
# all required for the Deesa-vs-Palanpur decision. Do not drop Bajra here: an
# earlier four-commodity draft accidentally omitted it even though the handover
# explicitly makes it part of the highest-stakes validation cell.
DEFAULT_COMMODITIES = ["Onion", "Wheat", "Bajra", "Castor", "Groundnut"]


async def probe(commodity: str, lat: float, lon: float) -> dict:
    # The tool's own default window (7 days). Wider buys nothing: the BPP caps
    # every response at 10 rows, so 7, 15 and 30 days return identical data.
    from_date, to_date = vistaar._resolve_date_range(None, None)
    intent = {
        "category": {"descriptor": {"code": "price-discovery"}},
        "item": {"descriptor": {"name": commodity}},
        "fulfillment": {"end": {"location": {"gps": f"{lat},{lon}"}}},
        "tags": [
            {"code": "from_date", "value": from_date},
            {"code": "to_date", "value": to_date},
        ],
    }
    started = time.monotonic()
    try:
        items = await vistaar._vistaar_search(intent)
    except vistaar.VistaarLegUnavailable as exc:
        return {
            "error": f"leg_unavailable: {exc}",
            "elapsed_s": round(time.monotonic() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 2),
        }
    markets = sorted({
        ", ".join(
            part for part in (
                vistaar._tag_values(i).get("Market"),
                vistaar._tag_values(i).get("District"),
                vistaar._tag_values(i).get("State"),
            ) if part
        )
        for i in items
    })
    return {
        "rows": len(items),
        "markets": markets,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def _result_signature(run: dict) -> tuple:
    """Comparable evidence, excluding latency noise."""
    if "error" in run:
        return ("error", run["error"])
    return ("ok", run.get("rows"), tuple(run.get("markets", [])))


def _checkpoint(path: Path, results: dict) -> None:
    """Atomically preserve completed cells without leaving partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(results, indent=2) + "\n")
    pending.replace(path)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--districts", nargs="*", default=sorted(DISTRICTS))
    parser.add_argument("--commodities", nargs="*", default=DEFAULT_COMMODITIES)
    parser.add_argument("--repeat", type=int, default=2,
                        help="probes per cell; the vistaar leg flaps, so 1 is not evidence")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between probes")
    parser.add_argument("--json", type=Path, default=None, help="write full results here")
    args = parser.parse_args()

    unknown = [d for d in args.districts if d not in DISTRICTS]
    if unknown:
        parser.error(f"unknown district key(s): {unknown}. Known: {sorted(DISTRICTS)}")

    print(f"seeker={vistaar.VISTAAR_SEEKER_URL or '(direct BAP)'} leg={vistaar.VISTAAR_LEG}")
    print(f"{len(args.districts)} districts x {len(args.commodities)} commodities "
          f"x {args.repeat} repeats\n")

    results: dict = {}
    answering_towns: set[str] = set()

    for key in args.districts:
        location = DISTRICTS[key]
        print(f"── {location.display} ({key})")
        results[key] = {}
        for candidate in location.candidates:
            cell: dict = {}
            for commodity in args.commodities:
                runs = []
                for _ in range(args.repeat):
                    runs.append(await probe(commodity, candidate.lat, candidate.lon))
                    await asyncio.sleep(args.pause)
                rows = [r.get("rows") for r in runs]
                agree = len({_result_signature(r) for r in runs}) == 1
                cell[commodity] = {"runs": runs, "stable": agree}
                markets = sorted({m for r in runs for m in r.get("markets", [])})
                errors = [r["error"] for r in runs if "error" in r]
                flag = "" if agree else "  ⚠ UNSTABLE — treat as unknown"
                if errors:
                    print(f"   {candidate.town:<16} {commodity:<11} ERROR {errors[0]}")
                else:
                    print(f"   {candidate.town:<16} {commodity:<11} rows={rows}{flag}")
                    for market in markets:
                        print(f"       -> {market}")
                    if any(r for r in rows if r) and agree:
                        answering_towns.add(candidate.town)
            results[key][candidate.town] = {
                "lat": candidate.lat, "lon": candidate.lon,
                "verified_before_run": candidate.verified, "commodities": cell,
            }
            if args.json:
                _checkpoint(args.json, results)

    print("\n── Suggested verified=True set (stable, non-zero on at least one commodity)")
    print("   ⚠ A suggestion, not an edit. A zero-row town may still be a correct")
    print("     coordinate (south Gujarat trades no cereals anywhere).")
    for town in sorted(answering_towns):
        print(f"   {town}")

    if args.json:
        _checkpoint(args.json, results)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
