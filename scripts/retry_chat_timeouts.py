#!/usr/bin/env python3
"""Retry failed rows from a bench CSV; first pass timeout T1, second pass T2."""
import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import jwt


def load_questions_by_row(golden_csv: Path) -> dict[str, str]:
    by_id: dict[str, str] = {}
    with golden_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rid = (r.get("row_id") or "").strip()
            q = (r.get("question_en") or "").strip()
            if rid and q:
                by_id[rid] = q
    return by_id


def call_chat(
    base_url: str,
    token: str,
    question: str,
    session_id: str,
    timeout: int,
) -> tuple[str, str, int, str]:
    params = {
        "query": question,
        "source_lang": "en",
        "target_lang": "en",
        "user_id": "gemma4-retry",
        "use_translation_pipeline": "false",
        "session_id": session_id,
    }
    url = base_url + "?" + urlencode(params)
    req = Request(url, headers={"Authorization": f"Bearer {token}"})
    t0 = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = str(resp.status)
            body = resp.read().decode("utf-8", errors="replace").strip()
        err = ""
    except Exception as e:
        status = ""
        body = ""
        err = str(e)
    latency = int((time.time() - t0) * 1000)
    return status, body, latency, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-csv", required=True)
    ap.add_argument("--golden-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--jwt-key", default="/home/azureuser/amul-oan-api/jwt_private_key.pem")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/api/chat/")
    ap.add_argument("--timeout-first", type=int, default=180)
    ap.add_argument("--timeout-second", type=int, default=300)
    args = ap.parse_args()

    bench = Path(args.bench_csv)
    golden = Path(args.golden_csv)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    qs = load_questions_by_row(golden)
    key = Path(args.jwt_key).read_text()
    payload = {
        "phone": "1111111111",
        "sub": "1111111111",
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400,
        "aud": "oan-ui-service",
        "iss": "mh-oan-api",
    }
    token = jwt.encode(payload, key, algorithm="RS256")

    rows: list[dict] = []
    with bench.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    failures = [r for r in rows if r.get("ok") != "yes"]
    results_log: list[dict] = []

    ts = int(time.time())
    for i, r in enumerate(failures):
        rid = (r.get("row_id") or "").strip()
        q = qs.get(rid) or (r.get("question_en") or "").strip()
        if not q:
            results_log.append({"row_id": rid, "phase": "skip", "note": "no_question"})
            continue

        sid1 = f"retry1-{ts}-{rid}-{i}"
        st, body, lat, err = call_chat(args.base_url, token, q, sid1, args.timeout_first)
        phase = "retry_180"
        if st == "200" and body and not err:
            results_log.append(
                {"row_id": rid, "phase": phase, "http_status": st, "latency_ms": lat, "error": ""}
            )
            # patch main row
            for row in rows:
                if (row.get("row_id") or "").strip() == rid:
                    row["http_status"] = st
                    row["latency_ms"] = str(lat)
                    row["response_len"] = str(len(body))
                    row["ok"] = "yes"
                    row["error"] = ""
                    break
            continue

        sid2 = f"retry2-{ts}-{rid}-{i}"
        st2, body2, lat2, err2 = call_chat(args.base_url, token, q, sid2, args.timeout_second)
        phase = "retry_300"
        ok = st2 == "200" and body2 and not err2
        results_log.append(
            {
                "row_id": rid,
                "phase": phase,
                "http_status": st2,
                "latency_ms": lat2,
                "error": err2 or "",
                "after_first_error": err[:80] if err else "",
            }
        )
        for row in rows:
            if (row.get("row_id") or "").strip() == rid:
                if ok:
                    row["http_status"] = st2
                    row["latency_ms"] = str(lat2)
                    row["response_len"] = str(len(body2))
                    row["ok"] = "yes"
                    row["error"] = ""
                else:
                    row["error"] = f"retry_failed: first={err[:60]!r} second={err2[:60]!r}"
                break

    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok_n = sum(1 for r in rows if r.get("ok") == "yes")
    still_bad = [r for r in rows if r.get("ok") != "yes"]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Retry report\n\n")
        f.write(f"- Input bench: `{bench}`\n")
        f.write(f"- Output CSV: `{out_csv}`\n")
        f.write(f"- Failures before retry: {len(failures)}\n")
        f.write(f"- OK after retry: {ok_n} / {len(rows)}\n")
        f.write(f"- Still failing: {len(still_bad)}\n")
        f.write(f"- Timeouts: first={args.timeout_first}s, second={args.timeout_second}s\n\n")
        if still_bad:
            f.write("## Still failing row_id\n")
            for r in still_bad:
                f.write(f"- {r.get('row_id')}: {r.get('error', '')[:120]}\n")

    print(json.dumps({"out_csv": str(out_csv), "ok": ok_n, "n": len(rows), "still_bad": len(still_bad)}, indent=2))


if __name__ == "__main__":
    main()
