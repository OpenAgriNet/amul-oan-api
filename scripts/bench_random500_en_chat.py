#!/usr/bin/env python3
"""Call local /api/chat for N random EN questions; write CSV + report."""
import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import jwt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="GoldenSet CSV with question_en")
    p.add_argument("--out-dir", default="/home/azureuser/goldenset_tests/20260407")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jwt-key", default="/home/azureuser/amul-oan-api/jwt_private_key.pem")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/api/chat/")
    args = p.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"gemma4_en_en_random{args.n}_{ts}.csv"
    out_md = out_dir / f"gemma4_en_en_random{args.n}_{ts}_report.md"

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
    with src.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("question_en") or "").strip()
            if q:
                rows.append(r)

    n = min(args.n, len(rows))
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, n) if len(rows) >= n else rows

    def one(idx: int, r: dict) -> dict:
        q = (r.get("question_en") or "").strip()
        params = {
            "query": q,
            "source_lang": "en",
            "target_lang": "en",
            "user_id": "gemma4-bench",
            "use_translation_pipeline": "false",
            "session_id": f"bench-{ts}-{idx}-{r.get('row_id', '')}",
        }
        url = args.base_url + "?" + urlencode(params)
        req = Request(url, headers={"Authorization": f"Bearer {token}"})
        t0 = time.time()
        try:
            with urlopen(req, timeout=args.timeout) as resp:
                status = str(resp.status)
                body = resp.read().decode("utf-8", errors="replace").strip()
            err = ""
        except Exception as e:
            status = ""
            body = ""
            err = str(e)
        return {
            "row_id": r.get("row_id", ""),
            "question_en": q[:500],
            "http_status": status,
            "latency_ms": int((time.time() - t0) * 1000),
            "response_len": len(body),
            "error": err,
            "ok": "yes" if (status == "200" and not err and body) else "no",
        }

    results_typed: list[dict] = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, i, r): i for i, r in enumerate(sampled)}
        done = 0
        for fut in as_completed(futs):
            results_typed.append(fut.result())
            done += 1
            if done % 50 == 0:
                print(f"progress {done}/{len(sampled)}", flush=True)

    elapsed = time.time() - t_start
    ok_count = sum(1 for x in results_typed if x["ok"] == "yes")
    err_count = len(results_typed) - ok_count
    empty_body = sum(
        1
        for x in results_typed
        if x["http_status"] == "200" and not x["error"] and not x["response_len"]
    )
    latencies = [x["latency_ms"] for x in results_typed if x["ok"] == "yes"]

    fields = ["row_id", "question_en", "http_status", "latency_ms", "response_len", "ok", "error"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in results_typed:
            w.writerow({k: x[k] for k in fields})

    err_types: dict[str, int] = {}
    for x in results_typed:
        if x["ok"] != "yes":
            key = (x["error"] or "no_body_or_bad_http")[:120]
            err_types[key] = err_types.get(key, 0) + 1

    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Gemma random {n} EN->EN run ({ts})\n\n")
        f.write(f"- Source: `{src}`\n")
        f.write(f"- Sampled: {len(results_typed)} (workers={args.workers}, timeout={args.timeout}s")
        if args.seed is not None:
            f.write(f", seed={args.seed}")
        f.write(")\n")
        f.write(f"- Output: `{out_csv}`\n\n")
        f.write("## Results\n")
        f.write(f"- OK (HTTP 200, non-empty body): {ok_count}\n")
        f.write(f"- Errors / incomplete: {err_count}\n")
        f.write(f"- Empty body despite 200: {empty_body}\n")
        f.write(f"- Wall time: {elapsed:.1f}s\n")
        if latencies:
            f.write(
                f"- Latency ms (OK only): min={min(latencies)}, max={max(latencies)}, "
                f"avg={sum(latencies) / len(latencies):.0f}\n"
            )
        f.write("\n## Error breakdown (top)\n")
        for k, v in sorted(err_types.items(), key=lambda kv: -kv[1])[:15]:
            f.write(f"- {v}x: {k}\n")

    print(
        json.dumps(
            {
                "out_csv": str(out_csv),
                "out_md": str(out_md),
                "n": len(results_typed),
                "ok": ok_count,
                "err": err_count,
                "wall_s": round(elapsed, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
