# Langfuse answer-collection harness — how to run (vm5 + tmux)

**What this is:** an offline **answer-collection harness** for human review.
It calls the dev chat HTTP API, waits for Langfuse traces, and exports CSV/JSON rows.

**What this is not:** an automated evaluator. There is **no judge, rubric, or pass/fail** on answer quality. A row with `status=success` only means "an answer came back without a transport/trace error" — medically wrong or hallucinated answers still export as successful collection rows.

Run these scripts **on the AI backend VM** (not your laptop). The batch runner hits the local chat API (`http://127.0.0.1:8000`) and reads traces from Langfuse.

## 1. One-time setup on the AI backend VM

Clone the OpenAgriNet repo (or your fork), check out `eval-pipeline` (or `main` once merged), create a venv, and install requirements.

```bash
# example paths — use your team's checkout location
cd ~/amul-oan-api-langfuse-eval
git checkout eval-pipeline   # or main after merge

python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Copy `.env` from the main app checkout (or create one) with at least:

```bash
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
# LANGFUSE_TRACING_ENVIRONMENT=chat-development
# LANGFUSE_TIMEOUT=60
# EVAL_CHAT_BASE_URL=http://127.0.0.1:8000
# EVAL_CHAT_JWT=<dev JWT from token-for-phone>
# — or —
# EVAL_CHAT_API_KEY=<dev key>   # never commit real keys
# EVAL_CHAT_USER_PHONE=9876543210   # synthetic test phone only
```

**Prerequisite:** the chat API container (`amul_app`) must be listening on port 8000 on that VM.

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
./venv/bin/python evals/test_langfuse_connection.py
```

## 2. SSH from your laptop

Do **not** copy host IPs, usernames, or `ProxyJump` blocks into this repo.

Use the **internal Amul SSH / runbook** (team docs) for jump-host and AI-backend host aliases, then:

```bash
ssh <ai-backend-host-alias>
```

## 3. Golden-set dedup (optional prep)

Place the confidential set at `evals/GoldenSet.csv` (gitignored — get from the team).

```bash
cd ~/amul-oan-api-langfuse-eval
./venv/bin/pip install rapidfuzz sentence-transformers scikit-learn

# Needs Gemma/vLLM URL in .env (INFERENCE_ENDPOINT_URL or OSS_INFERENCE_ENDPOINT_URL)
./venv/bin/python evals/deduplicate_golden_set.py evals/GoldenSet.csv \
  --output-dir eval_outputs/golden_dedup
```

Outputs under `eval_outputs/golden_dedup/`:

| File | Description |
|------|-------------|
| `GoldenSet_deduped.csv` | Kept questions after merge stages |
| `GoldenSet_duplicates_report.csv` | Applied merges only |
| `GoldenSet_merge_audit.csv` | Full audit (merges, kept-distinct, judge errors, cap skips) |
| `GoldenSet_review.csv` | Human-review candidates (non-merged / errors / cap skips) |

Treat auto-merge thresholds as **unaudited** until someone reviews `GoldenSet_merge_audit.csv`. Prefer keeping the canonical 801-Q set for collection runs unless dedup is explicitly accepted.

## 4. Run golden-set collection in tmux

```bash
cd ~/amul-oan-api-langfuse-eval
chmod +x evals/run_golden_set_eval_tmux.sh
sed -i 's/\r$//' evals/run_golden_set_eval_tmux.sh   # if cloned from Windows
./evals/run_golden_set_eval_tmux.sh
```

**Monitor:**

```bash
tmux attach -t eval-golden          # Ctrl+B then D to detach
tail -f eval_outputs/golden_set_eval.log
ls eval_outputs/golden_langfuse_raw_json/query_*.json | wc -l
```

**Outputs:**

| File | Description |
|------|-------------|
| `eval_outputs/golden_set_eval_full.csv` | Shareable collection CSV (tools, latencies, traces, row_id, category) |
| `eval_outputs/golden_langfuse_raw_json/query_XXXX.json` | Full trace JSON per query |
| `eval_outputs/golden_set_eval.log` | Run log |

## 5. Resume after interrupt

Prefer the highest completed index (gaps possible if a mid-run file is missing):

```bash
NEXT=$(ls eval_outputs/golden_langfuse_raw_json/query_*.json \
  | sed 's/.*query_//;s/\.json//' \
  | sort -n \
  | tail -1)
NEXT=$((NEXT + 1))

./venv/bin/python evals/batch_langfuse_queries.py \
  eval_outputs/golden_dedup/GoldenSet_deduped.csv \
  --shareable-csv eval_outputs/golden_set_eval_full.csv \
  --raw-json-dir eval_outputs/golden_langfuse_raw_json \
  --csv-columns full \
  --max-index 679 \
  --start-from "$NEXT" \
  --langfuse-wait 180
```

Or rebuild CSV only (no re-run):

```bash
./venv/bin/python evals/rebuild_golden_csv.py --format full
```

## 6. Smaller test run (`queries.csv`)

```bash
./venv/bin/python evals/batch_langfuse_queries.py evals/queries.csv --limit 3   # smoke test
chmod +x evals/run_retry_failed_tmux.sh
./evals/run_retry_failed_tmux.sh   # retry failed rows in tmux (not never-run indices)
```

## 7. Download results to laptop

```bash
scp <ai-backend-host-alias>:~/amul-oan-api-langfuse-eval/eval_outputs/golden_set_eval_full.csv .
```
