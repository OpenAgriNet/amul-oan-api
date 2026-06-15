# Langfuse eval pipeline — how to run (vm5 + tmux)

Run eval scripts **on vm5** (not your laptop). The script calls the dev chat API on the same machine (`http://127.0.0.1:8000`), then reads traces from Langfuse.

## 1. One-time setup on vm5

```bash
git clone https://github.com/nexus69420/amul-oan-api.git ~/amul-oan-api-langfuse-eval
cd ~/amul-oan-api-langfuse-eval
git checkout eval-pipeline

python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Copy .env from main amul-oan-api repo or create one with at least:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
#   LANGFUSE_TRACING_ENVIRONMENT=chat-development
#   LANGFUSE_TIMEOUT=60
#   EVAL_CHAT_BASE_URL=http://127.0.0.1:8000
#   EVAL_CHAT_JWT=<dev JWT from token-for-phone>
```

**Prerequisite:** `amul_app` Docker container must be running on vm5 (port 8000). That is the chat API the eval hits.

```bash
# optional check
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
./venv/bin/python evals/test_langfuse_connection.py
```

## 2. SSH from your laptop

Add to `~/.ssh/config` (Windows: `C:\Users\<you>\.ssh\config`):

```
Host amul-vm3-uintele
    HostName 4.187.152.138
    User azureuser
    IdentityFile ~/.ssh/id_ed25519

Host amul-vm5-ai-backend
    HostName 10.5.25.36
    User azureuser
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump amul-vm3-uintele
```

Connect:

```bash
ssh amul-vm5-ai-backend
```

## 3. Golden set dedup (801 → 679 questions)

Place `GoldenSet.csv` at `evals/GoldenSet.csv` (not in git — get from team).

```bash
cd ~/amul-oan-api-langfuse-eval
./venv/bin/pip install rapidfuzz sentence-transformers scikit-learn

# Needs Gemma/vLLM URL in .env (INFERENCE_ENDPOINT_URL or OSS_INFERENCE_ENDPOINT_URL)
./venv/bin/python evals/deduplicate_golden_set.py evals/GoldenSet.csv \
  --output-dir eval_outputs/golden_dedup
```

Output: `eval_outputs/golden_dedup/GoldenSet_deduped.csv`

## 4. Run golden set eval in tmux (679 queries)

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
| `eval_outputs/golden_set_eval_full.csv` | Team CSV (tools, latencies, traces, row_id, category) |
| `eval_outputs/golden_langfuse_raw_json/query_XXXX.json` | Full trace JSON per query |
| `eval_outputs/golden_set_eval.log` | Run log |

## 5. Resume after interrupt

If the server stops mid-run, count completed JSONs and resume:

```bash
N=$(ls eval_outputs/golden_langfuse_raw_json/query_*.json | wc -l)
NEXT=$((N + 1))

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

## 6. Smaller test run (queries.csv, 200 queries)

```bash
./venv/bin/python evals/batch_langfuse_queries.py evals/queries.csv --limit 3   # smoke test
chmod +x evals/run_retry_failed_tmux.sh
./evals/run_retry_failed_tmux.sh   # retry failed rows in tmux
```

## 7. Download results to laptop

```bash
scp amul-vm5-ai-backend:~/amul-oan-api-langfuse-eval/eval_outputs/golden_set_eval_full.csv .
```
